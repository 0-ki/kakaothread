"""잡 시스템 — '방 1개 = job 1개'로 처리 이력을 관리하는 파일 기반 잡 스토어.

외부 브로커(Celery/Redis) 없이 로컬에서 쓰는 최소 구현:
  - data/jobs/<job_id>.json 파일 하나가 잡 하나의 상태 (원자적 교체로 저장)
  - submit  → pending 큐잉 (같은 방을 다시 제출하면 기존 잡을 재큐잉)
  - worker  → pending 잡을 제출 순서대로 하나씩 실행 (단일 worker 전제)
  - 실행은 항상 병렬 러너 + 체크포인트:
      · 이전 실행이 중단됐으면(체크포인트 존재) 이어서 재개
      · 이 방의 완료 run 이 있으면 증분 처리 (새 구간만 분류)
      · 처음이면 전체 실행
  - cancel  → pending 잡 취소 (running 은 프로세스 중단 = 체크포인트로 재개 가능)

동시성 주의: 파일 락이 없으므로 worker 는 한 프로세스만 돌린다는 전제.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .incremental import NothingToUpdate, run_incremental
from .outputs import new_run_dir, save_and_report
from .parallel import (
    CHECKPOINT_NAME,
    checkpoint_source,
    delete_checkpoint,
    file_sha256,
    load_checkpoint,
    options_from_checkpoint,
    run_parallel,
)
from .pipeline import PipelineOptions

logger = logging.getLogger(__name__)

JOBS_DIR = Path("data/jobs")

# 잡 상태 전이: pending → running → done | failed,  pending → cancelled
STATUSES = ("pending", "running", "done", "failed", "cancelled")


@dataclass
class Job:
    job_id: str
    source: str                 # 가장 최근 제출된 원본 파일 경로
    source_sha256: str = ""
    status: str = "pending"
    options: dict = field(default_factory=dict)  # PipelineOptions 필드 + concurrency
    run_dir: str = ""           # 현재/직전 실행 폴더 (체크포인트 재개 기준)
    last_run_dir: str = ""      # 마지막 '완료' run (증분 처리의 이전 기준)
    submitted_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    history: list = field(default_factory=list)  # 완료된 run 기록


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_job_id(name: str) -> str:
    """파일명/방이름 → 안전한 잡 id (소문자, 공백→-, 경로 위험 문자 제거)."""
    name = re.sub(r"\s+", "-", name.strip().lower())
    name = re.sub(r"[^0-9a-z가-힣_-]", "", name)
    return name or "job"


def _path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def save_job(job: Job) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(job.job_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _path(job.job_id))


def load_job(job_id: str) -> Job:
    data = json.loads(_path(job_id).read_text(encoding="utf-8"))
    return Job(**data)


def list_jobs() -> list[Job]:
    if not JOBS_DIR.exists():
        return []
    out = []
    for p in sorted(JOBS_DIR.glob("*.json")):
        try:
            out.append(Job(**json.loads(p.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, TypeError):
            logger.warning("잡 파일 파싱 실패 — 건너뜀: %s", p)
    return out


def submit(source: str, job_id: str | None = None, options: dict | None = None) -> Job:
    """잡 제출/재제출. 같은 job_id 면 기존 이력을 유지한 채 pending 으로 재큐잉.

    단일 worker 전제라, 'running' 상태로 남은 잡은 이전 worker 가 비정상 종료해
    고착된 것으로 보고 재큐잉을 허용한다 (경고). 체크포인트가 남아 있으면
    execute 가 원본 해시를 대조해 이어서 실행한다.
    """
    jid = normalize_job_id(job_id or Path(source).stem)
    if _path(jid).exists():
        job = load_job(jid)
        if job.status == "running":
            logger.warning("잡 '%s' 이(가) running 상태로 남아 있음 — 고착된 잡으로 보고 재큐잉 "
                           "(worker 가 실제 실행 중이면 중복 실행 금지).", jid)
    else:
        job = Job(job_id=jid, source=str(source))
    job.source = str(source)
    job.source_sha256 = file_sha256(source)
    job.options = options or job.options
    job.status = "pending"
    job.submitted_at = _now()
    job.error = ""
    save_job(job)
    logger.info("잡 제출: %s (%s)", jid, source)
    return job


def cancel(job_id: str) -> Job:
    job = load_job(job_id)
    if job.status == "pending":
        job.status = "cancelled"
        job.finished_at = _now()
        save_job(job)
        logger.info("잡 취소: %s", job_id)
    elif job.status == "running":
        logger.warning("잡 %s 은(는) 실행 중 — worker 프로세스를 중단하면 "
                       "체크포인트가 남아 재제출 시 이어서 실행됩니다.", job_id)
    else:
        logger.info("잡 %s 은(는) 이미 %s 상태", job_id, job.status)
    return job


def _pipeline_options(job: Job) -> PipelineOptions:
    o = job.options
    return PipelineOptions(
        use_domain=o.get("use_domain", True),
        room_desc=o.get("room_desc", ""),
        fixed_taxonomy=o.get("fixed_taxonomy", False),
        anonymize=o.get("anonymize", False),
    )


def _decide_mode(job: Job) -> str:
    """실행 모드 판단: 'resume' | 'incremental' | 'full'.

    재개는 '직전 run_dir 의 체크포인트가 지금 처리할 원본과 같은 해시'일 때만 한다.
    (다른 원본을 재제출했는데 옛 체크포인트를 재개하면 새 파일이 무단 폐기되거나,
     같은 경로를 덮어써 해시가 바뀌면 재개가 영구 실패한다 — 둘 다 방지)
    """
    if job.run_dir:
        ck_hash = checkpoint_source(job.run_dir)
        if ck_hash is not None and ck_hash == job.source_sha256:
            return "resume"
    return "incremental" if job.last_run_dir else "full"


def execute(job: Job) -> Job:
    """잡 하나 실행 — 재개/증분/전체를 자동 판단. 상태 전이와 산출물 저장까지."""
    concurrency = int(job.options.get("concurrency", 4))
    mode = _decide_mode(job)
    run_dir = Path(job.run_dir) if mode == "resume" else new_run_dir()

    job.status = "running"
    job.started_at = _now()
    job.run_dir = str(run_dir)
    job.error = ""
    save_job(job)

    t0 = time.perf_counter()
    try:
        if mode == "resume":
            ckpt = load_checkpoint(run_dir)
            source = ckpt["source"]
            src_hash = ckpt.get("source_sha256")
            logger.info("잡 %s: 체크포인트 재개 (%s)", job.job_id, run_dir)
            if "prefix" in ckpt:
                res = run_incremental(source, ckpt.get("prev_run", ""), resume_data=ckpt,
                                      run_dir=run_dir, concurrency=concurrency)
            else:
                res = run_parallel(source, options_from_checkpoint(ckpt), resume_data=ckpt,
                                   run_dir=run_dir, concurrency=concurrency)
        else:
            source = job.source
            src_hash = job.source_sha256 or file_sha256(source)
            opts = _pipeline_options(job)
            if mode == "incremental":
                logger.info("잡 %s: 증분 처리 (이전 run: %s)", job.job_id, job.last_run_dir)
                res = run_incremental(source, job.last_run_dir, opts,
                                      run_dir=run_dir, concurrency=concurrency)
            else:
                logger.info("잡 %s: 전체 실행", job.job_id)
                res = run_parallel(source, opts, run_dir=run_dir, concurrency=concurrency)
    except NothingToUpdate as e:
        logger.info("잡 %s: %s", job.job_id, e)
        # 새 run 폴더를 만들었는데 처리할 게 없으면 빈 폴더를 치우고 이전 완료 run 유지
        if mode != "resume" and not any(run_dir.iterdir()):
            run_dir.rmdir()
        job.run_dir = job.last_run_dir
        job.status = "done"
        job.finished_at = _now()
        save_job(job)
        return job
    except Exception as e:  # noqa: BLE001 — 잡 단위 실패 격리 (worker 는 다음 잡 진행)
        job.status = "failed"
        job.error = f"{type(e).__name__}: {e}"
        job.finished_at = _now()
        save_job(job)
        logger.exception("잡 %s 실패", job.job_id)
        return job

    elapsed = time.perf_counter() - t0
    note = (f"증분 실행 — 이전 run({job.last_run_dir})의 배정을 재사용"
            if mode == "incremental" else None)
    save_and_report(res.messages, res.threads, res.assignments, run_dir, source=source,
                    tracker=res.tracker, merge_map=res.merge_map, elapsed=elapsed,
                    chunks_done=res.chunks_done, chunks_total=res.chunks_total,
                    source_sha256=src_hash, anonymize=res.anonymize, note=note)
    delete_checkpoint(run_dir)  # 산출물 저장 성공 후에만 정리 (실패 시 재개로 복구 가능)
    job.status = "done"
    job.finished_at = _now()
    job.last_run_dir = str(run_dir)
    job.history.append({
        "run_dir": str(run_dir), "finished_at": job.finished_at,
        "n_threads": len(res.threads), "n_messages": len(res.messages),
        "cost": res.tracker.cost,
    })
    save_job(job)
    logger.info("잡 %s 완료 → %s", job.job_id, run_dir)
    return job


def next_pending() -> Job | None:
    pending = [j for j in list_jobs() if j.status == "pending"]
    if not pending:
        return None
    return min(pending, key=lambda j: j.submitted_at)


def reclaim_stale() -> int:
    """이전 worker 가 비정상 종료해 'running' 으로 남은 잡을 pending 으로 되돌린다.

    단일 worker 전제 — worker 시작 시점에 running 인 잡은 실제로 도는 게 아니라
    직전 크래시의 잔재다. run_dir 은 보존하므로 execute 가 체크포인트에서 재개한다.
    """
    n = 0
    for j in list_jobs():
        if j.status == "running":
            j.status = "pending"
            save_job(j)
            logger.warning("고착된 running 잡 회수 → pending: %s", j.job_id)
            n += 1
    return n


def worker(once: bool = False) -> int:
    """pending 잡을 제출 순서대로 실행. 반환: 처리한 잡 수."""
    reclaim_stale()  # 크래시로 running 에 고착된 잡을 먼저 회수
    n = 0
    while True:
        job = next_pending()
        if job is None:
            break
        execute(job)
        n += 1
        if once:
            break
    return n
