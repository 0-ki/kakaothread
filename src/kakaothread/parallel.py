"""세션 병렬 실행기 (LangGraph 순차 루프의 asyncio 대안).

세션(큰 시간 공백 경계)끼리는 열린 스레드가 리셋되어 완전히 독립이므로
동시에 처리할 수 있다. 세션 내부 청크는 open_threads 캐리 때문에 순차.

- 전역 thread_id: 공유 IdAlloc 발급 → 세션 간 충돌 없음 (오프셋 reconcile 불필요).
  asyncio 는 단일 스레드라 '분류 결과 통합(integrate)+상태 갱신'이 await 없이
  한 번에 실행되는 한 락이 필요 없다.
- 동시성 상한: Semaphore (동시 활성 세션 수). 실제 처리율은 슬롯 페이싱이
  지배하므로, 병렬화의 이득은 LLM 응답 대기가 겹치는 부분이다.
- 체크포인트: 청크가 끝날 때마다 checkpoint.json 기록 → 중단돼도 --resume 으로
  이어서 실행. 완료되면 체크포인트는 삭제된다.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from . import config, janitor
from .cost import UsageTracker
from .llm_segment import Thread, classify_chunk_async, reset_cache
from .pipeline import (
    Chunk,
    IdAlloc,
    PipelineOptions,
    RunResult,
    domain_context,
    integrate,
    load_input,
)
from .preprocess import Message
from .provider_pool import ProviderPool

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint.json"
CKPT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    """원본 파일 식별용 해시 (재개 검증·중복 실행 감지)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class _Ctx:
    """세션 워커들이 공유하는 실행 상태."""
    pool: ProviderPool | None
    tracker: UsageTracker
    examples: str
    alloc: IdAlloc
    sem: asyncio.Semaphore
    vocab: list[str] | None = None  # 고정 택소노미 (None=자유 모드)
    all_threads: dict[int, Thread] = field(default_factory=dict)
    assignments: dict[int, int] = field(default_factory=dict)
    done: dict[int, int] = field(default_factory=dict)        # sidx -> 완료 청크 수(prefix)
    open_map: dict[int, list[Thread]] = field(default_factory=dict)  # 미완료 세션 캐리
    pbar: tqdm | None = None
    ckpt_path: Path | None = None
    ckpt_meta: dict = field(default_factory=dict)  # source/hash/max_chunks 등 고정 필드


# ── 체크포인트 ──────────────────────────────────────────────────────
def _save_checkpoint(ctx: _Ctx) -> None:
    data = {
        "version": CKPT_VERSION,
        **ctx.ckpt_meta,
        "examples": ctx.examples,
        "category_vocab": ctx.vocab,
        "alloc_next": ctx.alloc.next,
        "done": {str(k): v for k, v in ctx.done.items()},
        "open_threads": {str(k): [t.model_dump() for t in v] for k, v in ctx.open_map.items()},
        "all_threads": {str(k): t.model_dump() for k, t in ctx.all_threads.items()},
        "assignments": {str(k): v for k, v in ctx.assignments.items()},
    }
    tmp = ctx.ckpt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ctx.ckpt_path)  # 원자적 교체 (중단돼도 파일이 깨지지 않게)


def load_checkpoint(run_dir: str | Path) -> dict:
    path = Path(run_dir) / CHECKPOINT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {path} (이미 완료된 run 이거나 병렬 모드로 실행되지 않음)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CKPT_VERSION:
        raise ValueError(f"체크포인트 버전 불일치: {data.get('version')} != {CKPT_VERSION}")
    return data


def checkpoint_source(run_dir: str | Path) -> str | None:
    """run_dir 의 체크포인트가 가리키는 원본 sha256 (없으면 None). 재개 정합 판단용."""
    path = Path(run_dir) / CHECKPOINT_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("source_sha256")
    except (OSError, json.JSONDecodeError):
        return None


def delete_checkpoint(run_dir: str | Path) -> None:
    """산출물 저장이 성공한 뒤 호출 — 체크포인트를 정리한다.

    러너 안에서 삭제하지 않는 이유: 분류 완료 후 save_and_report(호출부) 도중
    죽으면 결과가 유실되는데, 체크포인트가 남아 있으면 재개로 복구할 수 있다.
    (재개 시 모든 청크가 done 이라 재분류 없이 janitor+저장만 다시 수행 — 저렴)
    """
    path = Path(run_dir) / CHECKPOINT_NAME
    if path.exists():
        path.unlink()


def _restore(ctx: _Ctx, data: dict) -> None:
    """체크포인트 내용을 ctx 에 복원."""
    ctx.examples = data.get("examples", "")
    ctx.vocab = data.get("category_vocab")
    ctx.alloc.next = data.get("alloc_next", 1)
    ctx.done = {int(k): v for k, v in data.get("done", {}).items()}
    ctx.open_map = {int(k): [Thread(**t) for t in v]
                    for k, v in data.get("open_threads", {}).items()}
    ctx.all_threads = {int(k): Thread(**t) for k, t in data.get("all_threads", {}).items()}
    ctx.assignments = {int(k): v for k, v in data.get("assignments", {}).items()}


def options_from_checkpoint(data: dict) -> PipelineOptions:
    """체크포인트에 저장된 실행 조건을 복원 (재개 시 동일 전처리 보장)."""
    return PipelineOptions(
        max_chunks=data.get("max_chunks"),
        use_domain=False,  # examples/vocab 은 체크포인트에서 복원되므로 재탐색 불필요
        anonymize=data.get("anonymize", False),
    )


# ── 병렬 분류 ───────────────────────────────────────────────────────
async def _session_worker(sidx: int, chunks: list[list[Message]], ctx: _Ctx) -> None:
    """한 세션의 청크들을 순서대로 분류 (열린 스레드 캐리)."""
    async with ctx.sem:
        open_threads = ctx.open_map.get(sidx, [])
        for i in range(ctx.done.get(sidx, 0), len(chunks)):
            result = await classify_chunk_async(
                chunks[i], open_threads, ctx.pool, ctx.tracker,
                examples=ctx.examples, category_vocab=ctx.vocab)
            # 여기부터 await 없음 → 통합+상태 갱신이 원자적 (asyncio 단일 스레드)
            open_threads, ctx.all_threads, assigns = integrate(
                result, open_threads, ctx.all_threads, ctx.alloc)
            ctx.assignments.update(assigns)
            ctx.done[sidx] = i + 1
            ctx.open_map[sidx] = open_threads
            if ctx.pbar is not None:
                ctx.pbar.update(1)
            if ctx.ckpt_path is not None:
                _save_checkpoint(ctx)
        # 세션 완주 — 캐리 상태는 더 필요 없음 (체크포인트 슬림화)
        ctx.open_map.pop(sidx, None)


async def _classify_all(sessions: dict[int, list[list[Message]]], ctx: _Ctx) -> None:
    await asyncio.gather(*(_session_worker(s, cs, ctx) for s, cs in sorted(sessions.items())))


def group_sessions(chunks: list[Chunk]) -> dict[int, list[list[Message]]]:
    sessions: dict[int, list[list[Message]]] = {}
    for sidx, chunk in chunks:
        sessions.setdefault(sidx, []).append(chunk)
    return sessions


def classify_parallel(chunks: list[Chunk], ctx: _Ctx) -> None:
    """세션 병렬 분류 실행 (진행바 관리 포함). ctx 가 그대로 결과를 담는다."""
    sessions = group_sessions(chunks)
    # 새 이벤트 루프 시작 전 LLM 캐시를 비운다 (이전 asyncio.run 의 죽은 루프에 묶인
    # async 클라이언트 재사용 방지 — 한 프로세스에서 여러 잡을 처리하는 worker 등).
    reset_cache()
    ctx.pbar = tqdm(total=len(chunks), initial=sum(ctx.done.values()), desc="segment(parallel)")
    try:
        asyncio.run(_classify_all(sessions, ctx))
    finally:
        ctx.pbar.close()


def consolidate_safe(all_threads: dict[int, Thread], pool, tracker
                     ) -> tuple[dict[int, Thread], dict[str, str]]:
    """janitor 호출 — 실패해도 원본 유지 (순차/병렬/증분 공용)."""
    try:
        return janitor.consolidate(all_threads, pool, tracker)
    except Exception as e:  # noqa: BLE001
        logger.warning("정리부 실패 — 병합 없이 진행: %s", e)
        return all_threads, {}


def run_parallel(
    path: str,
    opts: PipelineOptions = PipelineOptions(),
    *,
    concurrency: int = 4,
    run_dir: Path | None = None,
    resume_data: dict | None = None,
    pool: ProviderPool | None = None,
) -> RunResult:
    """세션 병렬 실행. run_dir 을 주면 checkpoint.json 을 기록/정리한다.

    resume_data: load_checkpoint() 결과 — 완료된 청크는 건너뛰고 이어서 실행.
    """
    slot_desc = ", ".join(f"{s.name}({s.model}, p{s.priority})" for s in config.SLOTS)
    logger.info("병렬 분류 시작: %s (동시 세션 %d, 슬롯=[%s])", path, concurrency, slot_desc)
    messages, chunks, chunks_total = load_input(path, opts)
    pool = pool or ProviderPool(config.SLOTS)
    tracker = UsageTracker()

    ctx = _Ctx(pool=pool, tracker=tracker, examples="", alloc=IdAlloc(1),
               sem=asyncio.Semaphore(max(1, concurrency)))
    if resume_data is not None:
        src_hash = file_sha256(path)
        if resume_data.get("source_sha256") not in (None, src_hash):
            raise ValueError("원본 파일이 체크포인트 생성 시점과 다릅니다 (sha256 불일치).")
        _restore(ctx, resume_data)
        n_done = sum(ctx.done.values())
        logger.info("재개: 청크 %d/%d 완료 상태에서 이어서 실행", n_done, len(chunks))
    else:
        ctx.examples, ctx.vocab = domain_context(messages, pool, tracker, opts)

    if run_dir is not None:
        ctx.ckpt_path = Path(run_dir) / CHECKPOINT_NAME
        ctx.ckpt_meta = {
            "source": str(path),
            "source_sha256": (resume_data or {}).get("source_sha256") or file_sha256(path),
            "max_chunks": opts.max_chunks,
            "anonymize": opts.anonymize,  # 재개 시 동일 전처리 보장 (등장 순서 기반이라 결정적)
        }

    classify_parallel(chunks, ctx)

    # 정리부(janitor): 흔들린 category 이름을 한 번에 통일 (LLM 1콜). 실패해도 원본 유지.
    all_threads, merge_map = consolidate_safe(ctx.all_threads, pool, tracker)

    # 체크포인트는 여기서 지우지 않는다 — 호출부가 save_and_report 성공 후 delete_checkpoint 호출.

    logger.info(
        "병렬 분류 완료: 스레드 %d개, 토큰 in=%d out=%d, 비용 ~$%.4f",
        len(all_threads), tracker.tok_in, tracker.tok_out, tracker.cost,
    )
    return RunResult(messages, all_threads, ctx.assignments, tracker, merge_map,
                     len(chunks), chunks_total, opts.anonymize)
