"""잡 시스템 테스트 — 제출/실행/증분/실패 격리/취소/크래시 복구."""
import json
from pathlib import Path

import pytest

from kakaothread import jobs
from kakaothread.cost import UsageTracker
from kakaothread.incremental import NothingToUpdate
from kakaothread.jobs import Job, cancel, execute, list_jobs, normalize_job_id, submit, worker
from kakaothread.pipeline import RunResult


def _result() -> RunResult:
    return RunResult(messages=[], threads={}, assignments={}, tracker=UsageTracker(),
                     merge_map={}, chunks_done=1, chunks_total=1)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """잡 스토어/런 폴더를 tmp 로 돌리고 러너·저장을 가짜로 대체."""
    calls = {"parallel": [], "incremental": [], "saved": 0, "runs": 0}
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")

    def fake_new_run_dir():
        calls["runs"] += 1
        d = tmp_path / "runs" / f"r{calls['runs']}"
        d.mkdir(parents=True)
        return d

    def fake_parallel(source, opts=None, **kw):
        calls["parallel"].append({"source": source, **kw})
        return _result()

    def fake_incremental(source, prev, opts=None, **kw):
        calls["incremental"].append({"source": source, "prev": prev, **kw})
        return _result()

    monkeypatch.setattr(jobs, "new_run_dir", fake_new_run_dir)
    monkeypatch.setattr(jobs, "run_parallel", fake_parallel)
    monkeypatch.setattr(jobs, "run_incremental", fake_incremental)
    monkeypatch.setattr(jobs, "save_and_report", lambda *a, **k: calls.__setitem__(
        "saved", calls["saved"] + 1))

    src = tmp_path / "등산모임 채팅.txt"
    src.write_text("2026년 5월 11일 오전 9:00, 철수 : ㅎㅇ\n", encoding="utf-8")
    return calls, src


def test_normalize_job_id():
    assert normalize_job_id("등산모임 채팅") == "등산모임-채팅"
    assert normalize_job_id("  My Room!! ") == "my-room"
    assert normalize_job_id("!!!") == "job"


def test_submit_and_worker_full_run(env):
    calls, src = env
    job = submit(str(src))
    assert job.job_id == "등산모임-채팅"
    assert job.status == "pending" and job.source_sha256

    assert worker() == 1
    job = jobs.load_job(job.job_id)
    assert job.status == "done"
    assert job.last_run_dir and len(job.history) == 1
    assert len(calls["parallel"]) == 1 and calls["saved"] == 1


def test_resubmit_runs_incremental(env):
    calls, src = env
    submit(str(src))
    worker()
    first_run = jobs.load_job("등산모임-채팅").last_run_dir

    submit(str(src))  # 같은 방 재제출 → 증분 경로
    assert worker() == 1
    job = jobs.load_job("등산모임-채팅")
    assert job.status == "done"
    assert len(calls["incremental"]) == 1
    assert calls["incremental"][0]["prev"] == first_run
    assert len(job.history) == 2


def test_nothing_new_marks_done_without_new_run(env, monkeypatch):
    calls, src = env
    submit(str(src))
    worker()
    first_run = jobs.load_job("등산모임-채팅").last_run_dir

    def raise_nothing(*a, **k):
        raise NothingToUpdate("새 메시지 없음")

    monkeypatch.setattr(jobs, "run_incremental", raise_nothing)
    submit(str(src))
    worker()
    job = jobs.load_job("등산모임-채팅")
    assert job.status == "done"
    assert job.last_run_dir == first_run  # 새 run 없음 — 기존 결과 유지
    assert len(job.history) == 1


def test_failure_isolated_and_recorded(env, monkeypatch):
    calls, src = env

    def boom(*a, **k):
        raise RuntimeError("LLM 전멸")

    monkeypatch.setattr(jobs, "run_parallel", boom)
    submit(str(src))
    worker()
    job = jobs.load_job("등산모임-채팅")
    assert job.status == "failed"
    assert "LLM 전멸" in job.error
    assert jobs.next_pending() is None  # failed 는 자동 재시도 안 함


def test_cancel_pending(env):
    calls, src = env
    submit(str(src), job_id="취소방")
    job = cancel("취소방")
    assert job.status == "cancelled"
    assert worker() == 0  # cancelled 는 실행되지 않음


def test_crash_recovery_resumes_from_checkpoint(env):
    """running 중 죽은 잡: run_dir 에 체크포인트가 남아 있으면 재개 경로를 탄다."""
    calls, src = env
    job = submit(str(src))
    # 크래시 시뮬레이션: run_dir 에 체크포인트만 남기고 상태를 pending 으로 재제출
    crash_dir = Path(jobs.new_run_dir())
    (crash_dir / "checkpoint.json").write_text(json.dumps({
        "version": 1, "source": str(src), "source_sha256": job.source_sha256,
        "max_chunks": None, "anonymize": False,
        "examples": "", "category_vocab": None, "alloc_next": 5,
        "done": {"0": 1}, "open_threads": {}, "all_threads": {}, "assignments": {},
    }), encoding="utf-8")
    job.run_dir = str(crash_dir)
    jobs.save_job(job)

    worker()
    job = jobs.load_job(job.job_id)
    assert job.status == "done"
    # 재개 경로: run_parallel 이 resume_data 를 받고, run_dir 는 기존 폴더 재사용
    assert calls["parallel"][0]["resume_data"]["alloc_next"] == 5
    assert calls["parallel"][0]["run_dir"] == crash_dir


def test_submit_reclaims_stale_running(env):
    """running 으로 고착된 잡을 재제출하면 거부가 아니라 pending 으로 회수한다."""
    calls, src = env
    job = submit(str(src))
    job.status = "running"
    jobs.save_job(job)
    job2 = submit(str(src))  # 예외 없이 재큐잉
    assert job2.status == "pending"


def test_worker_reclaims_stale_running(env):
    """worker 시작 시 크래시로 running 에 남은 잡을 pending 으로 회수해 실행한다."""
    calls, src = env
    job = submit(str(src))
    job.status = "running"  # 크래시 시뮬레이션 (pending 으로 되돌리지 않음)
    jobs.save_job(job)
    assert worker() == 1
    assert jobs.load_job(job.job_id).status == "done"


def test_stale_checkpoint_for_different_source_not_resumed(env, tmp_path):
    """다른 원본의 옛 체크포인트가 남아 있어도 재개하지 않고 전체/증분으로 처리한다.

    (같은 경로 덮어쓰기로 해시가 바뀌면 옛 체크포인트 재개는 영구 실패하고,
     다른 파일 재제출이면 옛 데이터로 무단 done 처리되는 것을 방지)
    """
    calls, src = env
    job = submit(str(src))
    # 다른 해시를 가진 체크포인트가 job.run_dir 에 남아 있는 상황을 구성
    stale_dir = Path(jobs.new_run_dir())
    (stale_dir / "checkpoint.json").write_text(json.dumps({
        "version": 1, "source": "old.txt", "source_sha256": "DIFFERENT_HASH",
        "max_chunks": None, "anonymize": False,
        "examples": "", "category_vocab": None, "alloc_next": 9,
        "done": {}, "open_threads": {}, "all_threads": {}, "assignments": {},
    }), encoding="utf-8")
    job.run_dir = str(stale_dir)
    jobs.save_job(job)

    worker()
    # 옛 체크포인트를 재개하지 않고(=resume_data 없이) 전체 실행 경로를 탄다
    assert len(calls["parallel"]) == 1
    assert calls["parallel"][0].get("resume_data") is None
    assert jobs.load_job(job.job_id).status == "done"


def test_list_jobs_skips_corrupt_files(env, tmp_path):
    calls, src = env
    submit(str(src))
    (jobs.JOBS_DIR / "broken.json").write_text("{not json", encoding="utf-8")
    assert len(list_jobs()) == 1
