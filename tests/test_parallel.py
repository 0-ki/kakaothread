"""세션 병렬 실행기 테스트 — 전역 id 유일성, 세션 내 캐리, 체크포인트/재개."""
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kakaothread import parallel
from kakaothread.cost import UsageTracker
from kakaothread.llm_segment import Assignment, ChunkResult, Thread
from kakaothread.parallel import (
    CHECKPOINT_NAME,
    _classify_all,
    _Ctx,
    _restore,
    file_sha256,
    load_checkpoint,
    run_parallel,
)
from kakaothread.pipeline import IdAlloc, PipelineOptions
from kakaothread.preprocess import Message


def _msgs(start: int, n: int, base_minute: int = 0) -> list[Message]:
    base = datetime(2026, 5, 11, 9, 0)
    return [Message(msg_id=start + i, dt=base + timedelta(minutes=base_minute + i),
                    sender="a", text=f"m{start + i}")
            for i in range(n)]


def make_fake(fail_at_call: int | None = None):
    """가짜 분류기: 청크마다 새 지역 스레드(id=7) 하나에 전부 배정."""
    calls = {"n": 0, "open_seen": []}

    async def fake(chunk, open_threads, pool, tracker, examples=None,
                   category_vocab=None, max_tries=6):
        calls["n"] += 1
        if fail_at_call is not None and calls["n"] == fail_at_call:
            raise RuntimeError("boom")
        calls["open_seen"].append((chunk[0].msg_id, sorted(t.thread_id for t in open_threads)))
        return ChunkResult(
            assignments=[Assignment(msg_id=m.msg_id, thread_id=7) for m in chunk],
            threads=[Thread(thread_id=7, category=f"cat{chunk[0].msg_id}", topic="t", summary="")],
        )

    return fake, calls


def _ctx(**kw) -> _Ctx:
    return _Ctx(pool=None, tracker=UsageTracker(), examples="", alloc=IdAlloc(1),
                sem=asyncio.Semaphore(4), **kw)


def test_parallel_sessions_unique_ids_and_carry(monkeypatch):
    fake, calls = make_fake()
    monkeypatch.setattr(parallel, "classify_chunk_async", fake)

    # 세션 2개 × 청크 2개
    sessions = {0: [_msgs(0, 3), _msgs(3, 3)], 1: [_msgs(6, 3), _msgs(9, 3)]}
    ctx = _ctx()
    asyncio.run(_classify_all(sessions, ctx))

    assert calls["n"] == 4
    # 전역 id 유일 (세션 간 충돌 없음): 청크마다 새 스레드 → 1..4
    assert sorted(ctx.all_threads) == [1, 2, 3, 4]
    # 모든 메시지 배정
    assert sorted(ctx.assignments) == list(range(12))
    # 캐리: 각 세션의 첫 청크는 빈 open, 둘째 청크는 이전 스레드 1개를 봤어야 함
    seen = dict(calls["open_seen"])
    assert seen[0] == [] and seen[6] == []
    assert len(seen[3]) == 1 and len(seen[9]) == 1


def test_checkpoint_resume(monkeypatch, tmp_path):
    sessions = {0: [_msgs(0, 2), _msgs(2, 2)], 1: [_msgs(4, 2), _msgs(6, 2)]}
    ckpt = tmp_path / CHECKPOINT_NAME

    # 1차: 3번째 호출(세션1 첫 청크)에서 실패 → 세션0(2청크)까지 체크포인트에 남음
    fake, calls = make_fake(fail_at_call=3)
    monkeypatch.setattr(parallel, "classify_chunk_async", fake)
    ctx = _ctx(ckpt_path=ckpt, ckpt_meta={"source": "x", "source_sha256": "h", "max_chunks": None})
    with pytest.raises(RuntimeError):
        asyncio.run(_classify_all(sessions, ctx))
    assert ckpt.exists()

    # 2차: 체크포인트 복원 후 이어서 — 남은 청크(2개)만 호출돼야 함
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    assert data["done"] == {"0": 2}
    fake2, calls2 = make_fake()
    monkeypatch.setattr(parallel, "classify_chunk_async", fake2)
    ctx2 = _ctx(ckpt_path=ckpt, ckpt_meta=ctx.ckpt_meta)
    _restore(ctx2, data)
    asyncio.run(_classify_all(sessions, ctx2))

    assert calls2["n"] == 2  # 완료분은 건너뜀
    assert sorted(ctx2.assignments) == list(range(8))
    assert sorted(ctx2.all_threads) == [1, 2, 3, 4]  # 재개 후에도 id 연속·유일


SAMPLE = """2026년 5월 11일 오전 9:00, 철수 : 첫 세션 이야기
2026년 5월 11일 오전 9:05, 영희 : 네 맞아요
2026년 5월 11일 오후 8:00, 철수 : 두번째 세션
2026년 5월 11일 오후 8:03, 영희 : 응답
"""


def test_run_parallel_end_to_end(monkeypatch, tmp_path):
    """파일 → 병렬 분류 → RunResult, 완료 시 체크포인트 삭제."""
    src = tmp_path / "chat.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    fake, calls = make_fake()
    monkeypatch.setattr(parallel, "classify_chunk_async", fake)

    res = run_parallel(str(src), PipelineOptions(use_domain=False), run_dir=tmp_path,
                       pool=object())  # janitor 는 pool 오류 → 병합 없이 진행

    assert res.chunks_done == res.chunks_total == 2  # 세션 2개 = 청크 2개
    assert sorted(res.assignments) == [0, 1, 2, 3]
    assert len(res.threads) == 2
    # 러너는 체크포인트를 지우지 않는다 — 호출부가 저장 성공 후 delete_checkpoint 로 정리
    assert (tmp_path / CHECKPOINT_NAME).exists()
    from kakaothread.parallel import delete_checkpoint
    delete_checkpoint(tmp_path)
    assert not (tmp_path / CHECKPOINT_NAME).exists()


def test_load_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path)


def test_file_sha256(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    h1 = file_sha256(p)
    assert file_sha256(p) == h1  # 결정적
    p.write_text("world", encoding="utf-8")
    assert file_sha256(p) != h1  # 내용 바뀌면 달라짐
