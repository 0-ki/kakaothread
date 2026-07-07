"""증분 처리 테스트 — 공통 접두, 배정 재사용, 캐리 시드, 경계 병합, 재개."""
import json
from pathlib import Path

import pytest

from kakaothread import parallel
from kakaothread.incremental import (
    NothingToUpdate,
    common_prefix,
    run_incremental,
)
from kakaothread.llm_segment import Assignment, ChunkResult, Thread
from kakaothread.parallel import CHECKPOINT_NAME, load_checkpoint
from kakaothread.pipeline import PipelineOptions
from kakaothread.preprocess import dump_messages, merge_consecutive, parse_kakao

OLD = """2026년 5월 11일 오전 9:00, 철수 : 세션A 첫 이야기
2026년 5월 11일 오전 9:05, 영희 : 세션A 응답
2026년 5월 11일 오후 8:00, 철수 : 세션B 시작
2026년 5월 11일 오후 8:03, 영희 : 세션B 응답
"""
# 새 export = 기존 + 세션B 연속 발화 1개 + 다음날 새 세션 1개
NEW = OLD + """2026년 5월 11일 오후 8:05, 철수 : 세션B 추가 질문
2026년 5월 12일 오전 10:00, 영희 : 새 세션 시작
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _make_prev_run(tmp_path: Path, old_file: Path) -> Path:
    """이전 run 산출물(threads.json/messages.jsonl/meta.json)을 구성."""
    prev = tmp_path / "prev_run"
    prev.mkdir()
    msgs = merge_consecutive(parse_kakao(old_file))
    dump_messages(msgs, prev / "messages.jsonl")
    payload = [
        {"thread_id": 1, "category": "A", "topic": "a", "summary": "", "msg_ids": [0, 1]},
        {"thread_id": 2, "category": "B", "topic": "b", "summary": "", "msg_ids": [2, 3]},
    ]
    (prev / "threads.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (prev / "meta.json").write_text(json.dumps({"anonymize": False}), encoding="utf-8")
    return prev


def make_fake(fail_at_call: int | None = None):
    calls = {"n": 0, "open_seen": []}

    async def fake(chunk, open_threads, pool, tracker, examples=None,
                   category_vocab=None, max_tries=6):
        calls["n"] += 1
        if fail_at_call is not None and calls["n"] == fail_at_call:
            raise RuntimeError("boom")
        calls["open_seen"].append((chunk[0].msg_id, sorted(t.thread_id for t in open_threads)))
        return ChunkResult(
            assignments=[Assignment(msg_id=m.msg_id, thread_id=99) for m in chunk],
            threads=[Thread(thread_id=99, category=f"new{chunk[0].msg_id}", topic="t", summary="")],
        )

    return fake, calls


def test_common_prefix_basic(tmp_path):
    old = merge_consecutive(parse_kakao(_write(tmp_path, "old.txt", OLD)))
    new = merge_consecutive(parse_kakao(_write(tmp_path, "new.txt", NEW)))
    assert common_prefix(old, new) == len(old) == 4
    assert common_prefix(old, old) == 4
    assert common_prefix([], new) == 0


def test_incremental_reuses_prefix_and_carries(monkeypatch, tmp_path):
    old_file = _write(tmp_path, "old.txt", OLD)
    new_file = _write(tmp_path, "new.txt", NEW)
    prev = _make_prev_run(tmp_path, old_file)
    fake, calls = make_fake()
    monkeypatch.setattr(parallel, "classify_chunk_async", fake)

    res = run_incremental(str(new_file), prev, PipelineOptions(use_domain=False),
                          pool=object())

    # 접두 4개는 이전 배정 그대로
    assert {i: res.assignments[i] for i in range(4)} == {0: 1, 1: 1, 2: 2, 3: 2}
    # 새 구간: 세션B 연속(msg 4) + 새 세션(msg 5) → 청크 2개, 새 id는 3부터
    assert res.chunks_done == 2
    assert res.assignments[4] == 3 and res.assignments[5] == 4
    assert set(res.threads) == {1, 2, 3, 4}
    # 캐리 시드: msg4 청크는 이전 tail 세션 스레드(2)를 open 으로 받아야 함
    seen = dict(calls["open_seen"])
    assert seen[4] == [2]
    assert seen[5] == []  # 새 세션은 리셋


def test_incremental_restores_prefix_noise(monkeypatch, tmp_path):
    """접두 구간에서 실제 스레드에 없던 메시지는 잡담(0)으로 복원돼야 한다.

    (threads.json 에는 잡담이 없어, 복원 안 하면 증분 결과의 잡담 수가 전체 실행과 불일치)
    """
    old_file = _write(tmp_path, "old.txt", OLD)
    new_file = _write(tmp_path, "new.txt", NEW)
    prev = tmp_path / "prev_run"
    prev.mkdir()
    msgs = merge_consecutive(parse_kakao(old_file))
    dump_messages(msgs, prev / "messages.jsonl")
    # 이전 run: msg 0,1 만 스레드 1, msg 2,3 은 잡담(threads.json 에 없음)
    payload = [{"thread_id": 1, "category": "A", "topic": "a", "summary": "", "msg_ids": [0, 1]}]
    (prev / "threads.json").write_text(json.dumps(payload), encoding="utf-8")
    (prev / "meta.json").write_text(json.dumps({"anonymize": False}), encoding="utf-8")

    fake, _ = make_fake()
    monkeypatch.setattr(parallel, "classify_chunk_async", fake)
    res = run_incremental(str(new_file), prev, PipelineOptions(use_domain=False), pool=object())

    # 접두의 잡담(2,3)이 명시적으로 0 으로 복원됨
    assert res.assignments[2] == 0 and res.assignments[3] == 0
    assert res.assignments[0] == 1 and res.assignments[1] == 1


def test_incremental_nothing_new(monkeypatch, tmp_path):
    old_file = _write(tmp_path, "old.txt", OLD)
    prev = _make_prev_run(tmp_path, old_file)
    with pytest.raises(NothingToUpdate):
        run_incremental(str(old_file), prev, PipelineOptions(use_domain=False), pool=object())


def test_incremental_boundary_merge_reclassifies_last(monkeypatch, tmp_path):
    """이전 마지막 발화가 새 연속발화와 병합돼 달라지면 그 발화도 재분류."""
    old_file = _write(tmp_path, "old.txt", OLD)
    # 같은 분(오후 8:03)·같은 화자 → merge_consecutive 가 하나로 합침
    new_text = OLD + "2026년 5월 11일 오후 8:03, 영희 : 덧붙임\n"
    new_file = _write(tmp_path, "new.txt", new_text)
    prev = _make_prev_run(tmp_path, old_file)
    fake, calls = make_fake()
    monkeypatch.setattr(parallel, "classify_chunk_async", fake)

    res = run_incremental(str(new_file), prev, PipelineOptions(use_domain=False),
                          pool=object())

    # 접두는 3개까지만 — 마지막 병합 메시지(id 3)는 재분류돼 새 스레드로
    assert {i: res.assignments[i] for i in range(3)} == {0: 1, 1: 1, 2: 2}
    assert res.assignments[3] == 3  # 재분류 (id 3부터 새로 발급)
    assert res.messages[3].text == "세션B 응답 덧붙임"


def test_incremental_checkpoint_resume(monkeypatch, tmp_path):
    old_file = _write(tmp_path, "old.txt", OLD)
    new_file = _write(tmp_path, "new.txt", NEW)
    prev = _make_prev_run(tmp_path, old_file)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # 1차: 첫 분류 호출에서 실패 → 초기 체크포인트(접두 상태)는 이미 기록됨
    fake, _ = make_fake(fail_at_call=1)
    monkeypatch.setattr(parallel, "classify_chunk_async", fake)
    with pytest.raises(RuntimeError):
        run_incremental(str(new_file), prev, PipelineOptions(use_domain=False),
                        pool=object(), run_dir=run_dir)
    ckpt = load_checkpoint(run_dir)
    assert ckpt["prefix"] == 4 and ckpt["prev_run"] == str(prev)

    # 1차 실패 시 다른 세션 워커(세션1)는 자기 청크를 완료하고 체크포인트에 남김
    assert ckpt["done"] == {"1": 1}

    # 2차: 재개 — 실패했던 세션0 청크 1개만 처리하고 완료 후 체크포인트 정리
    fake2, calls2 = make_fake()
    monkeypatch.setattr(parallel, "classify_chunk_async", fake2)
    res = run_incremental(str(new_file), prev, resume_data=ckpt,
                          pool=object(), run_dir=run_dir)
    assert calls2["n"] == 1
    assert sorted(res.assignments) == [0, 1, 2, 3, 4, 5]
    # 러너는 체크포인트를 지우지 않는다 (호출부가 저장 후 정리) — 여전히 존재
    assert (run_dir / CHECKPOINT_NAME).exists()
