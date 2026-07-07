"""세션 분할 + 예산 청킹 테스트."""
from datetime import datetime, timedelta

from kakaothread.chunking import budget_chunks, make_session_chunks, split_sessions
from kakaothread.preprocess import Message


def _msgs(*minutes: int, text: str = "안녕") -> list[Message]:
    base = datetime(2026, 5, 11, 9, 0)
    return [
        Message(msg_id=i, dt=base + timedelta(minutes=m), sender="a", text=text)
        for i, m in enumerate(minutes)
    ]


def test_split_sessions_on_gap():
    msgs = _msgs(0, 10, 300, 310)  # 10→300 사이 290분 공백
    sessions = split_sessions(msgs, gap_minutes=180)
    assert [len(s) for s in sessions] == [2, 2]


def test_split_sessions_empty():
    assert split_sessions([]) == []


def test_budget_chunks_by_count():
    msgs = _msgs(*range(5))
    chunks = budget_chunks(msgs, char_budget=10_000, max_messages=2)
    assert [len(c) for c in chunks] == [2, 2, 1]


def test_budget_chunks_by_chars():
    msgs = _msgs(0, 1, 2, text="가" * 100)  # 건당 대략 109자
    chunks = budget_chunks(msgs, char_budget=250, max_messages=100)
    assert [len(c) for c in chunks] == [2, 1]


def test_budget_chunks_oversized_single_message():
    msgs = _msgs(0, text="가" * 1000)
    chunks = budget_chunks(msgs, char_budget=100, max_messages=10)
    assert [len(c) for c in chunks] == [1]  # 예산 초과라도 홀로 한 청크


def test_make_session_chunks_carries_session_index():
    msgs = _msgs(0, 10, 300, 310)
    out = make_session_chunks(msgs, gap_minutes=180, char_budget=10_000)
    assert [sidx for sidx, _ in out] == [0, 1]
