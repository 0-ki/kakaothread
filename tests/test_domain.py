"""도메인 탐색·택소노미 테스트 — 방 설명 주입, 고정 어휘 추출/규칙."""
from datetime import datetime, timedelta

from kakaothread import domain, llm_segment
from kakaothread.domain import DomainProfile, ExamplePair, category_vocab, discover
from kakaothread.llm_segment import (
    FIXED_CATEGORY_RULE,
    FREE_CATEGORY_RULE,
    _category_rule,
    _chunk_prompt,
)
from kakaothread.preprocess import Message


def _msgs(n: int) -> list[Message]:
    base = datetime(2026, 5, 11, 9, 0)
    return [Message(msg_id=i, dt=base + timedelta(minutes=i), sender="a", text=f"메시지 {i}")
            for i in range(n)]


PROFILE = DomainProfile(domain="등산 동호회 정보방", examples=[
    ExamplePair(category="코스", topic="주말 산행지 추천"),
    ExamplePair(category="장비", topic="등산화 고르기"),
    ExamplePair(category="코스", topic="야간 산행 준비"),   # 중복 category
    ExamplePair(category=" 모임 ", topic="정기 모임 일정"),  # 공백 트림
])


def test_category_vocab_dedup_and_order():
    assert category_vocab(PROFILE) == ["코스", "장비", "모임"]


def test_discover_injects_room_desc(monkeypatch):
    captured = {}

    def fake_invoke(pool, prompt, schema, tracker=None, **kw):
        captured["prompt"] = prompt
        return PROFILE

    monkeypatch.setattr(domain, "invoke_structured", fake_invoke)
    profile = discover(_msgs(5), pool=None, room_desc="동네 등산 동호회 모임방")
    assert profile is PROFILE
    assert "[방 설명 (사용자 제공)]" in captured["prompt"]
    assert "동네 등산 동호회 모임방" in captured["prompt"]

    discover(_msgs(5), pool=None)  # 설명 없으면 블록도 없음
    assert "[방 설명" not in captured["prompt"]


def test_discover_empty_messages_returns_none():
    assert discover([], pool=None) is None


def test_category_rule_free_vs_fixed():
    assert _category_rule(None) == FREE_CATEGORY_RULE
    fixed = _category_rule(["코스", "장비"])
    assert '"코스", "장비"' in fixed
    assert "기타" in fixed


def test_chunk_prompt_contains_fixed_vocab():
    prompt = _chunk_prompt(_msgs(2), [], examples=None, category_vocab=["코스", "장비"])
    assert "반드시 다음 고정 목록" in prompt
    assert '"코스", "장비"' in prompt
    # 자유 모드 문구는 없어야 함
    assert "자연스럽게 뽑으세요" not in prompt

    free = _chunk_prompt(_msgs(2), [], examples=None, category_vocab=None)
    assert "자연스럽게 뽑으세요" in free
    assert "고정 목록" not in free
