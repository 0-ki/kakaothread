"""설정 로더 테스트 — .env 슬롯 구성과 Cerebras 기본 프리셋."""
from kakaothread.config import (
    DEFAULT_CEREBRAS_MODELS,
    _default_cerebras_slots,
    _load_slots,
)


def _clear_llm_env(monkeypatch):
    import os

    for k in list(os.environ):
        if k.startswith("LLM_") or k == "CEREBRAS_API_KEY":
            monkeypatch.delenv(k, raising=False)


def test_default_preset_from_cerebras_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test")
    slots = _load_slots()
    assert [s.model for s in slots] == list(DEFAULT_CEREBRAS_MODELS)
    assert {s.priority for s in slots} == {1}  # 전부 같은 티어 → 라운드로빈
    assert all(s.rpm == 5 for s in slots)


def test_no_key_no_slots(monkeypatch):
    _clear_llm_env(monkeypatch)
    assert _load_slots() == []
    assert _default_cerebras_slots() == []


def test_explicit_slots_override_preset(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test")
    monkeypatch.setenv("LLM_CONN_MY_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_CONN_MY_API_KEY", "key")
    monkeypatch.setenv("LLM_SLOTS", "mine")
    monkeypatch.setenv("LLM_SLOT_MINE_CONN", "MY")
    monkeypatch.setenv("LLM_SLOT_MINE_MODEL", "some-model")
    monkeypatch.setenv("LLM_SLOT_MINE_PRIORITY", "2")
    monkeypatch.setenv("LLM_SLOT_MINE_RPD", "50")
    slots = _load_slots()
    assert [(s.name, s.model, s.priority, s.rpd) for s in slots] == [
        ("mine", "some-model", 2, 50)
    ]
