"""janitor(카테고리 정리부) 테스트 — 병합맵 해소와 적용."""
from kakaothread import janitor
from kakaothread.janitor import Merge, MergePlan, _resolve, consolidate
from kakaothread.llm_segment import Thread


def _threads(*cats: str) -> dict[int, Thread]:
    return {
        i + 1: Thread(thread_id=i + 1, category=c, topic=f"t{i}", summary="")
        for i, c in enumerate(cats)
    }


def test_resolve_chain():
    assert _resolve("a", {"a": "b", "b": "c"}) == "c"


def test_resolve_cycle_terminates():
    assert _resolve("a", {"a": "b", "b": "a"}) in {"a", "b"}


def test_consolidate_applies_merges(monkeypatch):
    plan = MergePlan(merges=[Merge(from_category="취업 정보", to_category="취업")])
    monkeypatch.setattr(janitor, "invoke_structured", lambda *a, **k: plan)

    threads = _threads("취업", "취업 정보", "세금")
    merged, mapping = consolidate(threads, pool=None)

    assert mapping == {"취업 정보": "취업"}
    assert {t.category for t in merged.values()} == {"취업", "세금"}
    # thread_id/topic 은 보존
    assert merged[2].topic == "t1"


def test_consolidate_ignores_unknown_and_selfmap(monkeypatch):
    plan = MergePlan(merges=[
        Merge(from_category="세금", to_category="세금"),      # 자기 자신 → 무시
        Merge(from_category="없는범주", to_category="세금"),  # 목록에 없음 → 무시
    ])
    monkeypatch.setattr(janitor, "invoke_structured", lambda *a, **k: plan)

    threads = _threads("취업", "세금")
    merged, mapping = consolidate(threads, pool=None)
    assert mapping == {}
    assert merged is threads


def test_consolidate_skips_single_category(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("category 1개면 LLM 호출 없어야 함")

    monkeypatch.setattr(janitor, "invoke_structured", boom)
    threads = _threads("취업", "취업")
    merged, mapping = consolidate(threads, pool=None)
    assert merged is threads and mapping == {}
