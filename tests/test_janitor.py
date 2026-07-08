"""janitor(카테고리 정리부) 테스트 — 병합맵 해소와 적용, 중복 스레드 병합."""
from kakaothread import janitor
from kakaothread.janitor import (
    Merge,
    MergePlan,
    _resolve,
    consolidate,
    merge_duplicate_threads,
)
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


def _dup_threads() -> dict[int, Thread]:
    # 같은 "종목 정보 > 삼성전자" 가 3개(id 1,3,5), 대표는 메시지 최다인 id 3
    return {
        1: Thread(thread_id=1, category="종목 정보", topic="삼성전자", summary="a"),
        2: Thread(thread_id=2, category="종목 정보", topic="SK하이닉스", summary="b"),
        3: Thread(thread_id=3, category="종목 정보", topic="삼성전자", summary="대표"),
        5: Thread(thread_id=5, category="종목 정보", topic=" 삼성전자 ", summary="c"),  # 공백만 다름
    }


def test_merge_duplicate_threads_collapses_same_label():
    threads = _dup_threads()
    # id3에 3건, id1에 1건, id5에 1건, id2에 2건
    assignments = {10: 1, 11: 3, 12: 3, 13: 3, 14: 5, 15: 2, 16: 2, 0: 0}
    merged, new_assign, n = merge_duplicate_threads(threads, assignments)

    assert n == 2  # id1, id5 가 id3 으로 흡수
    assert set(merged) == {2, 3}  # 대표(3=최다 배정) + 하이닉스(2) 만 남음
    assert merged[3].summary == "대표"  # 대표 스레드의 요약 유지
    # 삼성전자였던 모든 배정이 대표 id 3 으로
    assert new_assign[10] == 3 and new_assign[14] == 3
    assert new_assign[15] == 2  # 다른 topic 은 그대로
    assert new_assign[0] == 0   # 잡담은 불변


def test_merge_duplicate_threads_noop_when_unique():
    threads = _threads("A", "B")  # topic t0, t1 로 서로 다름
    assignments = {1: 1, 2: 2}
    merged, new_assign, n = merge_duplicate_threads(threads, assignments)
    assert n == 0 and merged is threads and new_assign is assignments
