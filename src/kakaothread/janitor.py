"""분류가 끝난 뒤 흔들린 category 이름을 한 번에 통일하는 정리부(janitor).

메시지를 다시 읽지 않는다. 최종 category 목록(+스레드 수·대표 topic)만 보고
동의어/포함관계를 병합하는 규칙표를 받아 적용한다 → LLM 1콜, 값싸게.
예: "코스>주말 산행"과 "모임>주말 산행"으로 갈린 표류를 한 이름으로 모은다.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from .llm_segment import Thread, invoke_structured

logger = logging.getLogger(__name__)

NL = "\n"

# 잡담 전용 스레드 (병합 대상에서 제외). pipeline.NOISE_THREAD_ID 와 동일 값.
NOISE_THREAD_ID = 0


def merge_duplicate_threads(
    all_threads: dict[int, Thread], assignments: dict[int, int]
) -> tuple[dict[int, Thread], dict[int, int], int]:
    """같은 ``category > topic`` 인 스레드들을 하나로 병합하고 배정을 재매핑한다.

    분류가 세션 경계 리셋·모델 편차 탓에 같은 주제("삼성전자")를 여러 thread_id 로
    쪼갠 것을, 메시지를 다시 읽지 않고 **결정적으로(LLM 없이)** 사후 통합한다.
    category 통일(consolidate) 뒤에 부르면 표기가 정규화된 상태라 더 잘 합쳐진다.

    - 대표 스레드 = 그 라벨에 배정된 메시지가 가장 많은 스레드(동률이면 작은 id).
      대표의 topic/summary 를 유지한다.
    - 잡담(0)은 건드리지 않는다.
    반환: (병합된 all_threads, 재매핑된 assignments, 병합으로 사라진 스레드 수)
    """
    counts = Counter(assignments.values())  # thread_id -> 배정된 메시지 수
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for tid, t in all_threads.items():
        if tid == NOISE_THREAD_ID:
            continue
        groups[(t.category.strip(), t.topic.strip())].append(tid)

    remap: dict[int, int] = {}
    for tids in groups.values():
        rep = sorted(tids, key=lambda x: (-counts.get(x, 0), x))[0]
        for tid in tids:
            remap[tid] = rep

    n_merged = sum(1 for tid, rep in remap.items() if tid != rep)
    if not n_merged:
        return all_threads, assignments, 0

    merged = {tid: t for tid, t in all_threads.items() if remap.get(tid, tid) == tid}
    new_assign = {mid: remap.get(tid, tid) for mid, tid in assignments.items()}
    logger.info("중복 스레드 병합: %d개 -> %d개 (%d건 통합)",
                len(all_threads), len(merged), n_merged)
    return merged, new_assign, n_merged


class Merge(BaseModel):
    from_category: str = Field(description="합쳐져 사라질 기존 category 이름")
    to_category: str = Field(description="대표로 남길 category 이름")


class MergePlan(BaseModel):
    merges: list[Merge] = Field(description="동의어·포함관계인 category 병합 규칙들. 없으면 빈 목록.")


JANITOR_PROMPT = """다음은 한 대화방을 주제 분류한 결과의 category 목록입니다.
같은 뜻인데 표기만 다른 것들(동의어·포함관계)을 하나로 합치는 규칙을 만들어 주세요.
- 명백히 같은 개념만 보수적으로 합치세요. 애매하면 그대로 두세요.
- 대표 이름은 더 넓고 자연스러운 쪽으로.
- 서로 다른 개념은 절대 합치지 마세요.

[category 목록 (이름 · 스레드수 · 대표 topic)]
{catalog}
"""


def _catalog(all_threads: dict[int, Thread]) -> str:
    counts = Counter(t.category for t in all_threads.values())
    topics: dict[str, list[str]] = defaultdict(list)
    for t in all_threads.values():
        if len(topics[t.category]) < 4:
            topics[t.category].append(t.topic)
    return NL.join(f'- "{cat}" · {n}개 · {", ".join(topics[cat])}'
                   for cat, n in counts.most_common())


def _resolve(cat: str, mapping: dict[str, str]) -> str:
    """a->b, b->c 체인을 최종 대표까지 해소 (순환 방지)."""
    seen: set[str] = set()
    while cat in mapping and cat not in seen:
        seen.add(cat)
        cat = mapping[cat]
    return cat


def consolidate(all_threads: dict[int, Thread], pool, tracker=None
                ) -> tuple[dict[int, Thread], dict[str, str]]:
    """category 이름을 통일. 반환: (갱신된 all_threads, 적용된 병합맵 old->new).

    thread_id/topic/summary/배정은 그대로 두고 category 문자열만 바꾼다.
    """
    cats = {t.category for t in all_threads.values()}
    if len(cats) <= 1:
        return all_threads, {}

    plan = invoke_structured(pool, JANITOR_PROMPT.format(catalog=_catalog(all_threads)),
                             MergePlan, tracker, label="정리부")
    mapping = {m.from_category: m.to_category for m in plan.merges
               if m.from_category != m.to_category and m.from_category in cats}
    if not mapping:
        logger.info("정리부: 병합할 category 없음")
        return all_threads, {}

    merged = {tid: Thread(thread_id=t.thread_id, category=_resolve(t.category, mapping),
                          topic=t.topic, summary=t.summary)
              for tid, t in all_threads.items()}
    logger.info("정리부: category %d개 -> %d개 (병합 %d건)",
                len(cats), len({t.category for t in merged.values()}), len(mapping))
    return merged, mapping
