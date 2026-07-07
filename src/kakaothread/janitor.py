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
