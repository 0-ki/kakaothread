"""샘플 대화로 방의 도메인을 파악해, 분류 프롬프트에 넣을 '예시 블록'을 만든다.

기본은 few-shot 힌트만 제공(분류는 자유) — 고정 강제는 편향·경직 위험.
예: 등산 모임방이면 코스/장비 예시, 요리 방이면 레시피/재료 예시가 프롬프트에 들어간다.

옵션(고정 택소노미): discover 가 뽑은 category 목록을 권위 어휘로 잠그고
분류가 그 안에서만 고르게 할 수 있다("기타" 버킷 포함) — 세분화 일관성이
필요한 서비스 시나리오용. 사용자 방 설명(room_desc)으로 부트스트랩 가능.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from .llm_segment import invoke_structured
from .preprocess import Message

logger = logging.getLogger(__name__)

NL = "\n"


class ExamplePair(BaseModel):
    category: str = Field(description="이 방에서 나올 법한 넓고 재사용성 높은 상위 범주 (짧은 명사)")
    topic: str = Field(description="그 category 안의 구체적인 세부 주제 (짧은 명사)")


class DomainProfile(BaseModel):
    domain: str = Field(description="이 대화방이 어떤 방인지 한 구절 (예: '주식 투자 정보 공유방')")
    examples: list[ExamplePair] = Field(description="이 방에 어울리는 category>topic 예시 6~10개")


DISCOVER_PROMPT = """다음은 한 오픈채팅방에서 뽑은 대화 샘플입니다.
{room_desc_block}이 방이 어떤 주제의 방인지 파악하고, 이후 '주제 분류'에 참고할 대표적인
category(넓은 상위 범주) > topic(세부 주제) 예시를 6~10개 만들어 주세요.
- category는 넓고 재사용성 높게, topic은 그 안의 구체 항목으로.
- 실제 샘플에 나타난 주제를 반영하되, 이 방에서 앞으로 나올 만한 것도 포함.
- 사용자가 방 설명을 제공했다면 그 설명을 우선 신뢰하세요.
- 짧은 명사(구)로.

[대화 샘플]
{sample}
"""


def sample_for_domain(messages: list[Message], n_long: int = 12, n_spread: int = 48) -> list[Message]:
    """긴 메시지 몇 개(정성/핵심주제/말투) + 시간순 골고루(빈도/전체 커버)를 섞은 결정적 샘플.

    긴 것만 뽑으면 '짧은 티키타카로 오가는 주제'를 놓치므로 둘을 섞는다.
    (같은 사람의 연속 단타는 이미 merge_consecutive로 병합돼 하나의 긴 메시지가 된 상태)
    """
    if not messages:
        return []
    longest = sorted(messages, key=lambda m: len(m.text), reverse=True)[:n_long]
    long_ids = {m.msg_id for m in longest}
    rest = [m for m in messages if m.msg_id not in long_ids]
    spread: list[Message] = []
    if rest and n_spread > 0:
        step = max(1, len(rest) // n_spread)
        spread = rest[::step][:n_spread]
    picked = longest + spread
    picked.sort(key=lambda m: m.msg_id)  # 재현성
    return picked


def _render(sample: list[Message]) -> str:
    return NL.join(f"{m.sender}: {m.text.replace(NL, ' ')}" for m in sample)


def render_examples_block(profile: DomainProfile) -> str:
    pairs = ", ".join(f'"{p.category} > {p.topic}"' for p in profile.examples)
    return (f'  (이 방은 {profile.domain}로 보입니다. 아래 category > topic 예시를 참고하되,\n'
            f'   실제 대화에 맞게 벗어나도 됩니다: {pairs})')


def category_vocab(profile: DomainProfile) -> list[str]:
    """profile 예시에서 category 목록을 순서 보존·중복 제거로 추출 (고정 택소노미용)."""
    seen: dict[str, None] = {}
    for p in profile.examples:
        cat = p.category.strip()
        if cat:
            seen.setdefault(cat, None)
    return list(seen)


def discover(
    messages: list[Message], pool, tracker=None,
    n_long: int = 12, n_spread: int = 48, room_desc: str = "",
) -> DomainProfile | None:
    """방 도메인을 파악해 DomainProfile 을 반환. LLM 1콜. 샘플이 없으면 None.

    room_desc: 서비스 사용자가 입력한 방 설명 — discover 를 부트스트랩한다.
    """
    sample = sample_for_domain(messages, n_long, n_spread)
    if not sample:
        return None
    desc_block = f"[방 설명 (사용자 제공)]\n{room_desc.strip()}\n\n" if room_desc.strip() else ""
    prompt = DISCOVER_PROMPT.format(room_desc_block=desc_block, sample=_render(sample))
    profile = invoke_structured(pool, prompt, DomainProfile, tracker, label="도메인탐색")
    logger.info("도메인 파악: %s (예시 %d개, 샘플 %d발화)",
                profile.domain, len(profile.examples), len(sample))
    return profile
