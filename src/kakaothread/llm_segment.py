"""
LLM structured-output 기반 대화 분리 (conversation disentanglement).

카톡 로그를 category(상위 범주) > topic(세부 주제) 2단 계층 스레드로 분리.
임베딩 클러스터링이 못 잡는 암묵적 주제(축약어·은어가 가리키는 실제 주제)를 LLM 세상지식으로 해결.

이 파일: 단일 청크 분류(classify_chunk). 전체 순회는 segment_graph.py (LangGraph).
"""
from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import LLM_TIMEOUT, Slot
from .cost import UsageTracker
from .preprocess import Message
from .provider_pool import ProviderPool

logger = logging.getLogger(__name__)

NL = "\n"


# ── structured output 스키마 ────────────────────────────────────────
class Assignment(BaseModel):
    msg_id: int = Field(description="대상 메시지의 id")
    thread_id: int = Field(description="속한 스레드 번호. 열린 스레드 재사용 또는 새 번호.")


class Thread(BaseModel):
    """category(상위 범주) + topic(세부 주제) 2단 계층 + 한 줄 요약."""
    thread_id: int
    category: str = Field(description="여러 세부 주제를 아우르는 넓고 재사용성 높은 상위 범주 (짧은 명사)")
    topic: str = Field(description="그 category 안의 구체적인 개별 주제 (짧은 명사)")
    summary: str = Field(
        default="",
        description="이 스레드에서 오간 대화 내용의 한 줄 요약 (40자 이내, 명사형 종결). 잡담(0)은 빈 문자열.",
    )


class ChunkResult(BaseModel):
    assignments: list[Assignment] = Field(description="청크 모든 메시지의 스레드 배정")
    threads: list[Thread] = Field(description="이 청크에서 새로 만들거나 사용한 스레드들")


# ── 프롬프트 ────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """당신은 뒤섞인 오픈채팅 대화를 주제별 스레드로 분리하는 전문가입니다.

여러 사람이 동시에 여러 주제로 대화하므로 주제가 서로 인터리빙(교차)되어 있습니다.
연속된 id가 같은 주제라고 가정하지 마세요.

규칙:
- 아래 '열린 스레드' 중 이어지는 주제가 있으면 그 thread_id를 재사용하세요.
- 새로운 주제이면 기존에 없는 새 thread_id(정수)를 부여하세요.
- 각 스레드에 category(상위 범주)와 topic(세부 주제)을 함께 매기세요.
    category: 여러 세부 주제를 아우르는 넓고 재사용성 높은 범주.
    topic: 그 category 안의 구체적인 개별 항목.
  하나의 category 아래에는 서로 다른 여러 topic이 올 수 있습니다.
{examples}
{category_rule}
- category/topic은 사건을 설명하는 문장이 아니라 짧은 명사(구)로.
- 각 스레드에 summary(한 줄 요약)도 매기세요: 그 스레드에서 실제 오간 내용을
  40자 이내 명사형으로 압축. 잡담(0)은 빈 문자열.
- 잡담/비아냥/말다툼/의미 없는 리액션(ㅋㅋ, ㅜㅜ 등)은 thread_id=0 (주제="잡담")에 배정하세요.
- 모든 메시지를 빠짐없이 배정하세요.

[열린 스레드]
{open_threads}

[메시지]
{messages}
"""

# 도메인 예시가 없을 때(탐색 생략/실패) 쓰는 기본 예시 블록.
# 방을 파악한 경우 domain.discover()가 이 자리를 방에 맞는 예시로 대체한다.
DEFAULT_EXAMPLES = """  (구조 예시 — 이 방과 무관한 예시일 뿐이니 단어에 얽매이지 마세요:
   category "가격" 아래 topic "가격 문의", "할인", "환불".)"""

# category 규칙 — 자유 모드(기본) vs 고정 택소노미 모드
FREE_CATEGORY_RULE = (
    "- category와 topic은 특정 분야를 가정하지 말고, 이 방의 실제 대화 내용에서 자연스럽게 뽑으세요.\n"
    "- 이미 만든 category가 적절하면 그 이름을 그대로 재사용하세요 (일관성 유지)."
)
FIXED_CATEGORY_RULE = """- category는 반드시 다음 고정 목록에서만 선택하세요. 딱 맞는 것이 없으면 "기타"를 사용하세요:
  {vocab}
- topic은 선택한 category 안에서 실제 대화 내용에 맞게 자유롭게 정하세요."""

OTHER_CATEGORY = "기타"


def _category_rule(category_vocab: list[str] | None) -> str:
    if not category_vocab:
        return FREE_CATEGORY_RULE
    return FIXED_CATEGORY_RULE.format(vocab=", ".join(f'"{c}"' for c in category_vocab))


def _render_messages(messages: list[Message]) -> str:
    return NL.join(f"{m.msg_id} | {m.sender}: {m.text.replace(NL, ' ')}" for m in messages)


def _render_open_threads(open_threads: list[Thread]) -> str:
    if not open_threads:
        return "(아직 없음 — 모두 새 스레드)"
    return NL.join(f"{t.thread_id}: {t.category} > {t.topic}" for t in open_threads)


_LLM_CACHE: dict[tuple[str, str], object] = {}


def reset_cache() -> None:
    """LLM 인스턴스 캐시 비우기.

    ChatOpenAI 는 내부 async 클라이언트(httpx)를 생성 이벤트 루프에 묶는다.
    병렬 러너는 실행마다 새 asyncio.run(=새 이벤트 루프)을 쓰므로, 이전 실행에서
    캐시된 클라이언트를 재사용하면 'Event loop is closed' 계열 오류가 난다.
    각 병렬 실행 시작 전에 호출해 루프별로 새 클라이언트를 만들게 한다.
    """
    _LLM_CACHE.clear()


def _resolve_reasoning_effort(slot: Slot) -> str:
    """이 슬롯에 적용할 reasoning_effort. 슬롯 명시값 우선, 없으면 gpt-5 계열은 low.

    gpt-5/o 계열은 기본 medium 추론이 켜져 있어 느리고 숨은 reasoning 토큰이 out
    토큰으로 잡혀 폭증한다. 반대로 minimal 은 이 작업(대화 분리+2단 계층 라벨링)에
    필요한 추론까지 꺼버려 대부분을 잡담(thread 0)으로 뭉개는 품질 붕괴가 났다.
    → 절충값 low 를 기본으로. 품질이 더 필요하면 슬롯에서 medium 으로 올린다.
    비추론 모델(Cerebras 등)은 ""(미적용)로 둔다.
    """
    if slot.reasoning_effort:
        return slot.reasoning_effort
    if "gpt-5" in slot.model.lower():  # openai/gpt-5-nano 같은 프록시 표기도 포함
        return "low"
    return ""


def build_llm(slot: Slot, schema=ChunkResult):
    """슬롯×스키마별 structured-output LLM (include_raw로 토큰 정보 보존). 인스턴스 캐시."""
    key = (slot.name, schema.__name__)
    llm = _LLM_CACHE.get(key)
    if llm is None:
        kwargs = dict(
            model=slot.model,
            base_url=slot.conn.base_url or None,  # 빈 문자열이면 OpenAI 기본
            api_key=slot.conn.api_key,
            timeout=LLM_TIMEOUT,  # 응답을 물고 있는 엔드포인트에서 무한 대기 방지
            max_retries=0,  # SDK 내부 재시도(429 시 Retry-After만큼 sleep) 끔 → 우리 풀이 즉시 페일오버
        )
        effort = _resolve_reasoning_effort(slot)
        if effort:
            # gpt-5/o 추론 모델은 temperature 조절 불가(기본값만 허용) → 넘기지 않는다.
            kwargs["reasoning_effort"] = effort
        else:
            kwargs["temperature"] = 0
        base = ChatOpenAI(**kwargs)
        llm = base.with_structured_output(schema, include_raw=True)
        _LLM_CACHE[key] = llm
    return llm


def _account(out, slot, pool, tracker, label):
    """호출 성공 후 토큰 집계 공통부. parsed 결과를 반환."""
    raw = out["raw"]
    u = getattr(raw, "usage_metadata", None) or {}
    tok_in = u.get("input_tokens", 0)
    tok_out = u.get("output_tokens", 0)
    pool.on_success(slot, tok_in + tok_out)
    if tracker is not None:
        tracker.add(slot, tok_in, tok_out)
    logger.info("%s 완료: slot=%s model=%s tok=%d/%d", label, slot.name, slot.model, tok_in, tok_out)
    return out["parsed"]


def invoke_structured(pool, prompt, schema, tracker=None, *, max_tries=6, label="LLM"):
    """풀에서 슬롯을 받아 구조화 출력 1콜. 실패(429 등)하면 다른 슬롯으로 페일오버.

    분류/도메인탐색/정리부가 공통으로 쓰는 호출 루프 (429·토큰집계·페일오버 한 곳).
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        slot = pool.acquire()
        logger.info("%s 호출: slot=%s model=%s (시도 %d/%d)",
                    label, slot.name, slot.model, attempt, max_tries)
        try:
            out = build_llm(slot, schema).invoke(prompt)  # {"raw":..., "parsed":...}
        except Exception as e:  # noqa: BLE001 — rate limit/일시 장애 → 다음 슬롯으로
            pool.on_error(slot, e)
            last_exc = e
            logger.warning("슬롯 %s(%s) 실패 → 페일오버: %s", slot.name, slot.model, e)
            continue
        return _account(out, slot, pool, tracker, label)

    raise RuntimeError(f"{label}: 모든 슬롯 실패 (마지막 오류: {last_exc})") from last_exc


async def ainvoke_structured(pool, prompt, schema, tracker=None, *, max_tries=6, label="LLM"):
    """invoke_structured 의 asyncio 버전 (세션 병렬 실행용)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        slot = await pool.acquire_async()
        logger.info("%s 호출: slot=%s model=%s (시도 %d/%d)",
                    label, slot.name, slot.model, attempt, max_tries)
        try:
            out = await build_llm(slot, schema).ainvoke(prompt)
        except Exception as e:  # noqa: BLE001
            pool.on_error(slot, e)
            last_exc = e
            logger.warning("슬롯 %s(%s) 실패 → 페일오버: %s", slot.name, slot.model, e)
            continue
        return _account(out, slot, pool, tracker, label)

    raise RuntimeError(f"{label}: 모든 슬롯 실패 (마지막 오류: {last_exc})") from last_exc


def _chunk_prompt(messages, open_threads, examples, category_vocab) -> str:
    return PROMPT_TEMPLATE.format(
        examples=examples or DEFAULT_EXAMPLES,
        category_rule=_category_rule(category_vocab),
        open_threads=_render_open_threads(open_threads),
        messages=_render_messages(messages),
    )


def classify_chunk(
    messages: list[Message],
    open_threads: list[Thread],
    pool: ProviderPool,
    tracker: UsageTracker | None = None,
    examples: str | None = None,
    category_vocab: list[str] | None = None,
    max_tries: int = 6,
) -> ChunkResult:
    """청크 메시지들을 열린 스레드 맥락 위에서 분류.

    examples: 이 방에 맞춘 category>topic 예시 블록(domain.discover). None이면 기본 예시.
    category_vocab: 고정 택소노미 — 주면 category를 이 목록(+"기타")에서만 고르게 한다.
    """
    prompt = _chunk_prompt(messages, open_threads, examples, category_vocab)
    result = invoke_structured(pool, prompt, ChunkResult, tracker, max_tries=max_tries, label="분류")
    logger.debug("분류: 발화 %d개 -> 스레드 %d개", len(messages), len(result.threads))
    return result


async def classify_chunk_async(
    messages: list[Message],
    open_threads: list[Thread],
    pool: ProviderPool,
    tracker: UsageTracker | None = None,
    examples: str | None = None,
    category_vocab: list[str] | None = None,
    max_tries: int = 6,
) -> ChunkResult:
    """classify_chunk 의 asyncio 버전 (세션 병렬 실행용)."""
    prompt = _chunk_prompt(messages, open_threads, examples, category_vocab)
    result = await ainvoke_structured(pool, prompt, ChunkResult, tracker,
                                      max_tries=max_tries, label="분류")
    logger.debug("분류: 발화 %d개 -> 스레드 %d개", len(messages), len(result.threads))
    return result
