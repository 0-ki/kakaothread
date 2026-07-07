"""순차 러너 — 전체 대화 로그를 주제 스레드로 분리하는 LangGraph 상태 루프.

청크를 순서대로 처리하며 '열린 스레드'를 다음 청크로 이어준다(상태 캐리).
세션(큰 시간 공백) 경계에서는 열린 스레드를 리셋한다 — 3시간 넘게 끊긴 대화는
이어질 가능성이 낮고, 리셋하면 프롬프트에 쌓이는 스레드 수도 제한돼 비용이 안정된다.

LLM이 청크 안에서 붙이는 thread_id는 그 청크 기준의 지역 번호이므로,
pipeline.integrate 가 전역 고유 id로 정규화한다.
세션 병렬 버전은 parallel.py, 증분 처리는 incremental.py.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph
from tqdm import tqdm

from . import config, janitor
from .cost import UsageTracker
from .llm_segment import Thread, classify_chunk
from .pipeline import (
    Chunk,
    IdAlloc,
    PipelineOptions,
    RunResult,
    domain_context,
    integrate,
    load_input,
)
from .preprocess import Message
from .provider_pool import ProviderPool

logger = logging.getLogger(__name__)


class SegmentState(TypedDict):
    chunks: list[Chunk]  # (session_idx, 청크 메시지들)
    cursor: int  # 다음에 처리할 청크 인덱스
    cur_session: int  # 현재 세션 (바뀌면 open_threads 리셋)
    next_id: int  # 새 스레드에 부여할 전역 고유 id
    open_threads: list[Thread]  # 현재 세션에서 활성 상태인 스레드들
    all_threads: dict[int, Thread]  # 지금까지 만들어진 모든 스레드 (최종 결과)
    assignments: dict[int, int]  # msg_id -> 전역 thread_id
    examples: str  # 방에 맞춘 category>topic 예시 블록 (분류 프롬프트 주입용)
    category_vocab: list[str] | None  # 고정 택소노미 (None=자유 모드)
    pool: ProviderPool  # LLM 슬롯 라운드로빈/페일오버
    tracker: UsageTracker
    pbar: tqdm


def classify_node(state: SegmentState) -> dict:
    """청크 하나를 분류하고 전역 상태를 갱신하는 노드."""
    sidx, chunk = state["chunks"][state["cursor"]]

    # 세션이 바뀌면 활성 스레드 리셋 (카테고리 어휘는 all_threads에 남아 유지)
    open_threads = [] if sidx != state["cur_session"] else state["open_threads"]

    result = classify_chunk(chunk, open_threads, state["pool"], state["tracker"],
                            examples=state["examples"],
                            category_vocab=state["category_vocab"])
    alloc = IdAlloc(state["next_id"])
    open_threads, all_threads, new_assign = integrate(
        result, open_threads, state["all_threads"], alloc
    )

    state["pbar"].update(1)
    logger.debug(
        "청크 %d/%d (세션 %d): 발화 %d -> 스레드 %d",
        state["cursor"] + 1,
        len(state["chunks"]),
        sidx,
        len(chunk),
        len(result.threads),
    )
    return {
        "cursor": state["cursor"] + 1,
        "cur_session": sidx,
        "next_id": alloc.next,
        "open_threads": open_threads,
        "all_threads": all_threads,
        "assignments": {**state["assignments"], **new_assign},
    }


def _should_continue(state: SegmentState) -> str:
    return "continue" if state["cursor"] < len(state["chunks"]) else "done"


def build_graph():
    """청크가 남아있는 동안 classify 노드를 반복하는 순환 그래프."""
    g = StateGraph(SegmentState)
    g.add_node("classify", classify_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _should_continue, {"continue": "classify", "done": END})
    return g.compile()


def run(path: str, opts: PipelineOptions = PipelineOptions()) -> RunResult:
    slot_desc = ", ".join(f"{s.name}({s.model}, p{s.priority})" for s in config.SLOTS)
    logger.info("분류 시작: %s (슬롯=[%s])", path, slot_desc)
    messages, chunks, chunks_total = load_input(path, opts)
    pool = ProviderPool(config.SLOTS)
    tracker = UsageTracker()
    examples, vocab = domain_context(messages, pool, tracker, opts)

    init: SegmentState = {
        "chunks": chunks,
        "cursor": 0,
        "cur_session": -1,
        "next_id": 1,  # 0은 잡담 전용이라 1부터
        "open_threads": [],
        "all_threads": {},
        "assignments": {},
        "examples": examples,
        "category_vocab": vocab,
        "pool": pool,
        "tracker": tracker,
        "pbar": tqdm(total=len(chunks), desc="segment"),
    }
    graph = build_graph()
    # LangGraph 기본 recursion_limit는 25 — 청크 수만큼 반복하므로 늘려준다
    final = graph.invoke(init, {"recursion_limit": len(chunks) + 10})
    init["pbar"].close()

    all_threads = final["all_threads"]
    # 정리부(janitor): 흔들린 category 이름을 한 번에 통일 (LLM 1콜). 실패해도 원본 유지.
    merge_map: dict[str, str] = {}
    try:
        all_threads, merge_map = janitor.consolidate(all_threads, pool, tracker)
    except Exception as e:  # noqa: BLE001
        logger.warning("정리부 실패 — 병합 없이 진행: %s", e)

    logger.info(
        "분류 완료: 스레드 %d개, 토큰 in=%d out=%d, 비용 ~$%.4f",
        len(all_threads), tracker.tok_in, tracker.tok_out, tracker.cost,
    )
    return RunResult(messages, all_threads, final["assignments"], tracker, merge_map,
                     len(chunks), chunks_total, opts.anonymize)
