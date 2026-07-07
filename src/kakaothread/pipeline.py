"""분류 파이프라인 공용 코어 — 순차(segment_graph)·병렬(parallel)·증분(incremental) 러너가 공유.

여기에 있는 것: 실행 옵션, 입력 준비(파싱→병합→익명화→청킹→부분제한),
전역 thread_id 발급, LLM 결과 통합(지역 id → 전역 id 정규화), 도메인 탐색.
러너별 순회 전략(LangGraph 루프 / asyncio 세션 병렬)은 각 러너 모듈에 있다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

from . import domain
from .chunking import make_session_chunks
from .cost import UsageTracker
from .llm_segment import Thread
from .preprocess import Message, merge_consecutive, parse_kakao
from .preprocess import anonymize as anonymize_messages

logger = logging.getLogger(__name__)

# 잡담/무의미 전용 스레드 (프롬프트 규칙의 thread_id=0 과 일치, 전역 고정)
NOISE_THREAD_ID = 0

# (session_idx, 청크 메시지들)
Chunk = tuple[int, list[Message]]


@dataclass(frozen=True)
class PipelineOptions:
    """러너 공통 실행 옵션 (CLI 플래그와 1:1)."""
    max_chunks: int | None = None    # --limit N: 앞 N청크만
    use_domain: bool = True          # --no-domain 의 반대
    room_desc: str = ""              # --room-desc: 도메인 탐색 부트스트랩
    fixed_taxonomy: bool = False     # --fixed-taxonomy: category 고정 어휘
    anonymize: bool = False          # --anonymize: 발신자 가명 치환


class RunResult(NamedTuple):
    messages: list[Message]
    threads: dict[int, Thread]
    assignments: dict[int, int]
    tracker: UsageTracker
    merge_map: dict[str, str]
    chunks_done: int
    chunks_total: int
    anonymize: bool = False  # messages 가 익명화된 상태인지 (meta 기록·증분 판단용)


class IdAlloc:
    """전역 thread_id 발급기. 순차·병렬 공용.

    asyncio 는 단일 스레드라 락 없이 안전. next 는 '다음에 발급될 값'.
    """

    def __init__(self, start: int = 1) -> None:
        self.next = start

    def __call__(self) -> int:
        v = self.next
        self.next += 1
        return v


def integrate(
    result,
    open_threads: list[Thread],
    all_threads: dict[int, Thread],
    alloc: IdAlloc,
) -> tuple[list[Thread], dict[int, Thread], dict[int, int]]:
    """LLM 결과의 지역 thread_id를 전역 고유 id로 정규화하고 상태를 갱신한다.

    - 현재 세션에 이미 열린 스레드면 그 id를 재사용
    - 새 주제면 alloc()으로 전역 id 발급 (과거·다른 세션 id와의 충돌 방지)
    반환: (갱신된 open_threads, 갱신된 all_threads, msg_id->전역id)
    """
    existing = {t.thread_id for t in open_threads}
    local2global = {NOISE_THREAD_ID: NOISE_THREAD_ID}
    all_threads = dict(all_threads)
    open_by_id = {t.thread_id: t for t in open_threads}

    for t in result.threads:
        if t.thread_id == NOISE_THREAD_ID:
            continue
        gid = t.thread_id if t.thread_id in existing else alloc()
        local2global[t.thread_id] = gid
        # 스레드가 청크를 넘나들면 요약은 최근 청크 것으로 갱신 (빈 값이면 이전 요약 유지)
        summary = t.summary or (all_threads[gid].summary if gid in all_threads else "")
        gt = Thread(thread_id=gid, category=t.category, topic=t.topic, summary=summary)
        all_threads[gid] = gt
        open_by_id[gid] = gt

    def resolve(local_id: int) -> int:
        if local_id in local2global:
            return local2global[local_id]
        if local_id in existing:  # 재선언 없이 재사용된 열린 스레드 (이미 전역 id)
            return local_id
        # 선언되지 않은 id로의 배정 — LLM이 threads에 없는 번호를 쓴 경우.
        # 지역 id를 전역으로 유출하면 무관한 스레드에 오배정되므로 잡담(0)으로 보낸다.
        logger.warning("배정이 선언되지 않은 thread_id=%d 를 참조 — 잡담(0)으로 처리", local_id)
        return NOISE_THREAD_ID

    assignments = {a.msg_id: resolve(a.thread_id) for a in result.assignments}
    return list(open_by_id.values()), all_threads, assignments


def limit_chunks(
    messages: list[Message],
    chunks: list[Chunk],
    max_chunks: int | None,
) -> tuple[list[Message], list[Chunk], int]:
    """부분 실행: 앞 max_chunks개 청크와 그 범위까지의 메시지만 남긴다.

    메시지도 함께 잘라야 리포트가 미처리 발화를 '미분류'로 오인하지 않는다.
    반환: (잘린 messages, 잘린 chunks, 원래 청크 총수)
    """
    total = len(chunks)
    if not chunks or max_chunks is None or max_chunks >= total:
        return messages, chunks, total
    chunks = chunks[:max_chunks]
    last_id = chunks[-1][1][-1].msg_id
    logger.info("부분 실행: 청크 %d/%d (msg_id ≤ %d)", len(chunks), total, last_id)
    return messages[: last_id + 1], chunks, total


def load_input(path: str, opts: PipelineOptions) -> tuple[list[Message], list[Chunk], int]:
    """입력 준비: 파싱 → 병합 → (익명화) → 청킹 → (부분 제한).

    반환: (messages, chunks, 원래 청크 총수)
    """
    messages = merge_consecutive(parse_kakao(path))
    if opts.anonymize:
        messages = anonymize_messages(messages)  # LLM 포함 이후 전 과정에 실명 미노출
    chunks = make_session_chunks(messages)
    return limit_chunks(messages, chunks, opts.max_chunks)


def discover_domain(
    messages: list[Message], pool, tracker,
    *, room_desc: str = "", fixed_taxonomy: bool = False,
) -> tuple[str, list[str] | None]:
    """도메인 탐색 (LLM 1콜) → (예시 블록, 고정 category 어휘 or None).

    실패해도 파이프라인은 계속 가야 하므로 예외는 삼키고 기본값으로 진행.
    """
    try:
        profile = domain.discover(messages, pool, tracker, room_desc=room_desc)
    except Exception as e:  # noqa: BLE001
        logger.warning("도메인 예시 탐색 실패 — 기본 예시로 진행: %s", e)
        return "", None
    if profile is None:
        return "", None
    examples = domain.render_examples_block(profile)
    vocab = domain.category_vocab(profile) if fixed_taxonomy else None
    if vocab:
        logger.info("고정 택소노미 모드: category %d개 + '기타' — %s", len(vocab), ", ".join(vocab))
    return examples, vocab


def domain_context(
    messages: list[Message], pool, tracker, opts: PipelineOptions,
) -> tuple[str, list[str] | None]:
    """옵션을 반영한 도메인 컨텍스트 (use_domain=False 면 기본값)."""
    if not opts.use_domain:
        return "", None
    return discover_domain(messages, pool, tracker,
                           room_desc=opts.room_desc, fixed_taxonomy=opts.fixed_taxonomy)
