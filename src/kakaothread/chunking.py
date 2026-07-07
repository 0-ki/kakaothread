"""
세션 분할 + 글자 예산 청킹. 표준 라이브러리만 사용.

흐름: messages -> [세션 분할: 큰 시간공백에서 컷] -> [세션별 글자·개수 예산 청킹] -> chunks
"""
from __future__ import annotations

import logging
from datetime import timedelta

from .config import CHUNK_CHAR_BUDGET, CHUNK_MAX_MESSAGES, SESSION_GAP_MINUTES
from .preprocess import Message

logger = logging.getLogger(__name__)


def split_sessions(
    messages: list[Message], gap_minutes: int = SESSION_GAP_MINUTES
) -> list[list[Message]]:
    """연속 발화 간 공백이 gap_minutes를 넘으면 세션을 나눈다 (경계)."""
    if not messages:
        return []
    gap = timedelta(minutes=gap_minutes)
    sessions: list[list[Message]] = [[messages[0]]]
    for prev, cur in zip(messages, messages[1:]):
        if cur.dt - prev.dt > gap:
            sessions.append([])
        sessions[-1].append(cur)
    return sessions


def _msg_chars(m: Message) -> int:
    """렌더링 시 대략 글자수 ('id | 화자: 텍스트' 오버헤드 포함)."""
    return len(m.sender) + len(m.text) + 8


def budget_chunks(
    session: list[Message],
    char_budget: int = CHUNK_CHAR_BUDGET,
    max_messages: int = CHUNK_MAX_MESSAGES,
) -> list[list[Message]]:
    """한 세션을 청크들로 쪼갠다 (크기).

    글자 예산(char_budget) 또는 메시지 개수(max_messages) 중 하나라도 넘기 직전에 끊는다.
    단일 메시지가 예산보다 커도 홀로 한 청크가 된다.
    """
    chunks: list[list[Message]] = []
    cur: list[Message] = []
    size = 0
    for m in session:
        c = _msg_chars(m)
        if cur and (size + c > char_budget or len(cur) >= max_messages):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(m)
        size += c
    if cur:
        chunks.append(cur)
    return chunks


def make_session_chunks(
    messages: list[Message],
    gap_minutes: int = SESSION_GAP_MINUTES,
    char_budget: int = CHUNK_CHAR_BUDGET,
) -> list[tuple[int, list[Message]]]:
    """세션 분할 후 세션별 예산 청킹. 각 청크에 소속 세션 인덱스를 붙인다.

    반환: list[(session_idx, chunk)]. 세션 경계에서 상태(열린 스레드)를
    리셋하려면 청크가 어느 세션인지 알아야 하므로 인덱스를 함께 준다.
    """
    sessions = split_sessions(messages, gap_minutes)
    out: list[tuple[int, list[Message]]] = []
    for sidx, session in enumerate(sessions):
        for chunk in budget_chunks(session, char_budget):
            out.append((sidx, chunk))
    logger.info(
        "청킹: 발화 %d개 -> 세션 %d개 -> 청크 %d개", len(messages), len(sessions), len(out)
    )
    logger.debug("청크당 메시지 수: %s", [len(c) for _, c in out])
    return out
