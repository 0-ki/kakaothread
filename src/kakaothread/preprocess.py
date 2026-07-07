"""
KakaoTalk 오픈채팅 export(.txt) 파서.

의존성 없음 — 표준 라이브러리만 사용 (portable / open-source friendly).
목표: txt 파일 -> list[Message]  (봇/시스템 줄 제거, 멀티라인 병합, 24h 변환)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 정규식 ──────────────────────────────────────────────────────────
# 날짜 표기가 내보내기 경로에 따라 다르다:
#   PC:     "2026년 5월 11일 오후 2:25, <내용>"
#   모바일: "2026. 5. 11. 오후 2:25, <내용>"
# 줄 단위로 두 패턴을 모두 시도하므로 포맷 자동 감지 (혼재해도 동작).
_DATE_PC = r"(\d{4})년 (\d{1,2})월 (\d{1,2})일"
_DATE_MOBILE = r"(\d{4})\. (\d{1,2})\. (\d{1,2})\."
_TIME = r"(오전|오후) (\d{1,2}):(\d{2})"

LINE_RES = [
    re.compile(rf"^{_DATE_PC} {_TIME}, (.*)$"),
    re.compile(rf"^{_DATE_MOBILE} {_TIME}, (.*)$"),
]
# 콤마 없는 '날짜/구분선' 줄 (스킵 대상) — 시각 구분선과 요일 날짜 헤더
DATE_ONLY_RES = [
    re.compile(rf"^{_DATE_PC} {_TIME}$"),
    re.compile(rf"^{_DATE_MOBILE} {_TIME}$"),
    re.compile(rf"^{_DATE_PC} [일월화수목금토]요일$"),
    re.compile(rf"^{_DATE_MOBILE} [일월화수목금토]요일$"),
]

SENDER_SEP = " : "          # 발신자/본문 구분자 (공백-콜론-공백)
BOT_SENDER = "오픈채팅봇"

# 본문 자체가 노이즈인 경우 (제거 대상)
NOISE_TEXTS = {
    "메시지가 삭제되었습니다.",
    "삭제된 메시지입니다.",
    # TODO: 필요시 "사진", "이모티콘", "동영상" 등 추가
}


@dataclass
class Message:
    msg_id: int
    dt: datetime
    sender: str
    text: str


# ── 헬퍼 ────────────────────────────────────────────────────────────
def to_24h(ampm: str, hour: int) -> int:
    """오전/오후 + 12시간제 hour -> 24시간제 hour.
    2026년 5월 11일 오후 2:19
    규칙: 오전 12 -> 0,  오후 12 -> 12,  오후 N(1~11) -> N+12,  오전 N -> N
    """
    if ampm == "오전":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def parse_line(line: str) -> tuple[datetime, str] | None:
    """줄이 '날짜, 내용' 패턴(PC/모바일)이면 (datetime, 내용)을 반환.

    아니면 None  -> 호출부에서 '날짜 구분선 / 헤더 / 연속줄' 중 하나로 처리.
    """
    for line_re in LINE_RES:
        m = line_re.match(line)
        if m is not None:
            year, month, day, ampm, hour, minute, content = m.groups()
            dt = datetime(int(year), int(month), int(day), to_24h(ampm, int(hour)), int(minute))
            return dt, content
    return None


def is_date_separator(line: str) -> bool:
    """콤마 없는 날짜/시각 구분선(스킵 대상)이면 True.

    놓치면 직전 메시지의 '연속 줄'로 오인돼 본문에 섞여 들어간다.
    """
    return any(r.match(line) for r in DATE_ONLY_RES)


def split_sender(content: str) -> tuple[str, str] | None:
    """'발신자 : 본문' -> (발신자, 본문).

    SENDER_SEP 가 없으면(입장/퇴장 같은 시스템 이벤트) None.
    """
    if SENDER_SEP not in content:
        return None
    sender, text = content.split(SENDER_SEP, 1)
    return sender, text


def is_noise(sender: str, text: str) -> bool:
    """봇 / 삭제마커 / 빈 본문 등 제거 대상이면 True."""
    if sender == BOT_SENDER:
        return True
    if text in NOISE_TEXTS:
        return True
    if not text.strip():
        return True
    return False


# ── 메인 파서 (상태 머신) ───────────────────────────────────────────
def parse_kakao(path: str | Path) -> list[Message]:
    messages: list[Message] = []

    # 현재 누적 중인 메시지 상태
    cur_dt: datetime | None = None
    cur_sender: str | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        """누적된 현재 메시지를 확정해 messages에 추가 (노이즈면 버림)."""
        nonlocal cur_dt, cur_sender, cur_lines
        if cur_dt is not None and cur_sender is not None:
            text = " ".join(cur_lines).strip()
            if not is_noise(cur_sender, text):
                messages.append(
                    Message(msg_id=-1, dt=cur_dt, sender=cur_sender, text=text)
                )
        cur_dt = None
        cur_sender = None
        cur_lines = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            parsed = parse_line(line)

            if parsed is None:
                # '날짜 구분선'이면 스킵, 아니면 직전 메시지의 연속 줄
                if is_date_separator(line):
                    continue
                if cur_sender is not None:
                    cur_lines.append(line)
                continue

            dt, content = parsed
            sd = split_sender(content)

            if sd is None:
                # 시스템 이벤트(입장/퇴장) — 진행 중 메시지 마무리 후 스킵
                flush()
                continue

            # 새 메시지 시작
            sender, first_line = sd
            flush()
            cur_dt = dt
            cur_sender = sender
            cur_lines = [first_line]

        flush()  # 파일 끝 — 마지막 메시지 확정

    # 연속된 msg_id 부여
    for i, m in enumerate(messages):
        m.msg_id = i
    logger.info("파싱 완료: %s -> %d개 발화", path, len(messages))
    return messages


def dump_messages(messages: list[Message], path: str | Path) -> Path:
    """파싱·병합이 끝난 메시지를 jsonl로 저장 (msg_id↔본문 매핑 영속화).

    threads.json 은 msg_id 만 담으므로, 이 파일이 있어야 원본 재파싱 없이
    run 폴더만으로 결과를 해석할 수 있다 (자기완결·재현성).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for m in messages:
            rec = {"msg_id": m.msg_id, "dt": m.dt.isoformat(), "sender": m.sender, "text": m.text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("메시지 저장: %s (%d개)", path, len(messages))
    return path


def load_messages(path: str | Path) -> list[Message]:
    """dump_messages 로 저장한 jsonl 을 Message 리스트로 복원."""
    out: list[Message] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Message(msg_id=r["msg_id"], dt=datetime.fromisoformat(r["dt"]),
                               sender=r["sender"], text=r["text"]))
    return out


def anonymize(messages: list[Message]) -> list[Message]:
    """발신자를 등장 순서대로 '참가자N' 가명으로 치환한다 (본문 내 언급 포함).

    - 파이프라인 초입에서 적용하면 LLM에도 실명이 전달되지 않는다.
    - 실명↔가명 매핑은 어디에도 저장하지 않는다 (역추적 방지).
    - 등장 순서 기반이라 같은 입력이면 결과가 결정적 (재개/재현 안전).
    - **접두 안정성**: 본문 언급 치환은 '그 메시지 시점까지 등장한 발화자'만 대상으로 한다.
      재내보내기(append-only)에서 뒤에 처음 등장하는 발화자는 앞선 메시지의 치환 결과에
      영향을 주지 않으므로, 증분 처리(incremental)의 공통 접두가 깨지지 않는다.
      (전역 발화자 집합으로 치환하면 새 발화자 이름이 옛 본문까지 바꿔 접두가 붕괴함)
    - 한계: 본문 치환은 '발신자명과 정확히 같은 문자열'만 — 별명/오타는 못 잡는다.
      아직 발화하지 않은 사람의 이름 언급은 마스킹되지 않는다(접두 안정성과의 트레이드오프).
    """
    mapping: dict[str, str] = {}
    for m in messages:
        if m.sender not in mapping:
            mapping[m.sender] = f"참가자{len(mapping) + 1}"

    out: list[Message] = []
    seen: set[str] = set()
    for m in messages:
        seen.add(m.sender)
        # 지금까지 등장한 발화자만, 긴 이름부터 치환
        # (짧은 이름이 긴 이름의 부분 문자열일 때 "김철"⊂"김철수" 오염 방지)
        text = m.text
        for name in sorted(seen, key=len, reverse=True):
            if name in text:
                text = text.replace(name, mapping[name])
        out.append(Message(msg_id=m.msg_id, dt=m.dt, sender=mapping[m.sender], text=text))
    logger.info("익명화: 발신자 %d명 -> 참가자N 가명 치환", len(mapping))
    return out


def merge_consecutive(messages: list[Message], within_seconds: int = 60) -> list[Message]:
    """같은 화자가 짧은 간격으로 연달아 보낸 발화를 하나로 합친다.

    카톡 특성상 한 생각을 엔터로 쪼개 보내는 경우가 많다.
    직전 발화와 (1) 같은 화자이고 (2) 시간차가 within_seconds 이내면 이어붙인다.

    주의: 시간차는 'burst의 시작'이 아니라 '바로 직전 발화'와 비교한다.
          (8초 간격으로 5번 보내면 총 40초라도 전부 합쳐져야 하므로)
    """
    merged: list[Message] = []
    prev_dt = None  # 바로 직전 '원본' 발화의 시각 (burst 연속성 판단용)

    for m in messages:
        can_merge = (
            merged
            and merged[-1].sender == m.sender
            and (m.dt - prev_dt).total_seconds() <= within_seconds
        )
        if can_merge:
            merged[-1].text += " " + m.text
        else:
            # 새 발화 — 원본을 복사해서 추가 (원본 리스트 보호)
            merged.append(Message(msg_id=-1, dt=m.dt, sender=m.sender, text=m.text))
        prev_dt = m.dt

    for i, x in enumerate(merged):
        x.msg_id = i
    logger.info("연속발화 병합: %d -> %d개", len(messages), len(merged))
    return merged
