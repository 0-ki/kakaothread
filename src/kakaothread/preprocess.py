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
# 카카오톡 내보내기는 두 계열의 포맷이 있고, 둘 다 자동 감지·지원한다.
#
# 포맷 A — 줄마다 날짜+시각+발신자가 다 들어있는 형태:
#   PC:     "2026년 5월 11일 오후 2:25, 철수 : 안녕"
#   모바일: "2026. 5. 11. 오후 2:25, 철수 : 안녕"
#
# 포맷 B — 날짜는 '요일 구분선'에만, 메시지 줄엔 시각만 있는 형태:
#   "--------------- 2026년 7월 3일 금요일 ---------------"
#   "[철수] [오후 2:37] 안녕"
#
# 줄 단위로 모든 패턴을 시도하므로 두 포맷이 섞여 있어도 동작한다.
_DATE_PC = r"(\d{4})년 (\d{1,2})월 (\d{1,2})일"
_DATE_MOBILE = r"(\d{4})\. (\d{1,2})\. (\d{1,2})\."
_TIME = r"(오전|오후) (\d{1,2}):(\d{2})"

# 포맷 A: "날짜 시각, 발신자 : 본문"
LINE_RES = [
    re.compile(rf"^{_DATE_PC} {_TIME}, (.*)$"),
    re.compile(rf"^{_DATE_MOBILE} {_TIME}, (.*)$"),
]

# 날짜 섹션 헤더 (요일 포함) — 포맷 B의 '유일한' 날짜 공급원.
# 대시로 감싼 형태와 대시 없는 형태 모두 인식하고 (y, m, d) 를 넘긴다.
_DASH = r"-*"
DATE_HEADER_RES = [
    re.compile(rf"^\s*{_DASH}\s*{_DATE_PC} [일월화수목금토]요일\s*{_DASH}\s*$"),
    re.compile(rf"^\s*{_DASH}\s*{_DATE_MOBILE} [일월화수목금토]요일\s*{_DASH}\s*$"),
]

# 콤마 없는 '시각 구분선' 줄 (스킵 대상) — 포맷 A의 세로 변형 "2026년 5월 11일 오후 2:19"
DATE_ONLY_RES = [
    re.compile(rf"^{_DATE_PC} {_TIME}$"),
    re.compile(rf"^{_DATE_MOBILE} {_TIME}$"),
]

# 포맷 B 메시지: "[발신자] [오후 H:MM] 본문" (발신자에 공백/이모지 허용)
BRACKET_MSG_RE = re.compile(rf"^\[(.+?)\] \[{_TIME}\] (.*)$")

# 입장/퇴장 등 시스템 줄 — 포맷 B에선 대괄호 없이 나타나므로 연속 줄로 오인 방지
SYSTEM_LINE_RES = [
    re.compile(r"^.{1,50}?님이 (들어왔습니다|나갔습니다)\.?$"),
    re.compile(r"^.{1,50}?님을 내보냈습니다\.?$"),
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
    """콤마 없는 시각 구분선(스킵 대상)이면 True.

    놓치면 직전 메시지의 '연속 줄'로 오인돼 본문에 섞여 들어간다.
    """
    return any(r.match(line) for r in DATE_ONLY_RES)


def match_date_header(line: str) -> tuple[int, int, int] | None:
    """'요일 날짜 헤더'면 (year, month, day) 반환, 아니면 None.

    포맷 B는 날짜가 이 헤더에만 있으므로, 파서가 여기서 날짜를 얻어 이후
    메시지(시각만 있는)에 물려준다. 포맷 A에선 단순 스킵 대상이라 무해하다.
    """
    for r in DATE_HEADER_RES:
        m = r.match(line)
        if m is not None:
            year, month, day = m.groups()
            return int(year), int(month), int(day)
    return None


def is_system_line(line: str) -> bool:
    """입장/퇴장/내보내기 같은 시스템 줄이면 True (포맷 B의 대괄호 없는 형태)."""
    return any(r.match(line) for r in SYSTEM_LINE_RES)


def parse_bracket_line(
    line: str, cur_date: tuple[int, int, int] | None
) -> tuple[datetime, str, str] | None:
    """포맷 B '[발신자] [오후 H:MM] 본문' -> (datetime, 발신자, 본문).

    cur_date(직전 날짜 헤더에서 얻은 y,m,d)가 있어야 완전한 시각을 만들 수 있다.
    패턴 불일치이거나 날짜 헤더가 아직 없으면 None.
    """
    m = BRACKET_MSG_RE.match(line)
    if m is None:
        return None
    sender, ampm, hour, minute, text = m.groups()
    if cur_date is None:
        return None  # 날짜 헤더보다 먼저 나온 메시지 — 날짜 미상이라 처리 불가
    year, month, day = cur_date
    dt = datetime(year, month, day, to_24h(ampm, int(hour)), int(minute))
    return dt, sender, text


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
    cur_date: tuple[int, int, int] | None = None  # 포맷 B: 최근 날짜 헤더의 (y,m,d)

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

            # 1) 날짜 섹션 헤더 — 포맷 B의 날짜를 갱신하고 스킵 (포맷 A의 요일 헤더도 여기서 처리)
            dh = match_date_header(line)
            if dh is not None:
                flush()
                cur_date = dh
                continue

            # 2) 포맷 A: "날짜 시각, 발신자 : 본문"
            parsed = parse_line(line)
            if parsed is not None:
                dt, content = parsed
                sd = split_sender(content)
                if sd is None:
                    # 시스템 이벤트(콤마 O, 콜론 X — 입장/퇴장) — 진행 중 메시지 마무리 후 스킵
                    flush()
                    continue
                sender, first_line = sd
                flush()
                cur_dt, cur_sender, cur_lines = dt, sender, [first_line]
                continue

            # 3) 포맷 B: "[발신자] [오후 H:MM] 본문"
            bracket = parse_bracket_line(line, cur_date)
            if bracket is not None:
                dt, sender, first_line = bracket
                flush()
                cur_dt, cur_sender, cur_lines = dt, sender, [first_line]
                continue

            # 4) 그 외 — 시각 구분선/시스템 줄/멀티라인 연속 줄/헤더 잡음
            if is_date_separator(line):
                continue
            if is_system_line(line):
                # 대괄호 없는 입/퇴장 줄 — 진행 중 메시지 마무리 후 스킵 (연속 줄 오인 방지)
                flush()
                continue
            if cur_sender is not None:
                cur_lines.append(line)  # 멀티라인 본문
            # else: 파일 헤더 등 잡음 — 버림

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
