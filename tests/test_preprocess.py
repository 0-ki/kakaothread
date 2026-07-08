"""카카오톡 파서 테스트 — 포맷 파싱, 노이즈 필터, 멀티라인/연속발화 병합."""
from datetime import datetime, timedelta

from kakaothread.preprocess import (
    Message,
    anonymize,
    dump_messages,
    load_messages,
    merge_consecutive,
    parse_kakao,
    to_24h,
)

SAMPLE = """회원님과 카카오톡 대화
저장한 날짜 : 2026-05-11 14:30:00

2026년 5월 11일 오후 2:19
2026년 5월 11일 오후 2:25, 철수 : 안녕하세요
2026년 5월 11일 오후 2:25, 철수 : 둘째줄
연속줄입니다
2026년 5월 11일 오후 2:26, 오픈채팅봇 : 홍보는 금지입니다
2026년 5월 11일 오후 2:27, 영희님이 들어왔습니다.
2026년 5월 11일 오후 2:28, 영희 : 메시지가 삭제되었습니다.
2026년 5월 11일 오후 2:30, 영희 : 반가워요
"""


def _write_sample(tmp_path):
    p = tmp_path / "chat.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


def test_to_24h():
    assert to_24h("오전", 12) == 0
    assert to_24h("오전", 9) == 9
    assert to_24h("오후", 12) == 12
    assert to_24h("오후", 2) == 14


def test_parse_kakao_filters_and_multiline(tmp_path):
    msgs = parse_kakao(_write_sample(tmp_path))
    # 헤더/날짜 구분선/봇/입장 이벤트/삭제 마커는 전부 제거
    assert [m.sender for m in msgs] == ["철수", "철수", "영희"]
    assert msgs[1].text == "둘째줄 연속줄입니다"  # 멀티라인 병합
    assert msgs[0].dt == datetime(2026, 5, 11, 14, 25)
    assert [m.msg_id for m in msgs] == [0, 1, 2]  # 연속 id 재부여


def test_merge_consecutive(tmp_path):
    msgs = parse_kakao(_write_sample(tmp_path))
    merged = merge_consecutive(msgs)
    # 같은 분(시간차 0초)의 철수 연속 발화 2건이 하나로
    assert len(merged) == 2
    assert merged[0].text == "안녕하세요 둘째줄 연속줄입니다"
    assert [m.msg_id for m in merged] == [0, 1]


def test_merge_consecutive_chain_uses_previous_message_time():
    """burst 시작이 아니라 '직전 발화' 기준으로 이어붙는지 (8초×5연타 시나리오)."""
    base = datetime(2026, 5, 11, 14, 0, 0)
    msgs = [
        Message(msg_id=i, dt=base + timedelta(seconds=i * 40), sender="a", text=f"m{i}")
        for i in range(3)  # 0s, 40s, 80s — 각 간격 40s ≤ 60s
    ]
    merged = merge_consecutive(msgs, within_seconds=60)
    assert len(merged) == 1
    assert merged[0].text == "m0 m1 m2"


def test_dump_load_roundtrip(tmp_path):
    msgs = parse_kakao(_write_sample(tmp_path))
    out = tmp_path / "messages.jsonl"
    dump_messages(msgs, out)
    assert load_messages(out) == msgs


def _msg(i: int, sender: str, text: str) -> Message:
    return Message(msg_id=i, dt=datetime(2026, 5, 11, 9, i), sender=sender, text=text)


def test_anonymize_senders_and_mentions():
    msgs = [
        _msg(0, "김철수", "안녕하세요"),
        _msg(1, "영희", "김철수님 반가워요"),
        _msg(2, "김철수", "영희씨도 반가워요"),
    ]
    anon = anonymize(msgs)
    assert [m.sender for m in anon] == ["참가자1", "참가자2", "참가자1"]
    assert anon[1].text == "참가자1님 반가워요"   # 본문 내 언급도 치환
    assert anon[2].text == "참가자2씨도 반가워요"
    assert msgs[0].sender == "김철수"  # 원본 불변


def test_anonymize_longest_name_first():
    """'김철'이 '김철수'의 부분 문자열이어도 긴 이름부터 치환해 오염 방지."""
    msgs = [
        _msg(0, "김철수", "ㅎㅇ"),
        _msg(1, "김철", "김철수님 저는 김철이에요"),
    ]
    anon = anonymize(msgs)
    assert anon[1].text == "참가자1님 저는 참가자2이에요"


def test_anonymize_deterministic():
    msgs = [_msg(0, "a", "x"), _msg(1, "b", "y"), _msg(2, "a", "z")]
    assert anonymize(msgs) == anonymize(msgs)  # 재개/재현 안전


def test_anonymize_prefix_stable_for_incremental():
    """뒤에 처음 등장하는 발화자는 앞선 메시지의 치환에 영향을 주지 않아야 한다.

    (증분 처리의 공통 접두가 새 발화자 등장으로 붕괴하는 것을 방지)
    """
    old = [_msg(0, "철수", "지현아 언제 와?"), _msg(1, "영희", "몰라")]
    # 재내보내기: 지현이 뒤에서 처음 발화 (같은 접두 + 새 메시지)
    new = old + [_msg(2, "지현", "곧 가요")]
    a_old = anonymize(old)
    a_new = anonymize(new)
    # 접두(0,1)의 익명화 결과가 동일 — '지현' 언급이 옛 메시지에서 마스킹되지 않음
    assert (a_old[0].sender, a_old[0].text) == (a_new[0].sender, a_new[0].text)
    assert (a_old[1].sender, a_old[1].text) == (a_new[1].sender, a_new[1].text)
    assert a_new[0].text == "지현아 언제 와?"  # 아직 발화 전이라 미마스킹
    assert a_new[2].sender == "참가자3"


MOBILE_SAMPLE = """철수님과 카카오톡 대화
저장한 날짜 : 2026. 5. 11. 14:30

2026. 5. 11. 일요일
2026. 5. 11. 오후 2:25, 철수 : 모바일에서 보냄
2026. 5. 11. 오후 2:26, 영희 : 저도요
둘째 줄
2026. 5. 11. 오후 2:27, 오픈채팅봇 : 공지
"""


def test_parse_mobile_format(tmp_path):
    p = tmp_path / "mobile.txt"
    p.write_text(MOBILE_SAMPLE, encoding="utf-8")
    msgs = parse_kakao(p)
    assert [m.sender for m in msgs] == ["철수", "영희"]
    assert msgs[0].dt == datetime(2026, 5, 11, 14, 25)
    assert msgs[1].text == "저도요 둘째 줄"  # 멀티라인 병합 동일 동작


BRACKET_SAMPLE = """주말 등산 모임방 님과 카카오톡 대화
저장한 날짜 : 2026-07-08 09:45:04

--------------- 2026년 7월 3일 금요일 ---------------
[산지기] [오후 2:37] 이번 주말 산행 어디로 갈까요
[김초보] [오후 2:38] 저는 초보라 쉬운 코스가 좋아요
[산지기] [오후 2:38] 그럼 관악산 어때요
등산로 잘 되어 있어요
[날다람쥐] [오후 2:40] 좋아요
김초보님이 나갔습니다.
--------------- 2026년 7월 4일 토요일 ---------------
[산지기] [오전 9:05] 다들 도착하셨나요
"""


def test_parse_bracket_format(tmp_path):
    """포맷 B: 날짜는 요일 구분선에만, 메시지는 '[발신자] [시각] 본문'."""
    p = tmp_path / "bracket.txt"
    p.write_text(BRACKET_SAMPLE, encoding="utf-8")
    msgs = parse_kakao(p)
    # 헤더/저장날짜/시스템(나감) 줄은 제거, 발신자 공백 포함 인식
    assert [m.sender for m in msgs] == ["산지기", "김초보", "산지기", "날다람쥐", "산지기"]
    # 날짜는 구분선에서, 시각은 메시지에서 결합
    assert msgs[0].dt == datetime(2026, 7, 3, 14, 37)
    assert msgs[2].text == "그럼 관악산 어때요 등산로 잘 되어 있어요"  # 멀티라인 병합
    # 두 번째 날짜 헤더 이후 메시지는 다음 날짜로
    assert msgs[4].dt == datetime(2026, 7, 4, 9, 5)


def test_weekday_date_header_not_swallowed(tmp_path):
    """요일 날짜 헤더가 직전 메시지 연속 줄로 오인되지 않아야 한다."""
    sample = (
        "2026년 5월 11일 오후 2:25, 철수 : 첫 메시지\n"
        "2026년 5월 12일 화요일\n"
        "2026년 5월 12일 오전 9:00, 철수 : 다음날 메시지\n"
    )
    p = tmp_path / "chat.txt"
    p.write_text(sample, encoding="utf-8")
    msgs = parse_kakao(p)
    assert msgs[0].text == "첫 메시지"  # 날짜 헤더가 본문에 섞이지 않음
    assert len(msgs) == 2
