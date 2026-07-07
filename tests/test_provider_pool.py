"""슬롯 풀 테스트 — 페이싱, 라운드로빈, 우선순위 티어, 쿨다운, 쿼터 (가짜 시계 사용)."""
import pytest

from kakaothread.config import Connection, Slot
from kakaothread.provider_pool import ProviderPool, _is_rate_limit

CONN = Connection(name="TEST", base_url="", api_key="k")


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


def _slot(name: str, **kw) -> Slot:
    return Slot(name=name, conn=CONN, model=name, **kw)


def _pool(slots: list[Slot], clock: FakeClock, *, wait: bool = False, **kw) -> ProviderPool:
    return ProviderPool(slots, wait=wait, clock=clock, sleep=clock.advance, **kw)


def test_empty_slots_raises():
    with pytest.raises(ValueError):
        ProviderPool([])


def test_rpm_paces_requests_evenly():
    """rpm=5 → 같은 슬롯은 12초 간격으로만 나간다 (버스트 금지)."""
    clock = FakeClock()
    pool = _pool([_slot("a", rpm=5)], clock)
    assert pool.acquire().name == "a"
    with pytest.raises(RuntimeError):  # 간격이 안 지났으므로 막힘
        pool.acquire()
    clock.advance(12)
    assert pool.acquire().name == "a"


def test_same_tier_round_robin():
    """같은 우선순위 3슬롯이면 A,B,C 순으로 번갈아 나간다."""
    clock = FakeClock()
    slots = [_slot(n, priority=1, rpm=5) for n in ("a", "b", "c")]
    pool = _pool(slots, clock)
    assert [pool.acquire().name for _ in range(3)] == ["a", "b", "c"]
    with pytest.raises(RuntimeError):  # 셋 다 페이싱 대기
        pool.acquire()
    clock.advance(12)
    assert [pool.acquire().name for _ in range(3)] == ["a", "b", "c"]


def test_priority_tier_failover():
    """상위 티어가 막히면 하위 티어로 내려간다."""
    clock = FakeClock()
    pool = _pool([_slot("free", priority=1, rpm=5), _slot("paid", priority=2)], clock)
    assert pool.acquire().name == "free"
    assert pool.acquire().name == "paid"  # free 는 페이싱 대기 중
    clock.advance(12)
    assert pool.acquire().name == "free"  # 풀리면 다시 상위 티어


def test_error_cooldown():
    clock = FakeClock()
    pool = _pool([_slot("a")], clock, error_cooldown=10, rate_cooldown=60)
    slot = pool.acquire()
    pool.on_error(slot, Exception("connection reset"))
    with pytest.raises(RuntimeError):
        pool.acquire()
    clock.advance(10.1)
    assert pool.acquire().name == "a"


def test_rate_limit_error_gets_longer_cooldown():
    clock = FakeClock()
    pool = _pool([_slot("a")], clock, error_cooldown=10, rate_cooldown=60)
    slot = pool.acquire()
    pool.on_error(slot, Exception("Error code: 429 - rate limit exceeded"))
    clock.advance(10.1)  # 일반 쿨다운으론 부족
    with pytest.raises(RuntimeError):
        pool.acquire()
    clock.advance(50)
    assert pool.acquire().name == "a"


def test_length_limit_is_not_rate_limit():
    """reasoning 모델의 출력 길이 초과는 rate 가 아니라 일반 에러 (짧은 쿨다운).

    실제 로그의 'length limit was reached' 가 'limit' 때문에 rate 로 오분류돼
    60초 쿨다운이 반복되던 버그의 회귀 방지.
    """
    assert _is_rate_limit(Exception(
        "Could not parse response content as the length limit was reached "
        "- CompletionUsage(completion_tokens=5413, total_tokens=8192)")) is False


def test_is_rate_limit_detection():
    # 상태코드 기반
    class RateErr(Exception):
        status_code = 429

    class ParseErr(Exception):
        status_code = 400

    assert _is_rate_limit(RateErr("whatever")) is True
    assert _is_rate_limit(ParseErr("bad request")) is False
    # 타입명 기반 (openai.RateLimitError 계열)
    assert _is_rate_limit(type("RateLimitError", (Exception,), {})()) is True
    # 문자열 기반
    assert _is_rate_limit(Exception("429 Too Many Requests")) is True
    assert _is_rate_limit(Exception("quota exceeded for today")) is True
    assert _is_rate_limit(Exception("connection reset by peer")) is False


def test_rpd_daily_cap_blocks_and_fails_over():
    """일일 요청 상한(rpd) 소진 시 하위 티어로 넘어간다 (OpenRouter 무료 시나리오)."""
    clock = FakeClock()
    pool = _pool(
        [_slot("free", priority=1, rpm=0, rpd=3), _slot("paid", priority=2)], clock
    )
    # free 를 하루 상한(3회)까지 소진
    for _ in range(3):
        assert pool.acquire().name == "free"
    # 이후엔 free 가 막혀 paid 로 (rpm/페이싱 제약이 없으므로 계속 paid)
    assert pool.acquire().name == "paid"
    assert pool.acquire().name == "paid"


def test_rpd_resets_after_a_day():
    clock = FakeClock()
    pool = _pool([_slot("a", rpm=0, rpd=2)], clock, wait=False)
    assert pool.acquire().name == "a"
    assert pool.acquire().name == "a"
    with pytest.raises(RuntimeError):
        pool.acquire()
    clock.advance(86_401)  # 하루 경과 → 윈도우에서 빠짐
    assert pool.acquire().name == "a"


def test_rpm_counts_only_within_minute_not_whole_day():
    """rpd 도입 후 req_times 가 하루치라도, rpm 은 '최근 1분' 요청만 센다.

    (분·일 카운트를 같은 deque에서 뽑으므로 회귀 방지)
    """
    clock = FakeClock()
    pool = _pool([_slot("a", rpm=2, rpd=100)], clock, wait=False)
    pool.acquire()
    clock.advance(30)
    pool.acquire()          # 1분 안에 2회 → 상한 도달
    with pytest.raises(RuntimeError):
        pool.acquire()
    clock.advance(31)       # 첫 요청이 1분 밖으로 → 분당 카운트 1로 감소
    assert pool.acquire().name == "a"


def test_tpm_quota_blocks_until_window_passes():
    clock = FakeClock()
    pool = _pool([_slot("a", tpm=100)], clock)
    slot = pool.acquire()
    pool.on_success(slot, total_tokens=100)
    with pytest.raises(RuntimeError):
        pool.acquire()
    clock.advance(61)
    assert pool.acquire().name == "a"


def test_wait_sleeps_until_available():
    """wait=True 면 예외 대신 (주입된) sleep 으로 풀릴 때까지 기다린다."""
    clock = FakeClock()
    pool = _pool([_slot("a", rpm=5)], clock, wait=True)
    assert pool.acquire().name == "a"
    assert pool.acquire().name == "a"  # sleep(=clock.advance) 이 시간을 흘려보냄
    assert clock.t >= 12
