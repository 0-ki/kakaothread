"""LLM 슬롯 선택기 — 우선순위 티어 + 라운드로빈 + rate limit/쿼터 + 에러 쿨다운.

라운드로빈 단위는 Slot(계정×모델). 각 슬롯이 독립 카운터를 가지므로
같은 계정의 여러 모델(예: Cerebras 3모델)도 쿼터가 따로 논다.

- priority 오름차순 티어를 순회, 가장 높은(작은 숫자) 티어부터 사용
- 같은 티어 안에서는 라운드로빈 포인터로 부하 분산
- rate limit(rpm/rpd/tpm/tph/tpd) 초과 또는 에러 쿨다운 중인 슬롯은 후보에서 제외
  (rpd=일일 요청 상한 — OpenRouter 무료처럼 '요청 수/일'로 제한하는 프로바이더용)
- rpm 은 '슬롯별 최소 간격(60/rpm 초)'으로도 페이싱한다 — 분당 5회를 버스트로
  몰아 쏘면 429가 나기 쉬우므로 고르게 편다. 같은 티어 슬롯 3개면
  A,B,C → (간격 대기) → A,B,C 형태로 자연히 번갈아 나간다.
- 모든 슬롯이 막혔으면 가장 빨리 풀리는 시점까지 sleep 후 재시도 (wait=True)

동시성: 스레드 락 없음. 순차 루프(LangGraph)와 asyncio(세션 병렬) 모두 지원 —
asyncio 는 단일 스레드 협력 스케줄링이라 acquire 의 '선택+예약' 사이에 await 가
없으면 원자적이므로 별도 락이 필요 없다. (멀티스레드에서 쓰려면 락 추가 필요)
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from .config import Slot

logger = logging.getLogger(__name__)

_MINUTE, _HOUR, _DAY = 60.0, 3600.0, 86400.0

# 레이트리밋 '문자열' 신호 — 상태코드로 못 잡을 때의 보조 판정용.
# 주의: 바로 "limit"/"exceed" 같은 넓은 단어는 쓰지 않는다. reasoning 모델이
# 출력 길이 한도에 걸려 잘리는 "length limit was reached" 를 rate 로 오분류해
# 불필요한 60초 쿨다운을 걸던 버그가 있었다.
_RATE_HINTS = ("429", "too many requests", "rate limit", "rate_limit",
               "quota", "ratelimit")


def _status_code(exc: Exception) -> int | None:
    """예외에서 HTTP 상태코드를 최대한 추출 (openai/httpx 계열 대응)."""
    for attr in ("status_code", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    val = getattr(resp, "status_code", None)
    return val if isinstance(val, int) else None


def _is_rate_limit(exc: Exception) -> bool:
    """429/레이트리밋 계열이면 True (긴 쿨다운 대상).

    판정 순서: (1) 상태코드 429, (2) 예외 타입명에 RateLimit,
    (3) 좁게 지정한 문자열 신호. 길이 초과·파싱 실패 등은 일반 에러로 둔다.
    """
    if _status_code(exc) == 429:
        return True
    if "ratelimit" in type(exc).__name__.lower():
        return True
    msg = str(exc).lower()
    return any(h in msg for h in _RATE_HINTS)


class _State:
    """슬롯별 런타임 사용 기록."""
    __slots__ = ("req_times", "tok_events", "cooldown_until")

    def __init__(self) -> None:
        # 요청 타임스탬프(monotonic). rpm(분)·rpd(일)를 같은 deque에서 세려고
        # 가장 긴 윈도우(하루) 기준으로 보관한다.
        self.req_times: deque[float] = deque()
        self.tok_events: deque[tuple[float, int]] = deque()  # (타임스탬프, 총 토큰)
        self.cooldown_until: float = 0.0


class ProviderPool:
    def __init__(
        self,
        slots: list[Slot],
        *,
        wait: bool = True,
        error_cooldown: float = 10.0,
        rate_cooldown: float = 60.0,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if not slots:
            raise ValueError(
                "사용 가능한 LLM 슬롯이 없습니다 (.env 의 CEREBRAS_API_KEY 또는 LLM_SLOTS 확인).")
        self.slots = slots
        self.state = {s.name: _State() for s in slots}
        self.wait = wait
        self.error_cooldown = error_cooldown
        self.rate_cooldown = rate_cooldown
        self._clock = clock  # 테스트에서 가짜 시계 주입용
        self._sleep = sleep
        self.tiers = sorted({s.priority for s in slots})
        self._rr = {p: 0 for p in self.tiers}  # 티어별 라운드로빈 포인터

    # ── 내부: 윈도우 계산 ────────────────────────────────────────────
    def _now(self) -> float:
        return self._clock()

    def _prune(self, st: _State, now: float) -> None:
        while st.req_times and now - st.req_times[0] > _DAY:
            st.req_times.popleft()
        while st.tok_events and now - st.tok_events[0][0] > _DAY:
            st.tok_events.popleft()

    @staticmethod
    def _tok_sum(st: _State, now: float, window: float) -> int:
        return sum(t for ts, t in st.tok_events if now - ts <= window)

    @staticmethod
    def _req_count(st: _State, now: float, window: float) -> int:
        return sum(1 for ts in st.req_times if now - ts <= window)

    def _available(self, slot: Slot, now: float) -> bool:
        st = self.state[slot.name]
        if now < st.cooldown_until:
            return False
        self._prune(st, now)
        if slot.rpm:
            if self._req_count(st, now, _MINUTE) >= slot.rpm:
                return False
            # 페이싱: 분당 N회를 버스트가 아니라 60/N초 간격으로 고르게
            if st.req_times and now - st.req_times[-1] < _MINUTE / slot.rpm:
                return False
        if slot.rpd and len(st.req_times) >= slot.rpd:  # req_times는 하루 윈도우
            return False
        if slot.tpm and self._tok_sum(st, now, _MINUTE) >= slot.tpm:
            return False
        if slot.tph and self._tok_sum(st, now, _HOUR) >= slot.tph:
            return False
        if slot.tpd and self._tok_sum(st, now, _DAY) >= slot.tpd:
            return False
        return True

    def _pick(self, now: float) -> Slot | None:
        for p in self.tiers:
            # 전체 티어 목록 기준으로 포인터를 돌려, 사용 가능한 첫 슬롯을 고른다.
            # (가용 슬롯만 추려 인덱싱하면 목록이 줄어들 때 순서가 뒤틀린다)
            tier = [s for s in self.slots if s.priority == p]
            for i in range(len(tier)):
                s = tier[(self._rr[p] + i) % len(tier)]
                if self._available(s, now):
                    self._rr[p] = (self._rr[p] + i + 1) % len(tier)
                    return s
        return None

    def _sleep_hint(self, now: float) -> float:
        """가장 빨리 available 될 때까지의 대략적 대기(초)."""
        best: float | None = None
        for s in self.slots:
            st = self.state[s.name]
            waits = []
            if now < st.cooldown_until:
                waits.append(st.cooldown_until - now)
            self._prune(st, now)
            if s.rpm and st.req_times:
                minute = [t for t in st.req_times if now - t <= _MINUTE]
                if len(minute) >= s.rpm:
                    waits.append(_MINUTE - (now - minute[0]))
                else:  # 페이싱 간격이 덜 지난 경우
                    waits.append(max(0.0, _MINUTE / s.rpm - (now - st.req_times[-1])))
            if s.rpd and len(st.req_times) >= s.rpd:  # 일일 상한 소진 — 긴 대기(아래서 상한 캡)
                waits.append(_DAY - (now - st.req_times[0]))
            # 토큰 윈도우는 예측이 어려워 짧게 재확인
            w = max(waits) if waits else 0.0
            best = w if best is None else min(best, w)
        return min(max(best or 0.5, 0.5), _MINUTE)

    # ── 외부 API ─────────────────────────────────────────────────────
    def acquire(self) -> Slot:
        """사용 가능한 슬롯을 골라 rpm 카운트를 예약해 반환.

        전부 막혔으면 wait=True 면 풀릴 때까지 sleep, 아니면 예외.
        """
        while True:
            now = self._now()
            slot = self._pick(now)
            if slot is not None:
                self.state[slot.name].req_times.append(now)  # 요청 슬롯 예약
                return slot
            if not self.wait:
                raise RuntimeError("모든 LLM 슬롯이 쿨다운/쿼터 소진 상태입니다.")
            nap = self._sleep_hint(now)
            logger.info("모든 슬롯 대기 중 — %.1fs 후 재시도", nap)
            self._sleep(nap)

    async def acquire_async(self) -> Slot:
        """acquire 의 asyncio 버전 — 대기가 이벤트 루프를 막지 않는다.

        슬롯 선택+예약(_pick~append)은 await 없이 수행되므로 원자적.
        """
        while True:
            now = self._now()
            slot = self._pick(now)
            if slot is not None:
                self.state[slot.name].req_times.append(now)
                return slot
            if not self.wait:
                raise RuntimeError("모든 LLM 슬롯이 쿨다운/쿼터 소진 상태입니다.")
            nap = self._sleep_hint(now)
            logger.info("모든 슬롯 대기 중 — %.1fs 후 재시도", nap)
            await asyncio.sleep(nap)

    def on_success(self, slot: Slot, total_tokens: int) -> None:
        self.state[slot.name].tok_events.append((self._now(), total_tokens))

    def on_error(self, slot: Slot, exc: Exception) -> None:
        is_rate = _is_rate_limit(exc)
        cd = self.rate_cooldown if is_rate else self.error_cooldown
        self.state[slot.name].cooldown_until = self._now() + cd
        logger.warning("슬롯 %s 쿨다운 %.0fs (%s)", slot.name, cd, "rate" if is_rate else "error")
