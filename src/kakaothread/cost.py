"""토큰 사용량 / 비용 추적 — 슬롯(모델)별로 누적, 슬롯별 단가로 환산."""
from __future__ import annotations

from .config import Slot


class UsageTracker:
    """LLM 호출의 입력/출력 토큰을 슬롯별로 누적하고 비용을 환산."""

    def __init__(self) -> None:
        # slot_name -> {"model","calls","in","out","pin","pout"}
        self.stats: dict[str, dict] = {}

    def add(self, slot: Slot, tok_in: int, tok_out: int) -> None:
        s = self.stats.get(slot.name)
        if s is None:
            s = {"model": slot.model, "calls": 0, "in": 0, "out": 0,
                 "pin": slot.price_in, "pout": slot.price_out}
            self.stats[slot.name] = s
        s["calls"] += 1
        s["in"] += tok_in
        s["out"] += tok_out

    @property
    def tok_in(self) -> int:
        return sum(s["in"] for s in self.stats.values())

    @property
    def tok_out(self) -> int:
        return sum(s["out"] for s in self.stats.values())

    @property
    def cost(self) -> float:
        return sum(s["in"] / 1e6 * s["pin"] + s["out"] / 1e6 * s["pout"]
                   for s in self.stats.values())

    def report(self, label: str = "누적") -> None:
        print(f"[{label}] in={self.tok_in:,} out={self.tok_out:,} tok  ~${self.cost:.4f}")
        for name, s in sorted(self.stats.items()):
            print(f"  - {name} ({s['model']}): calls={s['calls']} "
                  f"in={s['in']:,} out={s['out']:,}")
