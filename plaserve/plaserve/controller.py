"""Paper Algorithm 2 with a defined policy for negative pool pressures."""

from dataclasses import dataclass
import math

from .model import SimConfig


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (also used in output metrics)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


@dataclass(frozen=True)
class Signal:
    worker: int
    kind: str
    backlog: float
    deadline_deviation: float
    utilization: float
    idle: bool = True


class PressureController:
    def __init__(self, config: SimConfig):
        self.config = config
        self.last_move = -math.inf

    def decide(self, now: float, signals: list[Signal]) -> tuple[int, str] | None:
        c = self.config
        if now - self.last_move < c.cooldown:
            return None
        pools = {kind: [s for s in signals if s.kind == kind]
                 for kind in ("short", "long")}
        if any(not pool for pool in pools.values()):
            return None
        pressure = {kind: percentile([
            c.pressure_queue * s.backlog
            + c.pressure_deadline * s.deadline_deviation
            - c.pressure_utilization * s.utilization for s in pool], 0.9)
            for kind, pool in pools.items()}
        high = max(pressure, key=pressure.get)
        low = "long" if high == "short" else "short"
        # For positive pressures this is the paper's multiplicative comparison.
        # The difference form avoids false oscillation when both are negative.
        gap = pressure[high] - pressure[low]
        if gap <= c.hysteresis * max(abs(pressure[low]), 1e-9):
            return None
        if len(pools[low]) <= c.min_pool:
            return None
        # Prefer an idle worker. A busy worker must drain before reassignment.
        donor = min(pools[low], key=lambda s: (not s.idle, s.backlog, s.worker))
        self.last_move = now
        return donor.worker, high
