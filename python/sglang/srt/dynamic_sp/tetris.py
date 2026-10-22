# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Tetris prefill planner, Algorithms 1-3 of arXiv:2511.06247v1.

The planner operates on independent instance calendars. TetrisScheduler adapts
its decisions to either independent group execution or synchronized steps.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sglang.srt.dynamic_sp.plan import PrefillPlan, PromptStep, local_positions
from sglang.srt.dynamic_sp.reservations import ReservationCalendar
from sglang.srt.dynamic_sp.scheduler import SchedulerConfig, SchedulerInput


@dataclass(frozen=True)
class LatencyModel:
    a: float
    b: float
    c: float
    d: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.a, self.b, self.c, self.d)):
            raise ValueError("Latency coefficients must be finite and nonnegative")
        if self.b == self.c == self.d == 0:
            raise ValueError("Latency model must increase with chunk length")

    def seconds(self, history: int, length: int) -> float:
        if history < 0 or length < 0:
            raise ValueError("Token counts must be nonnegative")
        return (
            0.0
            if length == 0
            else self.a
            + (self.b + self.c * history) * length
            + self.d * length * length
        )

    def tokens(self, history: int, budget: float, limit: int) -> int:
        """Largest integer chunk fitting the budget, including constant overhead.

        Integer bisection avoids cancellation in the quadratic formula and handles
        linear models. Complexity is logarithmic in prompt length.
        """
        low, high = 0, limit
        while low < high:
            mid = (low + high + 1) // 2
            if self.seconds(history, mid) <= budget:
                low = mid
            else:
                high = mid - 1
        return low


@dataclass(frozen=True)
class Chunk:
    history: int
    length: int
    ranks: tuple[int, ...]
    start: float
    finish: float


class CDSPPlanner:
    """Full recursive CDSP search with monotone instance-group expansion.

    Times are relative to this planning invocation. Queues are absolute within
    that coordinate system; advancing the cursor implements Eq. (2) without
    double-counting previous chunks. Memoization does not prune candidates.
    """

    def __init__(self, models, nodes, min_chunk_tokens=1):
        self.models = dict(sorted(models.items()))
        self.nodes = tuple(nodes)
        self.min_chunk_tokens = min_chunk_tokens
        if not self.nodes or any(type(n) is not int or n < 0 for n in self.nodes):
            raise ValueError("nodes must map every rank to a nonnegative node ID")
        if not self.models or any(
            type(s) is not int or not 1 <= s <= len(self.nodes) for s in self.models
        ):
            raise ValueError("SP degrees must fit the instance pool")
        if type(min_chunk_tokens) is not int or min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be positive")

    def group(self, pool, initial, degree, queues):
        """GetGroup: prefer existing nodes, then shortest-queued whole nodes.

        Single-node choices minimize the s-th order statistic. Cross-node choices
        take complete available nodes before a partial final node. Rank ID breaks
        ties deterministically. Existing rank order is retained.
        """
        if degree < len(initial) or not set(initial) <= set(pool) or degree > len(pool):
            return None
        chosen = list(initial)
        existing_nodes = {self.nodes[r] for r in initial}
        same = sorted(
            (r for r in pool if r not in chosen and self.nodes[r] in existing_nodes),
            key=lambda r: (queues[r], r),
        )
        chosen.extend(same[: degree - len(chosen)])
        while len(chosen) < degree:
            need = degree - len(chosen)
            by_node = {}
            for r in pool:
                if r not in chosen:
                    by_node.setdefault(self.nodes[r], []).append(r)
            groups = [
                sorted(rs, key=lambda r: (queues[r], r)) for rs in by_node.values()
            ]
            fitting = [rs for rs in groups if len(rs) >= need]
            if fitting:
                best = min(
                    fitting, key=lambda rs: (queues[rs[need - 1]], tuple(rs[:need]))
                )
                chosen.extend(best[:need])
            else:
                best = min(
                    groups, key=lambda rs: (max(queues[r] for r in rs), tuple(rs))
                )
                chosen.extend(best)
        return tuple(chosen)

    def plan(
        self,
        length,
        queues,
        *,
        history=0,
        initial=(),
        improvement_rate=0.0,
        feasible=None,
    ):
        if (
            type(length) is not int
            or length <= 0
            or type(history) is not int
            or history < 0
        ):
            raise ValueError(
                "Expected positive remaining length and nonnegative history"
            )
        queues = tuple(queues)
        if len(queues) != len(self.nodes) or any(
            not math.isfinite(t) or t < 0 for t in queues
        ):
            raise ValueError("Expected one finite nonnegative queue time per rank")
        if not math.isfinite(improvement_rate) or not 0 <= improvement_rate < 1:
            raise ValueError("improvement_rate must be in [0, 1)")
        if len(set(initial)) != len(initial) or any(
            r not in range(len(queues)) for r in initial
        ):
            raise ValueError("Invalid initial group")
        fits = feasible or (lambda ranks: True)

        @lru_cache(None)
        def search(remaining, prefix, old, candidates, pool, cursor):
            best = None
            # Algorithm 2: sequential comparison with a strict improvement gate.
            for s in candidates:
                ranks = self.group(pool, old, s, queues)
                if ranks is None or not fits(ranks):
                    continue
                start = max(cursor, max(queues[r] for r in ranks))
                finish = start + self.models[s].seconds(prefix, remaining)
                if best is None or finish - cursor < (best[-1].finish - cursor) * (
                    1 - improvement_rate
                ):
                    best = (Chunk(prefix, remaining, ranks, start, finish),)
            if best is None:
                return None
            initial_best = best[0]
            sizes = tuple(
                s for s in candidates if len(old) <= s <= len(initial_best.ranks)
            )
            # Algorithms 1 and 3: fit the current chunk in the queue-time gap.
            for index, current in enumerate(sizes):
                for following in sizes[index + 1 :]:
                    ranks = self.group(initial_best.ranks, old, current, queues)
                    next_ranks = self.group(
                        initial_best.ranks, ranks, following, queues
                    )
                    if not fits(ranks):
                        continue
                    start = max(cursor, max(queues[r] for r in ranks))
                    next_start = max(cursor, max(queues[r] for r in next_ranks))
                    count = self.models[current].tokens(
                        prefix, next_start - start, remaining
                    )
                    if count < self.min_chunk_tokens or count >= remaining:
                        continue
                    finish = start + self.models[current].seconds(prefix, count)
                    rest = search(
                        remaining - count,
                        prefix + count,
                        ranks,
                        tuple(s for s in sizes if s >= following),
                        initial_best.ranks,
                        finish,
                    )
                    if rest is not None and rest[-1].finish < best[-1].finish:
                        best = (Chunk(prefix, count, ranks, start, finish),) + rest
            return best

        return search(
            length,
            history,
            tuple(initial),
            tuple(self.models),
            tuple(range(len(queues))),
            0.0,
        )


class ArrivalWindow:
    """Track observed submissions once, including requests completed in-window."""

    def __init__(self, seconds, unit="tokens"):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Arrival window must be finite and positive")
        self.seconds = seconds
        self.unit = unit
        self.events = []
        self.live = set()
        self.last_version = -1

    def observe(self, state):
        # Failed execution/repeated next_plan calls must not count arrivals twice.
        if state.version < self.last_version:
            raise ValueError("Scheduler observations must be monotonic")
        current = {r.prompt.request_id for r in state.requests}
        for r in state.requests:
            rid = r.prompt.request_id
            if rid not in self.live:
                self.events.append((r.submitted_at, len(r.prompt.token_ids)))
        self.live = current
        self.last_version = state.version
        self.events = [
            event for event in self.events if event[0] > state.now - self.seconds
        ]
        return (
            sum(
                (n if self.unit == "tokens" else 1)
                for t, n in self.events
                if t <= state.now
            )
            / self.seconds
        )


class TetrisScheduler:
    """Online prefill policy for synchronous or independent group execution.

    Rebuild a FIFO virtual calendar from committed state and busy-worker
    availability. Dispatch ready first chunks on disjoint groups. Predictions
    never count as actual progress; request/KV changes require successful commit.
    """

    def __init__(self, config: SchedulerConfig, profile):
        self.config = config
        self.profile = profile
        self.models = {
            int(s): LatencyModel(**values) for s, values in profile["models"].items()
        }
        nodes = profile.get("nodes", [0] * config.world_size)
        if len(nodes) != config.world_size:
            raise ValueError("Profile topology does not match world size")
        self.planner = CDSPPlanner(
            self.models, nodes, profile.get("min_chunk_tokens", 1)
        )
        self.window = ArrivalWindow(profile["arrival_window_seconds"], unit="requests")
        self.calendar = ReservationCalendar(config.world_size)
        self.update_interval = profile.get("threshold_update_seconds", 30.0)
        if not math.isfinite(self.update_interval) or self.update_interval <= 0:
            raise ValueError("Threshold update interval must be positive")
        self.next_update = None
        self.improvement_rate = None
        self.thresholds = tuple(
            (float(row["requests_per_second"]), float(row["improvement_rate"]))
            for row in profile["thresholds"]
        )
        if (
            not self.thresholds
            or self.thresholds[0][0] != 0
            or any(
                not math.isfinite(load)
                or load < 0
                or not math.isfinite(rate)
                or not 0 <= rate < 1
                for load, rate in self.thresholds
            )
            or any(a[0] >= b[0] for a, b in zip(self.thresholds, self.thresholds[1:]))
        ):
            raise ValueError(
                "Thresholds must start at zero with strictly increasing request rates and improvement rates in [0, 1)"
            )
        self.last_decision = {}
        self.execution_mode = "sync"

    @classmethod
    def from_file(cls, config, path):
        profile = json.loads(Path(path).read_text())
        if profile.get("schema_version") != 2:
            raise ValueError(
                "Unsupported Tetris profile schema; refit for persistent reservations"
            )
        return cls(config, profile)

    def metadata(self):
        canonical = json.dumps(self.profile, sort_keys=True).encode()
        return {
            "profile": self.profile,
            "profile_sha256": hashlib.sha256(canonical).hexdigest(),
            "execution": self.execution_mode,
            "last_decision": self.last_decision,
        }

    def next_plan(self, state: SchedulerInput):
        c = self.config
        load = self.window.observe(state)
        if self.next_update is None or state.now >= self.next_update:
            self.improvement_rate = min(
                self.thresholds, key=lambda row: (abs(row[0] - load), row[0])
            )[1]
            self.next_update = state.now + self.update_interval
        rate = self.improvement_rate
        busy = set(state.busy_ranks)
        views = {r.prompt.request_id: r for r in state.requests}
        available = self.calendar.observe(state)
        for rid in tuple(self.calendar.requests):
            if rid not in views:
                self.calendar.retire(rid)
        decisions = []
        # FIFO arrivals receive complete, persistent chunk reservations. Neither
        # resident priority nor per-dispatch token budgets change the paper plan.
        for request in sorted(state.requests, key=lambda r: r.submitted_at):
            rid = request.prompt.request_id
            if request.complete or rid in self.calendar.requests:
                continue
            if request.ranks or request.end:
                raise ValueError(
                    "Tetris requires the complete request lifecycle for reservations"
                )
            available = self.calendar.predict(state.now, state.worker_available_at)
            chunks = self.planner.plan(
                len(request.prompt.token_ids),
                [max(0.0, t - state.now) for t in available],
                improvement_rate=rate,
            )
            if chunks is None:
                raise RuntimeError(
                    "No Tetris plan exists for the profiled SP candidates"
                )
            self.calendar.reserve(rid, chunks, self.models, state.now)
            decisions.append(
                {"request_id": rid, "chunks": [vars(chunk) for chunk in chunks]}
            )
        release = tuple(
            r.prompt.request_id
            for r in state.requests
            if r.complete and not r.in_flight and not busy.intersection(r.ranks)
        )
        steps, admit = [], []
        # Releasing a finished wide group must not connect two independent next
        # requests into one transaction through its old owners. Dispatch cleanup
        # separately; unrelated groups can still advance while it is in flight.
        ready = () if release else self.calendar.ready(busy)
        for task in ready:
            request = views[task.request_id]
            if request.in_flight or busy.intersection(request.ranks):
                continue
            if request.end != task.history:
                raise RuntimeError("Committed progress differs from the reserved chunk")
            steps.append(PromptStep(task.request_id, task.ranks, task.end))
            if not request.ranks:
                admit.append(request.prompt)
        # Capacity is a feasibility check, never a silent policy modification.
        # Insufficient memory must fail the experiment rather than substitute
        # different degrees, chunks, or request priorities under Tetris's name.
        placements = {
            r.prompt.request_id: (
                r.ranks,
                r.end if r.reserved_end is None else r.reserved_end,
            )
            for r in state.requests
            if r.ranks and r.prompt.request_id not in release
        }
        for step in steps:
            placements[step.request_id] = (step.ranks, step.end)
        for rank in range(c.world_size):
            if (
                sum(
                    len(local_positions(end, ranks, rank))
                    for ranks, end in placements.values()
                )
                > c.max_local_tokens
            ):
                raise RuntimeError(
                    "Tetris reserved plan exceeds KV capacity; increase max-local-tokens or reduce load. The baseline does not alter its plan for memory limits."
                )
        self.last_decision = {
            "version": state.version,
            "observed_requests_per_second": load,
            "improvement_rate": rate,
            "next_threshold_update": self.next_update,
            "plans": decisions,
            "reservations": [
                vars(t).copy()
                for t in self.calendar.tasks.values()
                if t.status != "completed"
            ],
        }
        return (
            PrefillPlan(state.version, tuple(steps), tuple(admit), release)
            if steps or release
            else None
        )
