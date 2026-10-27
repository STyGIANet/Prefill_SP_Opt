# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""LAPS dual queues, Adaptive Wait-Depth, and instance-pressure control."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json

from sglang.srt.dynamic_sp.plan import PrefillPlan, PromptStep, local_positions


@dataclass(frozen=True)
class LAPSConfig:
    instance_degree: int = 1
    length_threshold: int = 256
    long_chunk_tokens: int = 8192
    short_token_budget: int = 16384
    graph_lengths: tuple = (8, 16, 32, 64, 128, 256)
    graph_batches: tuple = (1, 2, 4, 8, 16, 32, 64)
    wait_min_seconds: float = 0.0
    wait_max_seconds: float = 0.005
    max_queue_wait_seconds: float = 0.05
    deadline_seconds: float | None = None
    slack_seconds: float = 0.001
    safety_seconds: float = 0.001
    initial_service_seconds: float = 0.005
    control_period_seconds: float = 1.0
    cooldown_seconds: float = 5.0
    hysteresis: float = 0.2
    minimum_instances: int = 1
    queue_weight: float = 1.0
    deadline_weight: float = 1.0
    utilization_weight: float = 1.0
    utilization_signal: str = "gpu"
    # Optional independently fitted (alpha, beta, gamma_w, gamma_r), Sec. 2.1.
    boundary_coefficients: tuple | None = None

    def __post_init__(self):
        if self.utilization_signal not in ("gpu", "occupancy"):
            raise ValueError("LAPS utilization_signal must be gpu or occupancy")
        for name in (
            "instance_degree",
            "length_threshold",
            "long_chunk_tokens",
            "short_token_budget",
            "minimum_instances",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"LAPS {name} must be a positive integer")
        for name in (
            "wait_min_seconds",
            "wait_max_seconds",
            "max_queue_wait_seconds",
            "slack_seconds",
            "safety_seconds",
            "control_period_seconds",
            "cooldown_seconds",
            "hysteresis",
            "initial_service_seconds",
            "queue_weight",
            "deadline_weight",
            "utilization_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"LAPS {name} must be finite and nonnegative")
        if (
            self.wait_max_seconds < self.wait_min_seconds
            or self.control_period_seconds <= 0
            or self.max_queue_wait_seconds <= 0
        ):
            raise ValueError("Invalid LAPS waiting/control bounds")
        if self.deadline_seconds is not None and (
            not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0
        ):
            raise ValueError("LAPS deadline_seconds must be positive or null")
        for values in (self.graph_lengths, self.graph_batches):
            if (
                not values
                or tuple(sorted(set(values))) != tuple(values)
                or any(type(n) is not int or n <= 0 or n & (n - 1) for n in values)
            ):
                raise ValueError(
                    "LAPS graph dimensions must be ascending distinct powers of two"
                )
        if self.boundary_coefficients is not None:
            if (
                len(self.boundary_coefficients) != 4
                or any(
                    not math.isfinite(x) or x < 0 for x in self.boundary_coefficients
                )
                or self.boundary_coefficients[0] <= 0
            ):
                raise ValueError(
                    "LAPS boundary coefficients require alpha > 0 and nonnegative finite costs"
                )

    @classmethod
    def read(cls, path=None):
        raw = json.loads(Path(path).read_text()) if path else {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown LAPS options: {sorted(unknown)}")
        for key in ("graph_lengths", "graph_batches", "boundary_coefficients"):
            if raw.get(key) is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def boundary(self, history=0):
        if self.boundary_coefficients is None:
            return self.length_threshold
        a, b, w, r = self.boundary_coefficients
        if history == 0:
            return max(0.0, (w - b) / a)
        term = 2 * a * history + b - w
        root = math.sqrt(term * term + 4 * a * r * history)
        # Rationalized positive root avoids cancellation at large history.
        return (
            2 * r * history / (root + term)
            if term >= 0 and root + term > 0
            else max(0.0, (root - term) / (2 * a))
        )


class LAPSScheduler:
    def __init__(self, config, options=None):
        self.config, self.options = config, options or LAPSConfig()
        o = self.options
        if config.world_size % o.instance_degree:
            raise ValueError(
                "LAPS instance_degree must divide the prefill worker count"
            )
        self.instances = [
            tuple(range(i, i + o.instance_degree))
            for i in range(0, config.world_size, o.instance_degree)
        ]
        n = len(self.instances)
        if n > 1 and 2 * o.minimum_instances > n:
            raise ValueError("LAPS minimum_instances leaves no instance for one pool")
        self.roles = [
            "short" if i < max(o.minimum_instances, n // 2) else "long"
            for i in range(n)
        ]
        feasible_depths = [
            b
            for b in o.graph_batches
            if b * min(o.graph_lengths)
            <= min(config.max_local_tokens, o.short_token_budget)
        ]
        if not feasible_depths:
            raise ValueError("LAPS has no memory-feasible graph shape")
        self.depth = [max(feasible_depths)] * n
        self.window = [o.wait_max_seconds] * n
        self.epochs = {}
        self.seen = set()
        self.arrival_times = []
        self.service = {
            "short": o.initial_service_seconds,
            "long": o.initial_service_seconds,
        }
        self.busy_intervals = [[] for _ in self.instances]
        self.dispatched = {}
        self.last_control = None
        self.last_move = -math.inf
        self.read_utilization = None
        self.moves = 0
        self.graph_batches = 0
        self.standard_batches = 0
        self.last_decision = {}
        self.wake_at = None
        self.arrivals_exhausted = False

    def metadata(self):
        return {
            "algorithm": "laps",
            "paper": "2601.11589v2",
            "config": asdict(self.options),
            "roles": self.roles,
            "controller_moves": self.moves,
            "graph_candidate_batches": self.graph_batches,
            "standard_batches": self.standard_batches,
            "service_seconds": self.service,
        }

    def _short(self, r):
        # Classify the request, not each long-prefill chunk.
        return len(
            r.prompt.token_ids
        ) - r.prompt.prefix_length <= self.options.boundary(r.prompt.prefix_length)

    def _deadline(self, r):
        slo = (
            getattr(r.prompt, "ttft_slo_seconds", None) or self.options.deadline_seconds
        )
        return r.submitted_at + slo if slo is not None else math.inf

    def completed_operation(self, operation, responses):
        if not operation.plan.steps or not all(
            r["status"] == "committed" for r in responses
        ):
            return
        indices = [
            i
            for i, ranks in enumerate(self.instances)
            if set(operation.members).intersection(ranks)
        ]
        start = min(r["started_at"] for r in responses)
        end = max(r["ready_at"] for r in responses)
        previous = {p.request_id: cursor for p, _, cursor in operation.previous}
        admissions = {p.request_id: p for p in operation.plan.admit}
        advances = any(
            step.end
            > previous.get(
                step.request_id,
                admissions[step.request_id].prefix_length
                if step.request_id in admissions
                else 0,
            )
            for step in operation.plan.steps
        )
        category = self.dispatched.get(operation.plan.steps[0].request_id, "long")
        if advances:
            self.service[category] = 0.8 * self.service[category] + 0.2 * (end - start)
        for index in indices:
            self.busy_intervals[index].append((start, end))
        for step in operation.plan.steps:
            self.dispatched.pop(step.request_id, None)
            self.dispatched.pop(("start", step.request_id), None)

    def _controller(self, state, pending, occupied):
        o, now = self.options, state.now
        if self.last_control is None:
            self.last_control = now
            return
        if (
            now - self.last_control < o.control_period_seconds
            or len(self.instances) == 1
        ):
            return
        duration = now - self.last_control
        utilization = (
            self.read_utilization()
            if o.utilization_signal == "gpu" and self.read_utilization is not None
            else None
        )
        if o.utilization_signal == "gpu" and utilization is None:
            raise RuntimeError("LAPS GPU utilization sampling is not configured")
        pressures = {"short": [], "long": []}
        for category in pressures:
            indices = [i for i, role in enumerate(self.roles) if role == category]
            queue = [
                r
                for r in pending
                if ("short" if self._short(r) else "long") == category
            ]
            # Global queues are virtually routed round-robin for per-instance
            # backlog and deadline signals. No KV or worker is assigned here.
            for offset, i in enumerate(indices):
                assigned = queue[offset :: len(indices)]
                q = len(assigned)
                e = max(
                    (
                        max(0.0, now + self.service[category] - self._deadline(r))
                        for r in assigned
                    ),
                    default=0.0,
                )
                intervals = self.busy_intervals[i]
                used = sum(
                    max(0.0, min(end, now) - max(start, self.last_control))
                    for start, end in intervals
                )
                live = [
                    self.dispatched.get(("start", r.prompt.request_id))
                    for r in state.requests
                    if r.in_flight and set(r.ranks).intersection(self.instances[i])
                ]
                starts = [x for x in live if x is not None]
                if starts:
                    used += max(0.0, now - max(min(starts), self.last_control))
                u = (
                    min(1.0, used / duration)
                    if utilization is None
                    else sum(utilization[r] for r in self.instances[i])
                    / len(self.instances[i])
                )
                pressures[category].append(
                    o.queue_weight * q
                    + o.deadline_weight * e
                    - o.utilization_weight * u
                )
                self.busy_intervals[i] = [(s, e) for s, e in intervals if e > now]
        self.last_control = now
        if now - self.last_move < o.cooldown_seconds or not pending:
            return
        p = {
            key: sorted(values)[math.ceil(0.9 * len(values)) - 1]
            for key, values in pressures.items()
        }
        target = None
        if p["short"] > (1 + o.hysteresis) * p["long"]:
            target = "short"
        elif p["long"] > (1 + o.hysteresis) * p["short"]:
            target = "long"
        if target:
            source = "long" if target == "short" else "short"
            candidates = [i for i, role in enumerate(self.roles) if role == source]
            if len(candidates) > o.minimum_instances:
                idle = [
                    i
                    for i in candidates
                    if not occupied.intersection(self.instances[i])
                ]
                if idle:
                    i = idle[-1]
                    self.roles[i] = target
                    self.epochs.pop(i, None)
                    self.last_move = now
                    self.moves += 1

    def next_plan(self, state):
        c, o, now = self.config, self.options, state.now
        self.wake_at = None
        busy = set(state.busy_ranks)
        children = {
            r.prompt.prefix_request_id
            for r in state.requests
            if r.prompt.prefix_request_id and not r.ranks
        }
        release = tuple(
            r.prompt.request_id
            for r in state.requests
            if r.complete
            and not r.in_flight
            and not busy.intersection(r.ranks)
            and (
                not r.prompt.retain_kv
                or (self.arrivals_exhausted and r.prompt.request_id not in children)
            )
        )
        resident = [r for r in state.requests if r.ranks and not r.complete]
        pending = sorted(
            (
                r
                for r in state.requests
                if not r.ranks and not r.in_flight and not r.complete
            ),
            key=lambda r: (r.submitted_at, r.prompt.request_id),
        )
        capacity = c.max_local_tokens * o.instance_degree
        for r in pending:
            if len(r.prompt.token_ids) > capacity:
                raise ValueError(
                    f"LAPS prompt {r.prompt.request_id} needs {len(r.prompt.token_ids)} KV slots; fixed instance capacity is {capacity}. Increase instance_degree or max-local-tokens; inputs are never truncated."
                )
            if r.prompt.request_id not in self.seen:
                self.seen.add(r.prompt.request_id)
                if self._short(r):
                    self.arrival_times.append(r.submitted_at)
        self.arrival_times = self.arrival_times[-256:]
        occupied = busy | {rank for r in state.requests for rank in r.ranks}
        self._controller(state, pending, occupied)
        by_id = {r.prompt.request_id: r for r in state.requests}
        for child in pending:
            if child.prompt.prefix_request_id is None:
                continue
            parent = by_id.get(child.prompt.prefix_request_id)
            if (
                parent is None
                or not parent.complete
                or parent.in_flight
                or busy.intersection(parent.ranks)
            ):
                continue
            category = "short" if self._short(child) else "long"
            for i, ranks in enumerate(self.instances):
                if len(self.instances) > 1 and self.roles[i] != category:
                    continue
                if busy.intersection(ranks) or any(
                    set(r.ranks).intersection(ranks)
                    for r in state.requests
                    if r.prompt.request_id != parent.prompt.request_id
                    and not r.complete
                ):
                    continue
                owners = (
                    (ranks[0],)
                    if category == "short"
                    and len(child.prompt.token_ids) <= c.max_local_tokens
                    else ranks
                )
                if len(child.prompt.token_ids) > len(owners) * c.max_local_tokens:
                    continue
                if any(
                    sum(
                        len(local_positions(r.end, r.ranks, rank))
                        for r in state.requests
                        if r.ranks and r.prompt.request_id != parent.prompt.request_id
                    )
                    + len(local_positions(len(child.prompt.token_ids), owners, rank))
                    > c.max_local_tokens
                    for rank in owners
                ):
                    continue
                self.dispatched[child.prompt.request_id] = category
                self.dispatched[("start", child.prompt.request_id)] = now
                return PrefillPlan(
                    state.version,
                    (
                        PromptStep(
                            child.prompt.request_id, owners, child.prompt.prefix_length
                        ),
                    ),
                    (child.prompt,),
                    tuple(dict.fromkeys((*release, parent.prompt.request_id))),
                )
        # An unready continuation cannot execute as an independent cold prompt.
        pending = [r for r in pending if r.prompt.prefix_request_id is None]
        short = [r for r in pending if self._short(r)]
        long = [r for r in pending if not self._short(r)]
        steps, admit, decisions = [], [], []
        for i, ranks in enumerate(self.instances):
            if busy.intersection(ranks) or any(
                r.in_flight and set(r.ranks).intersection(ranks) for r in state.requests
            ):
                continue
            # Cleanup is a separate transaction so slot reuse never precedes commit.
            if any(
                r.prompt.request_id in release and set(r.ranks).intersection(ranks)
                for r in state.requests
            ):
                continue
            active = [r for r in resident if set(r.ranks).intersection(ranks)]
            continuations = [r for r in active if self._short(r)]
            if continuations:
                if any(len(r.ranks) > 1 for r in continuations):
                    # Historical KV cannot fit the single-worker graph budget.
                    batch = continuations[:1]
                else:
                    available = c.max_local_tokens - sum(
                        len(local_positions(r.end, r.ranks, ranks[0]))
                        for r in state.requests
                        if r.ranks
                    )
                    batch = self._short_batch(i, continuations + short, available, now)
                for request in batch:
                    owners = request.ranks or (ranks[0],)
                    steps.append(
                        PromptStep(
                            request.prompt.request_id,
                            owners,
                            len(request.prompt.token_ids),
                            use_graph=len(owners) == 1,
                        )
                    )
                    if not request.ranks:
                        admit.append(request.prompt)
                        short.remove(request)
                    self.dispatched[request.prompt.request_id] = "short"
                    self.dispatched[("start", request.prompt.request_id)] = now
                if batch:
                    decisions.append((i, "short", len(batch)))
                continue
            category = (
                self.roles[i]
                if len(self.instances) > 1
                else ("short" if short else "long")
            )
            if len(self.instances) == 1 and (active or long):
                oldest_long = min(
                    active or long, key=lambda r: (self._deadline(r), r.submitted_at)
                )
                long_due = (
                    self._deadline(oldest_long) - now - self.service["long"]
                    <= o.slack_seconds
                )
                if long_due and (
                    not short
                    or self._deadline(oldest_long)
                    < min(self._deadline(r) for r in short)
                ):
                    category = "long"
            if active and len(self.instances) > 1:
                category = "long"
            if not short:
                self.epochs.pop(i, None)
            if category == "short" and short:
                available = c.max_local_tokens - sum(
                    len(local_positions(len(r.prompt.token_ids), r.ranks, ranks[0]))
                    for r in state.requests
                    if r.ranks
                )
                batch = self._short_batch(i, short, available, now)
                if batch:
                    for r in batch:
                        n = len(r.prompt.token_ids)
                        steps.append(
                            PromptStep(
                                r.prompt.request_id, (ranks[0],), n, use_graph=True
                            )
                        )
                        admit.append(r.prompt)
                        short.remove(r)
                    decisions.append((i, "short", len(batch)))
                elif len(self.instances) == 1:
                    category = "long"  # A long chunk may execute during the short wait.
            if category == "long":
                r = active[0] if active else (long.pop(0) if long else None)
                if (
                    r is not None
                    and not r.ranks
                    and any(
                        sum(
                            len(local_positions(other.end, other.ranks, rank))
                            for other in state.requests
                            if other.ranks
                        )
                        + len(local_positions(len(r.prompt.token_ids), ranks, rank))
                        > c.max_local_tokens
                        for rank in ranks
                    )
                ):
                    r = None
                if r is not None and not r.in_flight:
                    steps.append(
                        PromptStep(
                            r.prompt.request_id,
                            ranks,
                            min(len(r.prompt.token_ids), r.end + o.long_chunk_tokens),
                        )
                    )
                    if not r.ranks:
                        admit.append(r.prompt)
                    decisions.append((i, "long", 1))
            for step in steps:
                if set(step.ranks).issubset(ranks):
                    self.dispatched[step.request_id] = category
                    self.dispatched[("start", step.request_id)] = now
        self.last_decision = {
            "batches": decisions,
            "roles": list(self.roles),
            "window_seconds": list(self.window),
            "target_depth": list(self.depth),
        }
        if (
            not steps
            and not release
            and self.arrivals_exhausted
            and not busy
            and not any(r.in_flight for r in state.requests)
        ):
            waiting_children = [
                r for r in state.requests if not r.ranks and r.prompt.prefix_request_id
            ]
            if waiting_children and not resident and not pending:
                raise ValueError(
                    "Retained sessions exhaust LAPS destination capacity; increase max-local-tokens or instance_degree"
                )
        if not steps and not release:
            self.wake_at = min(self.wake_at or math.inf, now + o.control_period_seconds)
            return None
        return PrefillPlan(state.version, tuple(steps), tuple(admit), release)

    def _short_batch(self, i, queue, capacity, now):
        o = self.options
        head = min(queue, key=lambda r: (self._deadline(r), r.submitted_at))
        length = next(
            (
                n
                for n in o.graph_lengths
                if n >= len(head.prompt.token_ids) - head.prompt.prefix_length
            ),
            None,
        )
        # Bucket-first, retaining deadline/arrival ordering within the bucket.
        candidates = sorted(
            queue,
            key=lambda r: (
                0
                if length is not None
                and (len(r.prompt.token_ids) - r.prompt.prefix_length) <= length
                and (
                    length == o.graph_lengths[0]
                    or (len(r.prompt.token_ids) - r.prompt.prefix_length) > length // 2
                )
                else 1,
                self._deadline(r),
                r.submitted_at,
            ),
        )
        batch, tokens = [], 0
        for r in candidates:
            n = len(r.prompt.token_ids) - r.prompt.prefix_length
            if tokens + n > min(capacity, o.short_token_budget):
                continue
            if batch and length is not None and n > length:
                continue
            batch.append(r)
            tokens += n
            if len(batch) >= self.depth[i]:
                break
        if not batch:
            if len(head.prompt.token_ids) - head.prompt.prefix_length <= capacity:
                self.standard_batches += 1
                self.epochs.pop(i, None)
                return [head]
            self.wake_at = now + 0.001
            return []
        if i not in self.epochs:
            self.epochs[i] = now
        start = self.epochs[i]
        slack = min(self._deadline(r) - now - self.service["short"] for r in queue)
        rate = (
            (
                (len(self.arrival_times) - 1)
                / (self.arrival_times[-1] - self.arrival_times[0])
            )
            if len(self.arrival_times) > 1
            and self.arrival_times[-1] > self.arrival_times[0]
            else 0.0
        )
        graph_wait = max(0, self.depth[i] - len(batch)) / max(rate, 1e-9)
        wait = max(
            o.wait_min_seconds,
            min(self.window[i], graph_wait, max(0.0, slack - o.safety_seconds)),
        )
        due = min(
            start + wait, min(r.submitted_at for r in queue) + o.max_queue_wait_seconds
        )
        ready = (
            len(batch) >= self.depth[i]
            or tokens >= o.short_token_budget
            or now >= due
            or slack <= o.slack_seconds
        )
        if not ready:
            self.wake_at = min(self.wake_at or math.inf, due)
            return []
        d = len(batch)
        if d >= self.depth[i]:
            self.window[i] = min(
                o.wait_max_seconds, max(o.wait_min_seconds, now - start)
            )
        else:
            self.depth[i] = max(1, d)
        self.epochs.pop(i, None)
        b = next((n for n in o.graph_batches if n >= d), None)
        length = next(
            (
                n
                for n in o.graph_lengths
                if n
                >= max(
                    (len(r.prompt.token_ids) - r.prompt.prefix_length) for r in batch
                )
            ),
            None,
        )
        if b and length and b * length <= min(capacity, o.short_token_budget):
            self.graph_batches += 1
        else:
            self.standard_batches += 1
        return batch
