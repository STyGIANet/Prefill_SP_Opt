"""Length-separated batching, adaptive wait/depth, and chunk admission.

The scheduler does not execute kernels; the simulator and GPU runner supply
completion times. One worker executes one batch at a time.
"""

from dataclasses import dataclass, field
import math

from .controller import PressureController, Signal
from .model import GraphCatalog, GraphShape, LatencyModel, Request, SimConfig

POLICIES = ("fcfs", "graphs", "disaggregation", "full")


@dataclass
class Work:
    request: Request
    kind: str
    offset: int = 0
    first_start: float | None = None
    service: float = 0.0

    @property
    def remaining(self) -> int:
        return self.request.tokens - self.offset

    @property
    def history(self) -> int:
        return self.request.history + self.offset


@dataclass
class AdaptiveState:
    window: float
    depth: int


@dataclass
class Worker:
    id: int
    kind: str
    busy: bool = False
    pending_kind: str | None = None
    adaptive: dict[int, AdaptiveState] = field(default_factory=dict)
    busy_total: float = 0.0
    busy_since: float | None = None
    measured_service: float | None = None


@dataclass
class Batch:
    worker: int
    kind: str
    work: list[Work]
    lengths: list[int]
    histories: list[int]
    shape: GraphShape | None
    kv_tokens: int
    start: float
    estimated_service: float
    bucket: int | None = None
    fill_time: float = 0.0
    target: int = 1


class Scheduler:
    def __init__(self, config: SimConfig, model: LatencyModel, policy: str):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        self.config, self.model, self.policy = config, model, policy
        self.separate = policy in ("disaggregation", "full")
        self.graphs = policy in ("graphs", "full")
        self.catalog = GraphCatalog(config)
        self.queues: dict[str, list[Work]] = {k: [] for k in ("mixed", "short", "long")}
        short_workers = max(config.min_pool, config.workers // 2)
        self.workers = [Worker(i, "mixed" if not self.separate or config.workers == 1
                               else "short" if i < short_workers else "long")
                        for i in range(config.workers)]
        self.controller = PressureController(config)
        self.completed: list[dict] = []
        self.migrations: list[dict] = []
        self.seen: set[str] = set()
        self.last_short_arrival: float | None = None
        self.short_rate: float = 0.0

    def classify(self, request: Request) -> str:
        threshold = (self.model.boundary(request.history) if self.config.classify_by_model
                     else self.config.short_threshold)
        return "short" if request.tokens <= threshold else "long"

    def submit(self, request: Request) -> None:
        if request.id in self.seen:
            raise ValueError(f"duplicate request id: {request.id}")
        if request.tokens + request.history > self.config.kv_token_budget:
            raise ValueError(f"request {request.id}: context exceeds per-worker KV token budget")
        self.seen.add(request.id)
        kind = self.classify(request)
        self.queues[kind if self.separate else "mixed"].append(Work(request, kind))
        if kind == "short":
            if self.last_short_arrival is not None and request.arrival > self.last_short_arrival:
                rate = 1 / (request.arrival - self.last_short_arrival)
                self.short_rate = rate if self.short_rate == 0 else 0.8 * self.short_rate + 0.2 * rate
            self.last_short_arrival = request.arrival

    def _priority(self, work: Work, now: float) -> tuple:
        # Deadline urgency comes first; aging protects older work otherwise.
        length = min(work.remaining, self.config.long_chunk) if work.kind == "long" else work.remaining
        service = self.model.duration([length], [work.history])
        urgent = (work.request.deadline is not None
                  and work.request.deadline - now <= service + self.config.slack)
        aged = now - work.request.arrival >= self.config.max_queue_wait
        return (not urgent, not aged, work.request.deadline if work.request.deadline is not None
                else math.inf, work.request.arrival, work.request.id)

    def _short_candidate(self, worker: Worker, now: float) -> tuple[Batch | None, float]:
        queue = self.queues["short"]
        if not queue:
            return None, math.inf
        c = self.config
        head = min(queue, key=lambda w: self._priority(w, now))
        bucket = self.catalog.bucket(head.remaining)
        similar = sorted((w for w in queue if self.catalog.bucket(w.remaining) == bucket),
                         key=lambda w: self._priority(w, now))
        selected: list[Work] = []
        kv = 0
        for work in similar:
            cost = work.history + work.remaining
            prospective = kv + cost
            if c.eager_padding:
                prospective = (sum(w.history for w in selected) + work.history
                               + (len(selected) + 1) * max([w.remaining for w in selected] + [work.remaining]))
            if len(selected) < c.max_batch and prospective <= c.kv_token_budget:
                selected.append(work)
                kv = prospective
        lengths = [w.remaining for w in selected]
        histories = [w.history for w in selected]
        state = worker.adaptive.setdefault(bucket, AdaptiveState(
            c.wait_max, self.catalog.target(bucket, [head.history])))
        target = min(state.depth, self.catalog.target(bucket, histories), c.max_batch)
        selected = selected[:target]
        lengths, histories = lengths[:target], histories[:target]
        shape = self.catalog.nearest(lengths, histories) if self.graphs else None
        eager_lengths = [max(lengths)] * len(lengths) if c.eager_padding else lengths
        estimate = self.model.duration(eager_lengths, histories, shape)
        # Actual GPU observations bound deadline estimates once available.
        deadline_service = max(estimate, worker.measured_service or 0)
        oldest = min(w.request.arrival for w in selected)
        ready = now
        if self.policy == "full" and len(selected) < target:
            graph_wait = ((target - len(selected)) / self.short_rate
                          if self.short_rate > 0 else c.wait_max)
            window = max(c.wait_min, min(c.wait_max, state.window, graph_wait))
            ready = min(oldest + window, oldest + c.max_queue_wait)
            deadlines = [w.request.deadline for w in selected if w.request.deadline is not None]
            if deadlines:
                safe = min(deadlines) - deadline_service - c.safety
                ready = min(ready, safe)
                if min(deadlines) - now - deadline_service <= c.slack:
                    ready = now
            # Memory saturation is another feasible dispatch boundary.
            if len(selected) < target and len(selected) < len(similar):
                ready = now
        if ready > now + 1e-12:
            return None, ready
        allocated = (shape.kv_tokens(histories) if shape else sum(histories) + sum(eager_lengths))
        batch = Batch(worker.id, "short", selected, lengths, histories, shape, allocated,
                      now, estimate, bucket, max(0.0, max(w.request.arrival for w in selected) - oldest), target)
        return batch, math.inf

    def _long_candidate(self, worker: Worker, now: float) -> Batch | None:
        queue = self.queues["long"]
        if not queue:
            return None
        work = min(queue, key=lambda w: self._priority(w, now))
        length = min(work.remaining, self.config.long_chunk)
        estimate = self.model.duration([length], [work.history])
        return Batch(worker.id, "long", [work], [length], [work.history], None,
                     work.history + length, now, estimate)

    def _mixed_candidate(self, worker: Worker, now: float) -> Batch | None:
        queue = self.queues["mixed"]
        if not queue:
            return None
        selected: list[Work] = []
        kv = 0
        # Unified FCFS admission, with no synthetic interference penalty.
        for work in queue:
            cost = work.remaining + work.history
            prospective = kv + cost
            if self.config.eager_padding:
                prospective = (sum(w.history for w in selected) + work.history
                               + (len(selected) + 1) * max([w.remaining for w in selected] + [work.remaining]))
            if len(selected) >= self.config.max_batch or prospective > self.config.kv_token_budget:
                break
            selected.append(work)
            kv = prospective
        lengths, histories = [w.remaining for w in selected], [w.history for w in selected]
        shape = (self.catalog.nearest(lengths, histories)
                 if self.graphs and all(w.kind == "short" for w in selected) else None)
        eager_lengths = [max(lengths)] * len(lengths) if self.config.eager_padding else lengths
        return Batch(worker.id, "mixed", selected, lengths, histories, shape,
                     shape.kv_tokens(histories) if shape else kv, now,
                     self.model.duration(eager_lengths, histories, shape))

    def next_batch(self, worker_id: int, now: float) -> tuple[Batch | None, float]:
        worker = self.workers[worker_id]
        if worker.busy:
            return None, math.inf
        if not self.separate:
            batch, wake = self._mixed_candidate(worker, now), math.inf
        elif worker.kind == "long":
            batch, wake = self._long_candidate(worker, now), math.inf
        else:
            short, wake = self._short_candidate(worker, now)
            long = self._long_candidate(worker, now) if worker.kind == "mixed" else None
            batch = short
            if long:
                short_head = min(self.queues["short"], key=lambda w: self._priority(w, now), default=None)
                long_head = long.work[0]
                if short and short_head and self._priority(long_head, now) < self._priority(short_head, now):
                    # A long chunk must not consume the short batch's deadline slack.
                    deadlines = [w.request.deadline for w in short.work if w.request.deadline is not None]
                    if not deadlines or now + long.estimated_service + short.estimated_service <= min(deadlines):
                        batch = long
                elif short is None and (wake == math.inf or now + long.estimated_service <= wake):
                    batch = long
        if batch:
            queue = self.queues[batch.kind if self.separate else "mixed"]
            identities = {id(w) for w in batch.work}
            queue[:] = [w for w in queue if id(w) not in identities]
            worker.busy, worker.busy_since = True, now
            for work in batch.work:
                if work.first_start is None:
                    work.first_start = now
            if self.policy == "full" and batch.bucket is not None:
                state = worker.adaptive[batch.bucket]
                if len(batch.work) >= batch.target:
                    state.window = max(self.config.wait_min, min(self.config.wait_max, batch.fill_time))
                else:
                    # Literal AWD depth reduction, rounded up to a captured depth.
                    state.depth = next((d for d in self.config.graph_depths if d >= len(batch.work)),
                                       len(batch.work))
            return batch, math.inf
        return None, wake

    def finish(self, batch: Batch, now: float) -> None:
        worker = self.workers[batch.worker]
        if not worker.busy or now < batch.start:
            raise ValueError("invalid batch completion")
        elapsed = now - batch.start
        worker.busy_total += elapsed
        worker.measured_service = (elapsed if worker.measured_service is None
                                   else 0.8 * worker.measured_service + 0.2 * elapsed)
        worker.busy, worker.busy_since = False, None
        for work, length in zip(batch.work, batch.lengths):
            work.offset += length
            work.service += elapsed
            if work.remaining:
                self.queues[work.kind if self.separate else "mixed"].append(work)
            else:
                r = work.request
                self.completed.append({"id": r.id, "kind": work.kind, "arrival": r.arrival,
                                       "tokens": r.tokens, "history": r.history, "deadline": r.deadline,
                                       "first_start": work.first_start, "finish": now,
                                       "prefill_latency": now - r.arrival,
                                       "queue_time": max(0.0, now - r.arrival - work.service),
                                       "service_time": work.service,
                                       "slo_violation": r.deadline is not None and now > r.deadline})
        if worker.pending_kind:
            old = worker.kind
            worker.kind, worker.pending_kind = worker.pending_kind, None
            self.migrations.append({"time": now, "worker": worker.id,
                                    "from": old, "to": worker.kind, "event": "applied"})

    def rebalance(self, now: float) -> None:
        if self.policy != "full" or self.config.workers < 2:
            return
        if any(w.pending_kind for w in self.workers):
            return
        signals = []
        for worker in self.workers:
            pool_size = sum(w.kind == worker.kind for w in self.workers)
            queue = self.queues[worker.kind]
            overdue = sum(max(0.0, now - w.request.deadline)
                          for w in queue if w.request.deadline is not None)
            busy = worker.busy_total + (now - worker.busy_since if worker.busy_since is not None else 0)
            # Global queues are apportioned evenly; deadline deviation is scaled
            # by max_queue_wait so weights remain dimensionless.
            signals.append(Signal(worker.id, worker.kind, len(queue) / pool_size,
                                  overdue / (pool_size * self.config.max_queue_wait),
                                  min(1.0, busy / max(now, 1e-9)), not worker.busy))
        move = self.controller.decide(now, signals)
        if move:
            worker_id, target = move
            worker = self.workers[worker_id]
            self.migrations.append({"time": now, "worker": worker_id, "from": worker.kind,
                                    "to": target, "event": "requested"})
            if worker.busy:
                worker.pending_kind = target
            else:
                old, worker.kind = worker.kind, target
                self.migrations.append({"time": now, "worker": worker_id,
                                        "from": old, "to": target, "event": "applied"})


def batch_record(batch: Batch, finish: float) -> dict:
    return {"worker": batch.worker, "kind": batch.kind, "start": batch.start,
            "finish": finish, "ids": [w.request.id for w in batch.work],
            "classes": [w.kind for w in batch.work], "lengths": batch.lengths,
            "histories": batch.histories, "kv_tokens": batch.kv_tokens,
            "shape": {"length": batch.shape.length, "depth": batch.shape.depth} if batch.shape else None,
            "target_depth": batch.target, "fill_time": batch.fill_time}
