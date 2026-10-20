# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Online scheduling policy and immutable observations, independent of transport."""

from dataclasses import dataclass
from typing import Protocol

from sglang.srt.dynamic_sp.plan import PrefillPlan, Prompt, PromptStep, local_positions


@dataclass(frozen=True)
class SchedulerConfig:
    world_size: int
    max_local_tokens: int
    chunk_tokens: int = 1024
    max_running_requests: int = 32
    max_batch_tokens: int = 32768

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError(
                "Scheduler dimensions and limits must be positive integers"
            )


@dataclass(frozen=True)
class RequestView:
    prompt: Prompt
    submitted_at: float
    ranks: tuple[int, ...] = ()
    end: int = 0
    last_scheduled_version: int = -1

    @property
    def complete(self):
        return self.end == len(self.prompt.token_ids)


@dataclass(frozen=True)
class SchedulerInput:
    version: int
    now: float
    requests: tuple[RequestView, ...]
    # Rank-ordered observations from the last committed step. Tuples keep the
    # observation immutable; policies may convert them to dictionaries locally.
    worker_metrics: tuple[tuple[tuple[str, object], ...], ...] = ()
    last_step_seconds: float | None = None


class SchedulingPolicy(Protocol):
    def next_plan(self, state: SchedulerInput) -> PrefillPlan | None: ...


class GreedyScheduler:
    """Capacity-aware online heuristic, not a claim of an optimal scheduler.

    Prefer existing work, rotate requests by last service, and admit FIFO into
    vacant slots. Target ceil(workers / runnable requests) workers per prompt,
    expanding further when capacity requires it. Choose lightly loaded workers
    and retain the old placement when equally sized and similarly balanced.
    Reserve full-prompt KV capacity to avoid admitting work that cannot finish.
    This reserve concerns KV tokens, not transient device allocation bytes.
    All policy decisions live in next_plan(); it has no speculative mutation.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def next_plan(self, state: SchedulerInput) -> PrefillPlan | None:
        c = self.config
        release = tuple(r.prompt.request_id for r in state.requests if r.complete)
        resident = [r for r in state.requests if r.ranks and not r.complete]
        pending = [r for r in state.requests if not r.ranks]
        candidates = sorted(
            resident,
            key=lambda r: (
                r.last_scheduled_version,
                r.submitted_at,
                r.prompt.request_id,
            ),
        )
        candidates += sorted(pending, key=lambda r: r.submitted_at)
        runnable = min(len(candidates), c.max_running_requests, c.max_batch_tokens)
        if not runnable:
            return PrefillPlan(state.version, release=release) if release else None
        desired = (c.world_size + runnable - 1) // runnable
        reserved = [0] * c.world_size
        load = [0] * c.world_size

        def charges(request, ranks):
            return [
                len(local_positions(len(request.prompt.token_ids), ranks, rank))
                for rank in range(c.world_size)
            ]

        for request in resident:
            for rank, count in enumerate(charges(request, request.ranks)):
                reserved[rank] += count
        steps, admit = [], []
        running, budget = len(resident), c.max_batch_tokens
        for request in candidates:
            if not budget:
                break
            if not request.ranks and running >= c.max_running_requests:
                continue
            length = len(request.prompt.token_ids)
            if length > c.world_size * c.max_local_tokens:
                raise ValueError(
                    f"Prompt {request.prompt.request_id} exceeds aggregate KV capacity"
                )
            end = min(length, request.end + c.chunk_tokens, request.end + budget)
            old = (
                charges(request, request.ranks) if request.ranks else [0] * c.world_size
            )
            available = [reserved[r] - old[r] for r in range(c.world_size)]

            def fits(ranks, available=available, request=request):
                return all(
                    a + b <= c.max_local_tokens
                    for a, b in zip(available, charges(request, ranks))
                )

            def score(ranks, end=end, request=request):
                return max(
                    load[r]
                    + len(local_positions(end, ranks, r))
                    - len(local_positions(request.end, ranks, r))
                    for r in range(c.world_size)
                )

            order = sorted(
                range(c.world_size),
                key=lambda r: (load[r], available[r], r not in request.ranks, r),
            )
            ranks = None
            for degree in range(desired, c.world_size + 1):
                chosen = []
                for offset in range(degree):
                    required = len(range(offset, length, degree))
                    worker = next(
                        (
                            r
                            for r in order
                            if r not in chosen
                            and available[r] + required <= c.max_local_tokens
                        ),
                        None,
                    )
                    if worker is None:
                        break
                    chosen.append(worker)
                if len(chosen) == degree:
                    ranks = tuple(chosen)
                    break
            if (
                request.ranks
                and fits(request.ranks)
                and (
                    ranks is None
                    or (
                        len(request.ranks) == len(ranks)
                        and score(request.ranks) <= 1.1 * score(ranks)
                    )
                )
            ):
                ranks = request.ranks
            if ranks is None:
                continue
            for rank, count in enumerate(charges(request, ranks)):
                reserved[rank] = available[rank] + count
                load[rank] += len(local_positions(end, ranks, rank)) - len(
                    local_positions(request.end, ranks, rank)
                )
            steps.append(PromptStep(request.prompt.request_id, ranks, end))
            budget -= end - request.end
            if not request.ranks:
                admit.append(request.prompt)
                running += 1
        if not steps and not release:
            raise RuntimeError(
                "No request fits the configured KV capacity; scheduler made no progress"
            )
        return PrefillPlan(state.version, tuple(steps), tuple(admit), release)


class OnlineScheduler:
    """Rank-zero request lifecycle. Submit at any boundary; observe only commits.

    Custom policies implement next_plan(SchedulerInput). They may change degree,
    membership and chunk end, pause by omission, or evict by releasing unfinished
    requests. Eviction requeues the prompt and recomputes its prefix on readmission.
    A failed execute must not call committed(); the pending queue stays intact.
    """

    def __init__(self, policy: SchedulingPolicy):
        self.policy = policy
        self._requests: dict[str, RequestView] = {}
        self.version = 0
        self.worker_metrics = ()
        self.last_step_seconds = None

    def submit(self, prompt: Prompt, now: float):
        if prompt.request_id in self._requests:
            raise ValueError(f"Duplicate live request {prompt.request_id}")
        if not prompt.token_ids:
            raise ValueError("Cannot submit an empty prompt")
        self._requests[prompt.request_id] = RequestView(prompt, now)

    def next_plan(self, now: float):
        state = SchedulerInput(
            self.version,
            now,
            tuple(self._requests.values()),
            self.worker_metrics,
            self.last_step_seconds,
        )
        plan = self.policy.next_plan(state)
        if plan is None and self._requests:
            raise RuntimeError("Policy returned stop while requests remain")
        if plan is not None:
            if plan.version != self.version:
                raise ValueError("Policy returned a stale version")
            if not (plan.steps or plan.admit or plan.release):
                raise ValueError("Policy returned an empty plan")
            for p in plan.admit:
                if (
                    p.request_id not in self._requests
                    or self._requests[p.request_id].prompt != p
                    or self._requests[p.request_id].ranks
                ):
                    raise ValueError("Policy admission must match a pending request")
        return plan

    def committed(self, plan: PrefillPlan, snapshot, seconds: float, worker_metrics):
        if plan.version != self.version:
            raise ValueError("Unexpected committed plan version")
        requests = dict(self._requests)
        for rid in plan.release:
            old = requests[rid]
            if old.complete:
                del requests[rid]
            else:
                requests[rid] = RequestView(old.prompt, old.submitted_at)
        advanced = {step.request_id for step in plan.steps}
        for rid, values in snapshot.items():
            old = requests[rid]
            requests[rid] = RequestView(
                old.prompt,
                old.submitted_at,
                tuple(values["ranks"]),
                values["end"],
                plan.version if rid in advanced else old.last_scheduled_version,
            )
        self._requests = requests
        self.version += 1
        self.last_step_seconds = seconds
        self.worker_metrics = tuple(tuple(sorted(m.items())) for m in worker_metrics)
