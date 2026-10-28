# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Committed request state and reservations for out-of-order group completions."""

from dataclasses import replace
from types import SimpleNamespace

from sglang.srt.dynamic_sp.async_executor import Operation
from sglang.srt.dynamic_sp.engine import DynamicSPPrefillEngine
from sglang.srt.dynamic_sp.plan import PrefillPlan
from sglang.srt.dynamic_sp.scheduler import RequestView, SchedulerInput


class AsyncScheduler:
    def __init__(self, policy, config, model):
        self.policy, self.config, self.model = policy, config, model
        self.requests = {}
        self.version = 0
        self.next_ticket = 0
        self.inflight = {}
        self.projected = {}
        self.releasing = set()
        self.predicted_finish = {}
        self.last_step_seconds = None
        if hasattr(policy, "execution_mode"):
            policy.execution_mode = "async"

    def submit(self, prompt, now):
        if prompt.request_id in self.requests or not prompt.token_ids:
            raise ValueError("Duplicate live or empty prompt")
        self.requests[prompt.request_id] = RequestView(prompt, now)
        self.version += 1

    def observation(self, now, busy):
        views = []
        for rid, request in self.requests.items():
            ranks = (
                self.projected[rid].ranks if rid in self.projected else request.ranks
            )
            views.append(
                replace(
                    request,
                    ranks=() if rid in self.releasing else ranks,
                    reserved_end=self.projected[rid].end
                    if rid in self.projected
                    else None,
                    in_flight=rid in self.inflight,
                )
            )
        available = tuple(
            max(now, self.predicted_finish.get(r, now))
            for r in range(self.config.world_size)
        )
        return SchedulerInput(
            self.version,
            now,
            tuple(views),
            last_step_seconds=self.last_step_seconds,
            busy_ranks=tuple(sorted(busy)),
            worker_available_at=available,
        )

    def next_plan(self, now, busy):
        return self.policy.next_plan(self.observation(now, busy))

    def operations(self, plan, now, busy):
        """Validate global capacity, then split independent connected components.

        Components include release and migration owners. Requests sharing any
        owner are one transaction; disjoint components commit independently.
        """
        if plan.version != self.version:
            raise ValueError("Stale async scheduling observation")
        for prompt in plan.admit:
            if (
                prompt.request_id not in self.requests
                or self.requests[prompt.request_id].prompt != prompt
            ):
                raise ValueError("Admission must match a submitted request")
        affected = {s.request_id for s in plan.steps} | set(plan.release)
        if not affected or affected.intersection(self.inflight):
            raise ValueError("Empty operation or request already in flight")
        # The existing pure validator checks token bounds, ownership and global
        # KV capacity, including the target states reserved by in-flight work.
        states = {}
        for rid, request in self.requests.items():
            if rid in self.releasing:
                continue
            projected = self.projected.get(rid)
            ranks = projected.ranks if projected else request.ranks
            end = projected.end if projected else request.end
            if ranks:
                states[rid] = SimpleNamespace(
                    prompt=request.prompt, ranks=ranks, end=end
                )
        validator = SimpleNamespace(
            version=self.version,
            _states=states,
            model=self.model,
            world_size=self.config.world_size,
            max_local_tokens=self.config.max_local_tokens,
        )
        DynamicSPPrefillEngine._validate(validator, plan)
        components = []
        for rid in sorted(affected):
            ranks = set(self.requests[rid].ranks)
            parent_id = self.requests[rid].prompt.prefix_request_id
            if parent_id is not None and rid in {p.request_id for p in plan.admit}:
                ranks.update(self.requests[parent_id].ranks)
            ranks.update(
                r for step in plan.steps if step.request_id == rid for r in step.ranks
            )
            if not ranks or ranks.intersection(busy):
                raise ValueError(
                    "Policy selected leased workers; inspect busy_ranks/in_flight"
                )
            ids = {rid}
            remaining = []
            for previous_ranks, previous_ids in components:
                if ranks.intersection(previous_ranks):
                    ranks.update(previous_ranks)
                    ids.update(previous_ids)
                else:
                    remaining.append((previous_ranks, previous_ids))
            components = remaining + [(ranks, ids)]
        operations = []
        for ranks, ids in components:
            part = PrefillPlan(
                plan.version,
                tuple(s for s in plan.steps if s.request_id in ids),
                tuple(p for p in plan.admit if p.request_id in ids),
                tuple(r for r in plan.release if r in ids),
            )
            old = tuple(
                (
                    self.requests[rid].prompt,
                    self.requests[rid].ranks,
                    self.requests[rid].end,
                )
                for rid in sorted(ids)
                if self.requests[rid].ranks
            )
            op = Operation(self.next_ticket, tuple(sorted(ranks)), part, old)
            self.next_ticket += 1
            for rid in ids:
                self.inflight[rid] = op.ticket
            self.releasing.update(part.release)
            for step in part.steps:
                self.projected[step.request_id] = step
            estimate = self.last_step_seconds or 0.001
            if hasattr(self.policy, "estimate_operation_seconds"):
                estimate = self.policy.estimate_operation_seconds(part)
            elif hasattr(self.policy, "models"):
                estimate = sum(
                    self.policy.models[len(s.ranks)].seconds(
                        self.requests[s.request_id].end,
                        s.end - self.requests[s.request_id].end,
                    )
                    for s in part.steps
                )
            for rank in ranks:
                self.predicted_finish[rank] = now + estimate
            operations.append(op)
        self.version += 1
        return operations

    def ticket(self):
        """Ticket for an operation built outside the policy, e.g. KV handoff."""
        self.next_ticket += 1
        return self.next_ticket - 1

    def hand_off(self, request_id):
        """Forget a finished prompt whose KV left every prefill owner."""
        request = self.requests[request_id]
        if not request.complete or request_id in self.inflight:
            raise ValueError("Only an idle, finished prompt can be handed off")
        del self.requests[request_id]
        self.version += 1

    def completed(self, operation, responses):
        ids = {s.request_id for s in operation.plan.steps} | set(operation.plan.release)
        if any(self.inflight.get(rid) != operation.ticket for rid in ids):
            raise ValueError("Stale or duplicate operation completion")
        if all(r["status"] == "committed" for r in responses):
            for rid in operation.plan.release:
                old = self.requests[rid]
                if old.complete:
                    del self.requests[rid]
                else:
                    self.requests[rid] = RequestView(old.prompt, old.submitted_at)
            for step in operation.plan.steps:
                self.requests[step.request_id] = replace(
                    self.requests[step.request_id],
                    ranks=step.ranks,
                    end=step.end,
                    last_scheduled_version=self.version,
                )
            self.last_step_seconds = max(r["seconds"] for r in responses)
            if hasattr(self.policy, "completed_operation"):
                self.policy.completed_operation(operation, responses)
        self.releasing.difference_update(ids)
        for rid in ids:
            self.inflight.pop(rid)
            self.projected.pop(rid, None)
        for rank in operation.members:
            self.predicted_finish.pop(rank, None)
        self.version += 1

    def snapshot(self):
        return {
            rid: {"ranks": request.ranks, "end": request.end}
            for rid, request in self.requests.items()
            if request.ranks
        }
