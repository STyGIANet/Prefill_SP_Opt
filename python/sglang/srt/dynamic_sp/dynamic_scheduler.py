# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""CDSP request-level SP: one whole prompt per buddy-aligned block sized by length."""

import math

from sglang.srt.dynamic_sp.plan import PrefillPlan, PromptStep, local_positions


def default_sp_sizes(world_size):
    return tuple(
        1 << i for i in range(world_size.bit_length()) if world_size % (1 << i) == 0
    )


def default_size_thresholds(num_sizes):
    # 256 and 1024 measured best for Llama-3-8B on 4 GPUs (CDSP Phase A);
    # the x4 extension to larger blocks is arbitrary.
    return tuple(256 * 4**i for i in range(num_sizes - 1))


class RequestSpScheduler:
    """Port of sglang.srt.cdsp.request_planner.RequestSpPlanner.

    A block is ranks [start, start + size) with start % size == 0. A request whose
    preferred block is taken falls back to a smaller idle block, else waits while
    later requests are tried. Prompts are never chunked.
    """

    def __init__(self, config, sp_sizes=None, size_thresholds=None):
        self.config = config
        self.sp_sizes = tuple(sorted(sp_sizes or default_sp_sizes(config.world_size)))
        self.size_thresholds = tuple(
            default_size_thresholds(len(self.sp_sizes))
            if size_thresholds is None
            else size_thresholds
        )
        if any(s <= 0 or s & (s - 1) or config.world_size % s for s in self.sp_sizes):
            raise ValueError(
                f"SP sizes {self.sp_sizes} must be powers of two dividing {config.world_size}"
            )
        if len(self.size_thresholds) != len(self.sp_sizes) - 1 or list(
            self.size_thresholds
        ) != sorted(self.size_thresholds):
            raise ValueError(
                f"Need ascending thresholds, one fewer than SP sizes {self.sp_sizes}"
            )
        self.last_decision = {}

    def size_for(self, num_tokens):
        for size, threshold in zip(self.sp_sizes, self.size_thresholds):
            if num_tokens < threshold:
                return size
        return self.sp_sizes[-1]

    def metadata(self):
        return {
            "sp_sizes": self.sp_sizes,
            "size_thresholds": self.size_thresholds,
            "last_decision": self.last_decision,
        }

    def next_plan(self, state):
        # With decode GPUs, busy_ranks also lists decode ranks past world_size.
        busy = set(state.busy_ranks).intersection(range(self.config.world_size))
        release = tuple(
            r.prompt.request_id
            for r in state.requests
            if r.complete and not r.in_flight and not busy.intersection(r.ranks)
        )
        # Placed requests keep their KV until released, but only unfinished ones
        # hold workers; a block serves at most one request per plan.
        occupied = busy | {
            rank for r in state.requests if not r.complete for rank in r.ranks
        }
        kv = [0] * self.config.world_size
        for r in state.requests:
            for rank in r.ranks:
                kv[rank] += self._kv_tokens(r, r.ranks, rank)
        running = sum(1 for r in state.requests if r.ranks and not r.complete)

        steps, admit, blocks = [], [], []
        claimed = set(busy)
        for r in state.requests:
            if (
                r.ranks
                and not r.complete
                and not r.in_flight
                and claimed.isdisjoint(r.ranks)
            ):
                steps.append(self._full_step(r, r.ranks))
                claimed.update(r.ranks)
        pending = sorted(
            (
                r
                for r in state.requests
                if not r.ranks and not r.in_flight and not r.complete
            ),
            key=lambda r: (r.submitted_at, r.prompt.request_id),
        )
        for r in pending:
            if len(occupied) >= self.config.world_size:
                break
            if running >= self.config.max_running_requests:
                break
            length = len(r.prompt.token_ids)
            if length > self.sp_sizes[-1] * self.config.max_local_tokens:
                raise ValueError(
                    f"Prompt {r.prompt.request_id} exceeds the KV capacity of the "
                    f"largest block ({self.sp_sizes[-1]} workers)"
                )
            ranks = self._pick_block(r, occupied, kv)
            if ranks is None:
                continue
            for rank in ranks:
                kv[rank] += self._kv_tokens(r, ranks, rank)
            occupied.update(ranks)
            steps.append(self._full_step(r, ranks))
            admit.append(r.prompt)
            blocks.append([r.prompt.request_id, list(ranks), length])
            running += 1

        self.last_decision = {"version": state.version, "blocks": blocks}
        if not steps and not release:
            if busy or not pending:
                return None
            raise RuntimeError(
                "No request fits the configured KV capacity; scheduler made no progress"
            )
        return PrefillPlan(state.version, tuple(steps), tuple(admit), release)

    def _pick_block(self, request, occupied, kv):
        """Largest idle aligned block not above the preferred size that fits the KV."""
        desired = self.size_for(len(request.prompt.token_ids))
        for size in sorted((s for s in self.sp_sizes if s <= desired), reverse=True):
            for start in range(0, self.config.world_size, size):
                ranks = tuple(range(start, start + size))
                if occupied.isdisjoint(ranks) and all(
                    kv[rank] + self._kv_tokens(request, ranks, rank)
                    <= self.config.max_local_tokens
                    for rank in ranks
                ):
                    return ranks
        return None

    @staticmethod
    def _kv_tokens(request, ranks, rank):
        return len(local_positions(len(request.prompt.token_ids), ranks, rank))

    @staticmethod
    def _full_step(request, ranks):
        return PromptStep(
            request.prompt.request_id, ranks, len(request.prompt.token_ids)
        )


class DynamicBatchScheduler:
    """Windowed shortest-first prefill in compute-bound steps of step_tokens.

    Every window_seconds the pending prompts are sorted by length. Prompts shorter
    than step_tokens are packed into groups of at most step_tokens and each group
    runs on one GPU; every longer prompt gets the shallowest GPU group that takes
    as many steps as the widest group left for it. Planned units start in order;
    while the head waits for GPUs, idle GPUs run the shortest pending units that
    end before it can start. Prompts taking more than long_steps steps run one step
    per operation and may pause for queued short groups, in total at most
    preempt_ratio of their own duration and never k steps in a row.
    """

    def __init__(
        self,
        config,
        step_seconds,
        window_seconds=2.0,
        step_tokens=256,
        long_steps=4,
        resume_every_steps=2,
        preempt_ratio=0.2,
    ):
        if not step_seconds > 0 or not window_seconds > 0:
            raise ValueError("step_seconds and window_seconds must be positive")
        if step_tokens <= 0 or long_steps <= 0 or resume_every_steps < 1:
            raise ValueError("Need positive step_tokens and long_steps, and k >= 1")
        if not 0 <= preempt_ratio < 1:
            raise ValueError("preempt_ratio must lie in [0, 1)")
        self.config = config
        self.step_seconds = step_seconds
        self.window_seconds = window_seconds
        self.step_tokens = step_tokens
        self.long_steps = long_steps
        self.resume_every_steps = resume_every_steps
        self.preempt_ratio = preempt_ratio
        self.queue = []
        self.window_end = None
        self.longs = {}
        self.ends = {}
        self.last_decision = {}

    @classmethod
    def from_profile(cls, config, profile, step_tokens=256, **kwargs):
        """Step time is the profiled single-GPU prefill of step_tokens tokens."""
        from sglang.srt.dynamic_sp.tetris import LatencyModel

        step_seconds = LatencyModel(**profile["models"]["1"]).seconds(0, step_tokens)
        return cls(config, step_seconds, step_tokens=step_tokens, **kwargs)

    def metadata(self):
        return {
            "step_seconds": self.step_seconds,
            "window_seconds": self.window_seconds,
            "step_tokens": self.step_tokens,
            "long_steps": self.long_steps,
            "resume_every_steps": self.resume_every_steps,
            "preempt_ratio": self.preempt_ratio,
            "last_decision": self.last_decision,
        }

    def estimate_operation_seconds(self, plan):
        groups = {}
        for s in plan.steps:
            new = s.end - self.ends.get(s.request_id, 0)
            groups.setdefault(frozenset(s.ranks), []).append(new)
        return max(
            (
                self._steps(math.ceil(sum(tokens) / len(ranks))) * self.step_seconds
                for ranks, tokens in groups.items()
            ),
            default=0.0,
        )

    def next_plan(self, state):
        c = self.config
        now = state.now
        busy = set(state.busy_ranks)
        views = {r.prompt.request_id: r for r in state.requests}
        self.ends = {rid: r.end for rid, r in views.items()}
        self.longs = {
            rid: info
            for rid, info in self.longs.items()
            if rid in views and not views[rid].complete
        }
        release = tuple(
            rid
            for rid, r in views.items()
            if r.complete and not r.in_flight and not busy.intersection(r.ranks)
        )
        kv = [0] * c.world_size
        held = set()
        for rid, r in views.items():
            if not r.ranks or rid in release:
                continue
            for rank in r.ranks:
                kv[rank] += self._kv(len(r.prompt.token_ids), r.ranks, rank)
            if not r.complete or r.in_flight:
                held.update(r.ranks)
        free = sorted(set(range(c.world_size)) - busy - held)
        ready = [
            r
            for r in views.values()
            if r.ranks
            and not r.complete
            and not r.in_flight
            and not busy.intersection(r.ranks)
        ]
        pending = {
            rid: r
            for rid, r in views.items()
            if not r.ranks and not r.in_flight and not r.complete
        }
        for r in pending.values():
            if len(r.prompt.token_ids) > c.world_size * c.max_local_tokens:
                raise ValueError(
                    f"Prompt {r.prompt.request_id} exceeds aggregate KV capacity"
                )
        fresh = self._plan_window(pending, now)
        free_at = self._free_at(free, ready, state, now)

        steps, admit, started = [], [], []
        head_start = None

        def launch(unit):
            if not self._start(unit, free, kv, steps, admit):
                return False
            finish = now + self._unit_steps(unit) * self.step_seconds
            for rank in unit["ranks"]:
                free_at[rank] = finish
            started.append(unit)
            return True

        for unit in self.queue:
            if head_start is None and not launch(unit):
                head_start = sorted(free_at)[unit["size"] - 1]
                if self._reshape(unit, len(free), now, head_start) and launch(unit):
                    head_start = None
            elif head_start is not None and self._ends_by(unit, now, head_start):
                launch(unit)
        # Idle GPUs take the shortest new arrivals that fit before the waiting head.
        waiting = [u for u in self.queue if u["size"] == 1 and "ranks" not in u]
        # Size by every GPU, not the momentarily free ones, or a long prompt is pinned narrow.
        for unit in self._units(fresh, self.config.world_size):
            started_now = free and self._ends_by(unit, now, head_start) and launch(unit)
            if not started_now and unit["size"] == 1:
                waiting.append(unit)
        paused = self._run_longs(ready, waiting, kv, steps, admit, started)
        self.queue = [u for u in self.queue if "ranks" not in u]

        self.last_decision = {
            "version": state.version,
            "started": [
                {
                    "requests": [r.prompt.request_id for r in u["requests"]],
                    "tokens": u["tokens"],
                    "ranks": u.get("ranks"),
                    "lent_by": u.get("lent_by"),
                }
                for u in started
            ],
            "paused": paused,
            "queued": sum(len(u["requests"]) for u in self.queue),
            "head_start": head_start,
        }
        if not steps and not release:
            if busy or held or not pending:
                return None
            raise RuntimeError(
                "No unit fits the configured KV capacity; scheduler made no progress"
            )
        return PrefillPlan(state.version, tuple(steps), tuple(admit), release)

    def _plan_window(self, pending, now):
        """At a window boundary, queue every unplanned prompt; return the unplanned rest."""
        queued = {r.prompt.request_id for u in self.queue for r in u["requests"]}
        self.queue = [
            u
            for u in self.queue
            if all(r.prompt.request_id in pending for r in u["requests"])
        ]
        fresh = [r for rid, r in pending.items() if rid not in queued]
        if self.window_end is None:
            self.window_end = now + self.window_seconds
        if now < self.window_end:
            return fresh
        self.window_end = now + self.window_seconds
        groups = sum(1 for u in self._units(fresh, 1) if u["size"] == 1)
        gpus = self.config.world_size - min(groups, self.config.world_size)
        self.queue += self._units(fresh, gpus or self.config.world_size)
        return []

    def _units(self, requests, gpus):
        """Short prompts packed into one-GPU groups; each longer prompt its own unit."""
        length = lambda r: len(r.prompt.token_ids)
        units, group = [], []
        for r in sorted(requests, key=lambda r: (length(r), r.submitted_at)):
            if length(r) >= self.step_tokens:
                steps = self._steps(length(r))
                widest = max(1, min(steps, gpus))
                # Shallowest group that still finishes in the widest group's steps.
                size = math.ceil(steps / self._steps_on(steps, widest))
                units.append({"requests": [r], "tokens": length(r), "size": size})
            elif group and sum(map(length, group)) + length(r) > self.step_tokens:
                units.append(self._group(group))
                group = [r]
            else:
                group.append(r)
        if group:
            units.append(self._group(group))
        units.sort(key=lambda u: u["tokens"])
        return units

    def _reshape(self, unit, gpus, now, head_start):
        """Shrink a blocked prompt to the free GPUs when that finishes no later."""
        if not gpus or unit["tokens"] < self.step_tokens:
            return False
        steps = self._steps(unit["tokens"])
        size = math.ceil(steps / self._steps_on(steps, min(steps, gpus)))
        waited = head_start + self._unit_steps(unit) * self.step_seconds
        if now + self._steps_on(steps, size) * self.step_seconds > waited:
            return False
        unit["size"] = size
        return True

    def _group(self, requests):
        tokens = sum(len(r.prompt.token_ids) for r in requests)
        return {"requests": requests, "tokens": tokens, "size": 1}

    def _free_at(self, free, ready, state, now):
        """Predicted time each rank becomes free, for locating bubbles."""
        free_at = list(state.worker_available_at or [now] * self.config.world_size)
        for rank in free:
            free_at[rank] = now
        for r in ready:
            left = len(r.prompt.token_ids) - r.end
            finish = now + self._steps_on(self._steps(left), len(r.ranks)) * (
                self.step_seconds
            )
            for rank in r.ranks:
                free_at[rank] = max(free_at[rank], finish)
        return free_at

    def _ends_by(self, unit, now, deadline):
        if deadline is None:
            return True
        seconds = self._unit_steps(unit) * self.step_seconds
        return now + seconds <= deadline

    def _start(self, unit, free, kv, steps, admit, ranks=None):
        """Launch unit on the lowest free ranks (or the given ranks) if it fits."""
        if ranks is None:
            if len(free) < unit["size"]:
                return False
            ranks = tuple(free[: unit["size"]])
        if not self._fits(unit["requests"], ranks, kv):
            return False
        for rank in ranks:
            if rank in free:
                free.remove(rank)
        unit["ranks"] = ranks
        preemptible = self._unit_steps(unit) > self.long_steps
        for owners, r in self._owners(unit["requests"], ranks):
            length = len(r.prompt.token_ids)
            for rank in owners:
                kv[rank] += self._kv(length, owners, rank)
            end = length
            if preemptible:
                chunk = self.step_tokens * len(ranks)
                end = min(length, chunk)
                self.longs[r.prompt.request_id] = {
                    "chunk": chunk,
                    "budget": self.preempt_ratio
                    * self._unit_steps(unit)
                    * self.step_seconds,
                    "paused": 0,
                }
            steps.append(PromptStep(r.prompt.request_id, owners, end))
            admit.append(r.prompt)
        return True

    def _run_longs(self, ready, waiting, kv, steps, admit, started):
        """Advance long prompts one step, or pause them for waiting short groups."""
        paused = []
        for r in ready:
            rid = r.prompt.request_id
            info = self.longs.get(rid)
            if info is None:
                continue
            can_pause = (
                info["paused"] < self.resume_every_steps - 1
                and info["budget"] >= self.step_seconds
            )
            lent = []
            for rank in r.ranks if can_pause else ():
                while waiting:
                    unit = waiting.pop(0)
                    if self._start(unit, [], kv, steps, admit, ranks=(rank,)):
                        unit["lent_by"] = rid
                        started.append(unit)
                        lent.append(rank)
                        break
            if lent:
                info["paused"] += 1
                info["budget"] -= self.step_seconds
                paused.append(rid)
                continue
            info["paused"] = 0
            end = min(len(r.prompt.token_ids), r.end + info["chunk"])
            steps.append(PromptStep(rid, r.ranks, end))
        return paused

    def _steps(self, tokens):
        return max(1, math.ceil(tokens / self.step_tokens))

    @staticmethod
    def _steps_on(steps, gpus):
        return math.ceil(steps / gpus)

    def _unit_steps(self, unit):
        return self._steps_on(self._steps(unit["tokens"]), unit["size"])

    def _fits(self, batch, ranks, kv):
        extra = [0] * self.config.world_size
        for owners, r in self._owners(batch, ranks):
            for rank in owners:
                extra[rank] += self._kv(len(r.prompt.token_ids), owners, rank)
        return all(
            kv[rank] + extra[rank] <= self.config.max_local_tokens for rank in ranks
        )

    @staticmethod
    def _owners(batch, ranks):
        # Rotate owner order so per-request remainders spread over the group.
        offset = 0
        for r in batch:
            rotation = offset % len(ranks)
            yield ranks[rotation:] + ranks[:rotation], r
            offset += len(r.prompt.token_ids)

    @staticmethod
    def _kv(length, ranks, rank):
        return len(local_positions(length, ranks, rank))
