# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""CDSP request-level SP: one whole prompt per buddy-aligned block sized by length."""

import dataclasses
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
    """Windowed shortest-first prefill, timed by profiled latency per SP degree.

    Every window_seconds the pending prompts are sorted by length. Prompts shorter
    than step_tokens are packed into groups of at most step_tokens and each group
    runs on one GPU; every longer prompt gets the profiled SP degree that finishes
    it first, the fewest GPUs on a tie. Planned units start in order, a unit
    shrinking to the free GPUs when that finishes no later. While the head waits
    for GPUs, the idle GPUs run the shortest prompts queued behind it or waiting
    for the window, regrouped the same way for those GPUs, that end before it can
    start; with no head waiting, arrivals are queued once a GPU idles. Prompts
    taking more than long_steps chunks of step_tokens per GPU run one chunk per
    operation and may pause, lending their GPUs to as many queued units as end
    within the remaining pause budget: in total at most preempt_ratio of their own
    duration, and never k operations in a row.
    """

    def __init__(
        self,
        config,
        models,
        window_seconds=2.0,
        step_tokens=256,
        long_steps=4,
        resume_every_steps=2,
        preempt_ratio=0.2,
    ):
        if not window_seconds > 0:
            raise ValueError("window_seconds must be positive")
        if step_tokens <= 0 or long_steps <= 0 or resume_every_steps < 1:
            raise ValueError("Need positive step_tokens and long_steps, and k >= 1")
        if not 0 <= preempt_ratio < 1:
            raise ValueError("preempt_ratio must lie in [0, 1)")
        # Only profiled degrees are planned; a group never spans more GPUs.
        self.models = {
            degree: model
            for degree, model in sorted(models.items())
            if degree <= config.world_size
        }
        if 1 not in self.models:
            raise ValueError("Dynamic SP needs a profiled SP=1 latency model")
        self.config = config
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
    def from_profile(cls, config, profile, **kwargs):
        """Latency models per SP degree from a Tetris profile."""
        from sglang.srt.dynamic_sp.tetris import LatencyModel

        models = {
            int(degree): LatencyModel(**values)
            for degree, values in profile["models"].items()
        }
        return cls(config, models, **kwargs)

    def metadata(self):
        return {
            "models": {
                degree: dataclasses.asdict(model)
                for degree, model in self.models.items()
            },
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
            history = self.ends.get(s.request_id, 0)
            groups.setdefault(s.ranks, []).append((history, s.end - history))
        return max(
            (self._seconds(len(ranks), pieces) for ranks, pieces in groups.items()),
            default=0.0,
        )

    def next_plan(self, state):
        config = self.config
        now = state.now
        busy = set(state.busy_ranks)
        requests = {request.prompt.request_id: request for request in state.requests}
        self.ends = {
            request_id: request.end for request_id, request in requests.items()
        }
        self.longs = {
            request_id: info
            for request_id, info in self.longs.items()
            if request_id in requests and not requests[request_id].complete
        }
        release = tuple(
            request_id
            for request_id, request in requests.items()
            if request.complete
            and not request.in_flight
            and not busy.intersection(request.ranks)
        )
        kv_tokens = [0] * config.world_size
        held = set()
        for request_id, request in requests.items():
            if not request.ranks or request_id in release:
                continue
            length = len(request.prompt.token_ids)
            for rank in request.ranks:
                kv_tokens[rank] += self._kv(length, request.ranks, rank)
            if not request.complete or request.in_flight:
                held.update(request.ranks)
        free = sorted(set(range(config.world_size)) - busy - held)
        ready = [
            request
            for request in requests.values()
            if request.ranks
            and not request.complete
            and not request.in_flight
            and not busy.intersection(request.ranks)
        ]
        pending = {
            request_id: request
            for request_id, request in requests.items()
            if not request.ranks and not request.in_flight and not request.complete
        }
        for request in pending.values():
            if (
                len(request.prompt.token_ids)
                > config.world_size * config.max_local_tokens
            ):
                raise ValueError(
                    f"Prompt {request.prompt.request_id} exceeds aggregate KV capacity"
                )
        fresh = self._plan_window(pending, now)
        free_at = self._free_at(free, ready, state, now)

        steps, admit, started = [], [], []
        head_start = None

        def launch(unit):
            if not self._start(unit, free, kv_tokens, steps, admit):
                return False
            finish = now + self._unit_seconds(unit)
            for rank in unit["ranks"]:
                free_at[rank] = finish
            started.append(unit)
            return True

        while True:
            for index, unit in enumerate(self.queue):
                if "ranks" in unit or launch(unit):
                    continue
                head_start = sorted(free_at)[unit["size"] - 1]
                shrunk = self._reshape(unit, len(free), now, head_start)
                if shrunk and launch(unit):
                    head_start = None
                    continue
                behind = self.queue[index + 1 :]
                self._backfill(behind, fresh, free, now, head_start, launch)
                break
            else:
                if fresh and free:
                    # No head is blocked and a GPU idles: queue arrivals now, sized
                    # by every GPU as at a window boundary. While every GPU works,
                    # arrivals are held, then queued shortest-first together.
                    self.queue += self._units(fresh, config.world_size)
                    fresh = []
                    continue
            break
        # Paused long prompts lend to queued units and held arrivals alike,
        # shortest first; units built from held arrivals are dropped if not lent.
        launched = {r.prompt.request_id for unit in started for r in unit["requests"]}
        fresh = [r for r in fresh if r.prompt.request_id not in launched]
        waiting = sorted(
            [unit for unit in self.queue if "ranks" not in unit]
            + self._units(fresh, config.world_size),
            key=lambda unit: unit["tokens"],
        )
        paused = self._run_longs(ready, waiting, kv_tokens, steps, admit, started)
        self.queue = [unit for unit in self.queue if "ranks" not in unit]

        self.last_decision = {
            "version": state.version,
            "started": [
                {
                    "requests": [r.prompt.request_id for r in unit["requests"]],
                    "tokens": unit["tokens"],
                    "ranks": unit.get("ranks"),
                    "lent_by": unit.get("lent_by"),
                }
                for unit in started
            ],
            "paused": paused,
            "queued": sum(len(unit["requests"]) for unit in self.queue),
            "head_start": head_start,
        }
        if not steps and not release:
            # Arrivals behind a blocked head are waiting for the window boundary.
            if busy or held or not self.queue:
                return None
            raise RuntimeError(
                "No unit fits the configured KV capacity; scheduler made no progress"
            )
        return PrefillPlan(state.version, tuple(steps), tuple(admit), release)

    def _plan_window(self, pending, now):
        """At a window boundary, queue every unplanned prompt; inside a window,
        return the arrivals waiting for its end."""
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
        # Size by every GPU, the fastest TTFT; launch shrinks a unit to the free
        # GPUs when that finishes no later.
        self.queue += self._units(fresh, self.config.world_size)
        return []

    def _units(self, requests, gpus):
        """Short prompts packed into one-GPU groups; each longer prompt its own unit."""
        length = lambda r: len(r.prompt.token_ids)
        units, group = [], []
        for r in sorted(requests, key=lambda r: (length(r), r.submitted_at)):
            if length(r) >= self.step_tokens:
                size = self._size(length(r), gpus)
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

    def _backfill(self, behind, fresh, free, now, head_start, launch):
        """Fill the GPUs idle until the blocked head starts with the shortest prompts
        queued behind it or waiting for the window, regrouped for those GPUs, that
        end before it starts."""
        requests = [r for u in behind for r in u["requests"]] + fresh
        moved = set()
        while free:
            # Regroup after every launch, since each one leaves fewer idle GPUs.
            unit = next(
                (
                    u
                    for u in self._units(requests, len(free))
                    if self._ends_by(u, now, head_start) and launch(u)
                ),
                None,
            )
            if unit is None:
                break
            ids = {r.prompt.request_id for r in unit["requests"]}
            moved |= ids
            requests = [r for r in requests if r.prompt.request_id not in ids]
        for u in behind:
            u["requests"] = [
                r for r in u["requests"] if r.prompt.request_id not in moved
            ]
            u["tokens"] = sum(len(r.prompt.token_ids) for r in u["requests"])
        self.queue = [u for u in self.queue if u["requests"]]

    def _reshape(self, unit, gpus, now, head_start):
        """Shrink a blocked prompt to the free GPUs when that finishes no later."""
        if not gpus or unit["tokens"] < self.step_tokens:
            return False
        size = self._size(unit["tokens"], gpus)
        waited = head_start + self._unit_seconds(unit)
        if now + self._unit_seconds(unit, size) > waited:
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
            length = len(r.prompt.token_ids)
            finish = now + self._chunked_seconds(length, len(r.ranks), r.end)
            for rank in r.ranks:
                free_at[rank] = max(free_at[rank], finish)
        return free_at

    def _ends_by(self, unit, now, deadline):
        if deadline is None:
            return True
        return now + self._unit_seconds(unit) <= deadline

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
        preemptible = self._preemptible(unit)
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
                    "budget": self.preempt_ratio * self._unit_seconds(unit),
                    "paused": 0,
                }
            steps.append(PromptStep(r.prompt.request_id, owners, end))
            admit.append(r.prompt)
        return True

    def _run_longs(self, ready, waiting, kv, steps, admit, started):
        """Advance long prompts one chunk, or pause them to lend their GPUs."""
        paused = []
        for r in ready:
            rid = r.prompt.request_id
            info = self.longs.get(rid)
            if info is None:
                continue
            if info["paused"] < self.resume_every_steps - 1:
                pause = self._lend(
                    rid, r.ranks, info["budget"], waiting, kv, steps, admit, started
                )
                if pause:
                    info["paused"] += 1
                    info["budget"] -= pause
                    paused.append(rid)
                    continue
            info["paused"] = 0
            end = min(len(r.prompt.token_ids), r.end + info["chunk"])
            steps.append(PromptStep(rid, r.ranks, end))
        return paused

    def _lend(self, lender, ranks, budget, waiting, kv, steps, admit, started):
        """Start the shortest queued units on the lender's GPUs, as many as fit,
        each ending within budget and never chunked; return the pause length."""
        spare = list(ranks)
        pause = 0.0
        for unit in list(waiting):
            if not spare:
                break
            size = unit["size"]
            if unit["tokens"] >= self.step_tokens:
                size = self._size(unit["tokens"], len(spare))
            if size > len(spare):
                continue
            seconds = self._unit_seconds(unit, size)
            if seconds > budget or self._preemptible(unit, size):
                continue
            original, unit["size"] = unit["size"], size
            if not self._start(unit, [], kv, steps, admit, ranks=tuple(spare[:size])):
                unit["size"] = original
                continue
            unit["lent_by"] = lender
            started.append(unit)
            waiting.remove(unit)
            spare = spare[size:]
            pause = max(pause, seconds)
        return pause

    def _seconds(self, size, pieces):
        """Profiled latency of one operation; pieces are (history, new tokens)."""
        model = self.models[size]
        return model.a + sum(
            (model.b + model.c * history) * tokens + model.d * tokens * tokens
            for history, tokens in pieces
        )

    def _chunked_seconds(self, length, size, start):
        """Prefill of one prompt from start, one chunk per operation."""
        chunk = self.step_tokens * size
        return sum(
            self._seconds(size, [(begin, min(chunk, length - begin))])
            for begin in range(start, length, chunk)
        )

    def _preemptible(self, unit, size=None):
        size = size or unit["size"]
        chunks = math.ceil(unit["tokens"] / (self.step_tokens * size))
        return len(unit["requests"]) == 1 and chunks > self.long_steps

    def _unit_seconds(self, unit, size=None):
        size = size or unit["size"]
        if len(unit["requests"]) == 1:
            return self._prompt_seconds(unit["tokens"], size)
        lengths = [len(r.prompt.token_ids) for r in unit["requests"]]
        return self._seconds(size, [(0, length) for length in lengths])

    def _prompt_seconds(self, length, size):
        if math.ceil(length / (self.step_tokens * size)) > self.long_steps:
            return self._chunked_seconds(length, size, 0)
        return self._seconds(size, [(0, length)])

    def _size(self, length, gpus):
        """Profiled degree within gpus that finishes the prompt first, fewest GPUs
        on a tie."""
        degrees = [d for d in self.models if d <= max(1, gpus)]
        return min(degrees, key=lambda d: (self._prompt_seconds(length, d), d))

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
