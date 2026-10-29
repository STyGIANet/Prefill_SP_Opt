# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""CDSP request-level SP: one whole prompt per buddy-aligned block sized by length."""

import bisect
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

    A unit is what launches together: a group of prompts shorter than step_tokens
    packed up to step_tokens on one GPU, or one longer prompt on the profiled SP
    degree that finishes it first, the fewest GPUs on a tie.

    Every next_plan call:
      1. Observe: release finished prompts, then find idle GPUs, resident KV, and
         long prompts waiting for their next chunk.
      2. Window (_plan_window): every window_seconds, append all arrivals to the
         plan as units, shortest first; inside a window, hold arrivals unless a
         GPU idles. The default 6 ms is the waiting window LAPS (MLSys 2026)
         found best for short prefills; 0 appends on every call. A planned unit
         keeps its place in the plan.
      3. Dispatch (_book): book the plan in order on per-GPU timelines that start
         at each GPU's predicted free time. Each unit takes the profiled degree
         and the earliest slot, now or later, where that many GPUs stay idle for
         its whole duration, the fewest GPUs on a tie. Earlier units are booked
         first, so a later unit only fills a bubble the earlier ones leave and
         never delays them. Units booked to start now on idle GPUs launch.
      4. Long prompts (_advance_long_prompts): a prompt taking more than
         long_steps chunks of step_tokens per GPU runs one chunk per operation.
         Between chunks it may pause and lend its GPUs (_lend_gpus) to waiting
         units, queued or held, that end within its budget: preempt_ratio of its
         own duration in total, and at most resume_every_steps - 1 pauses in a
         row. A pause starts only when the waiting units hold at least
         step_tokens per lender GPU. A pause may last the whole remaining budget: each lender GPU that
         frees up is lent again to the shortest waiting units that end in time,
         and the lender resumes early once nothing is left to lend.
    """

    def __init__(
        self,
        config,
        models,
        window_seconds=0.006,
        step_tokens=450,
        long_steps=4,
        resume_every_steps=2,
        preempt_ratio=0.2,
    ):
        if not window_seconds >= 0:
            raise ValueError("window_seconds must be nonnegative")
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
        # Units not launched yet, in booking order; a unit keeps its place.
        self.plan = []
        self.window_end = None
        # Running long prompts: request ID -> {"chunk", "budget", "paused",
        # "pause_start", "pause_until"}; the pause times are set while it lends.
        self.long_prompts = {}
        # Committed prefill end per request, for estimate_operation_seconds.
        self.committed_ends = {}
        # _lend_gpus calls over the whole run, whether or not they lent anything.
        self.lend_calls = 0
        self.last_decision = {}

    @classmethod
    def from_profile(cls, config, profile, step_tokens=None, **kwargs):
        """Latency models per SP degree from a Tetris profile; step_tokens defaults
        to its min_chunk_tokens, the smallest chunk Tetris plans."""
        from sglang.srt.dynamic_sp.tetris import LatencyModel

        models = {
            int(degree): LatencyModel(**values)
            for degree, values in profile["models"].items()
        }
        if step_tokens is None:
            step_tokens = profile["min_chunk_tokens"]
        return cls(config, models, step_tokens=step_tokens, **kwargs)

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
            "lend_calls": self.lend_calls,
            "last_decision": self.last_decision,
        }

    def estimate_operation_seconds(self, plan):
        pieces_by_ranks = {}
        for step in plan.steps:
            history = self.committed_ends.get(step.request_id, 0)
            pieces_by_ranks.setdefault(step.ranks, []).append(
                (history, step.end - history)
            )
        return max(
            (
                self._operation_seconds(len(ranks), pieces)
                for ranks, pieces in pieces_by_ranks.items()
            ),
            default=0.0,
        )

    def next_plan(self, state):
        config = self.config
        now = state.now
        busy = set(state.busy_ranks)
        requests = {request.prompt.request_id: request for request in state.requests}
        self.committed_ends = {
            request_id: request.end for request_id, request in requests.items()
        }
        self.long_prompts = {
            request_id: long_prompt
            for request_id, long_prompt in self.long_prompts.items()
            if request_id in requests and not requests[request_id].complete
        }

        # 1. Observe.
        release = tuple(
            request_id
            for request_id, request in requests.items()
            if request.complete
            and not request.in_flight
            and not busy.intersection(request.ranks)
        )
        kv_tokens = [0] * config.world_size
        # Rank -> unfinished prompts placed on it; a lender shares its ranks with
        # its borrowers.
        holders = {}
        for request_id, request in requests.items():
            if not request.ranks or request_id in release:
                continue
            length = len(request.prompt.token_ids)
            for rank in request.ranks:
                kv_tokens[rank] += self._local_kv_tokens(length, request.ranks, rank)
            if not request.complete or request.in_flight:
                for rank in request.ranks:
                    holders.setdefault(rank, set()).add(request_id)
        held = set(holders)
        free = sorted(set(range(config.world_size)) - busy - held)
        # Placed prompts between operations: long prompts awaiting their next chunk.
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

        # 2. Window.
        fresh = self._plan_window(pending, now)
        free_at = self._predict_free_at(free, ready, state, now)
        if fresh and free:
            # A GPU idles: plan arrivals now rather than at the window end.
            self.plan += self._make_units(fresh, config.world_size)
            fresh = []

        # 3. Dispatch.
        steps, admit, started = [], [], []
        bookings = self._book(free_at, now)
        for index, (unit, (start, finish, ranks)) in enumerate(
            zip(self.plan, bookings)
        ):
            if start > now or not set(ranks) <= set(free):
                continue
            original_size, unit["size"] = unit["size"], len(ranks)
            if not self._start_unit(unit, free, kv_tokens, steps, admit, ranks=ranks):
                unit["size"] = original_size
                continue
            # Launched while an earlier unit still waits: it fills a bubble.
            unit["bubblefill"] = any(
                "ranks" not in earlier for earlier in self.plan[:index]
            )
            for rank in ranks:
                free_at[rank] = finish
            started.append(unit)
        self.plan = [unit for unit in self.plan if "ranks" not in unit]
        next_start = min(
            (start for start, _, _ in bookings if start > now), default=None
        )

        # 4. Long prompts. Paused ones lend to queued units and held arrivals
        # alike, shortest first; units built from held arrivals are dropped if
        # not lent.
        launched = {
            request.prompt.request_id
            for unit in started
            for request in unit["requests"]
        }
        fresh = [
            request for request in fresh if request.prompt.request_id not in launched
        ]
        waiting = sorted(
            self.plan + self._make_units(fresh, config.world_size),
            key=lambda unit: unit["tokens"],
        )
        paused = self._advance_long_prompts(
            requests,
            busy,
            holders,
            now,
            free_at,
            waiting,
            kv_tokens,
            steps,
            admit,
            started,
        )
        self.plan = [unit for unit in self.plan if "ranks" not in unit]

        self.last_decision = {
            "version": state.version,
            "started": [
                {
                    "requests": [
                        request.prompt.request_id for request in unit["requests"]
                    ],
                    "tokens": unit["tokens"],
                    "ranks": unit.get("ranks"),
                    "lent_by": unit.get("lent_by"),
                    "bubblefill": unit.get("bubblefill", False),
                }
                for unit in started
            ],
            "paused": paused,
            "queued": sum(len(unit["requests"]) for unit in self.plan),
            "next_start": next_start,
        }
        if not steps and not release:
            # Planned units wait for GPUs; arrivals wait for the window boundary.
            if busy or held or not self.plan:
                return None
            raise RuntimeError(
                "No unit fits the configured KV capacity; scheduler made no progress"
            )
        return PrefillPlan(state.version, tuple(steps), tuple(admit), release)

    # ===== Window and units =====

    def _plan_window(self, pending, now):
        """At a window boundary, append every unplanned prompt to the plan; inside
        a window, return the arrivals waiting for its end."""
        queued = {
            request.prompt.request_id
            for unit in self.plan
            for request in unit["requests"]
        }
        self.plan = [
            unit
            for unit in self.plan
            if all(request.prompt.request_id in pending for request in unit["requests"])
        ]
        fresh = [
            request
            for request_id, request in pending.items()
            if request_id not in queued
        ]
        if self.window_end is None:
            self.window_end = now + self.window_seconds
        if now < self.window_end:
            return fresh
        self.window_end = now + self.window_seconds
        self.plan += self._make_units(fresh, self.config.world_size)
        return []

    def _make_units(self, requests, gpus):
        """Short prompts packed into one-GPU groups; each longer prompt its own unit,
        sized within gpus. Returned shortest first."""
        units, group, group_tokens = [], [], 0
        for request in sorted(
            requests, key=lambda r: (len(r.prompt.token_ids), r.submitted_at)
        ):
            length = len(request.prompt.token_ids)
            if length >= self.step_tokens:
                size = self._best_size(length, gpus)
                units.append({"requests": [request], "tokens": length, "size": size})
            elif group and group_tokens + length > self.step_tokens:
                units.append(self._pack_group(group))
                group, group_tokens = [request], length
            else:
                group.append(request)
                group_tokens += length
        if group:
            units.append(self._pack_group(group))
        units.sort(key=lambda unit: unit["tokens"])
        return units

    def _pack_group(self, requests):
        tokens = sum(len(request.prompt.token_ids) for request in requests)
        return {"requests": requests, "tokens": tokens, "size": 1}

    # ===== Dispatch =====

    def _predict_free_at(self, free, ready, state, now):
        """Predicted time each rank becomes free, for locating bubbles."""
        free_at = list(state.worker_available_at or [now] * self.config.world_size)
        for rank in free:
            free_at[rank] = now
        for request in ready:
            length = len(request.prompt.token_ids)
            finish = now + self._chunked_seconds(
                length, len(request.ranks), request.end
            )
            for rank in request.ranks:
                free_at[rank] = max(free_at[rank], finish)
        return free_at

    def _book(self, free_at, now):
        """Book the plan in order; return (start, finish, ranks) per unit.

        A GPU's timeline starts at its predicted free time and holds the units
        booked before. Each unit tries every profiled degree it may take and
        keeps the earliest finish, the fewest GPUs on a tie.
        """
        # Rank -> sorted (start, finish) of booked units.
        timelines = {
            rank: [(now, max(now, free_at[rank]))]
            for rank in range(self.config.world_size)
        }
        bookings = []
        for unit in self.plan:
            # Packed groups of short prompts stay on one GPU.
            sizes = [1] if unit["tokens"] < self.step_tokens else list(self.models)
            best = None
            for size in sizes:
                seconds = self._unit_seconds(unit, size)
                start, ranks = self._earliest_slot(timelines, size, seconds, now)
                if best is None or start + seconds < best[1]:
                    best = (start, start + seconds, ranks)
            start, finish, ranks = best
            for rank in ranks:
                bisect.insort(timelines[rank], (start, finish))
            bookings.append(best)
        return bookings

    @staticmethod
    def _earliest_slot(timelines, size, seconds, now):
        """Earliest start at which size GPUs stay idle for seconds; the lowest
        ranks among them."""
        # A GPU set can only become idle when one of its GPUs starts idling: now,
        # or a booking end not followed at once by the next booking.
        starts = {now}
        for slots in timelines.values():
            for (_, finish), (begin, _) in zip(slots, slots[1:] + [(math.inf, 0)]):
                if now < finish < begin:
                    starts.add(finish)
        for start in sorted(starts):
            end = start + seconds
            idle = []
            for rank, slots in timelines.items():
                # Bookings on a GPU never overlap, so the last one starting
                # before end also finishes last.
                before = bisect.bisect_left(slots, (end,))
                if before == 0 or slots[before - 1][1] <= start:
                    idle.append(rank)
            if len(idle) >= size:
                return start, tuple(idle[:size])
        raise AssertionError("Every GPU idles after its last booking")

    def _start_unit(self, unit, free, kv_tokens, steps, admit, ranks=None):
        """Place unit on the lowest free ranks (or the given ranks) if its KV fits,
        and emit its first operation."""
        if ranks is None:
            if len(free) < unit["size"]:
                return False
            ranks = tuple(free[: unit["size"]])
        if not self._kv_fits(unit["requests"], ranks, kv_tokens):
            return False
        for rank in ranks:
            if rank in free:
                free.remove(rank)
        unit["ranks"] = ranks
        preemptible = self._preemptible(unit)
        for owners, request in self._owners(unit["requests"], ranks):
            length = len(request.prompt.token_ids)
            for rank in owners:
                kv_tokens[rank] += self._local_kv_tokens(length, owners, rank)
            end = length
            if preemptible:
                chunk = self.step_tokens * len(ranks)
                end = min(length, chunk)
                self.long_prompts[request.prompt.request_id] = {
                    "chunk": chunk,
                    "budget": self.preempt_ratio * self._unit_seconds(unit),
                    "paused": 0,
                    "pause_start": None,
                    "pause_until": None,
                }
            steps.append(PromptStep(request.prompt.request_id, owners, end))
            admit.append(request.prompt)
        return True

    # ===== Long prompts =====

    def _advance_long_prompts(
        self,
        requests,
        busy,
        holders,
        now,
        free_at,
        waiting,
        kv_tokens,
        steps,
        admit,
        started,
    ):
        """Advance long prompts one chunk, or pause them to lend their GPUs; return
        the paused request IDs.

        A pause opens a window over the lender's remaining budget. Every call
        until it ends, the window is planned again on the lender's GPUs (_lend_gpus),
        so GPUs freed by finished borrowers and newly arrived prompts fill it. The
        lender resumes when the window ends, or earlier once its GPUs idle with
        nothing to lend; its budget is charged the time it paused.
        """
        paused = []
        for request_id, request in requests.items():
            long_prompt = self.long_prompts.get(request_id)
            # Prompts launched in this call are not placed in state yet.
            if long_prompt is None or not request.ranks or request.in_flight:
                continue
            # Lender GPUs not running an operation and not held by a borrower.
            idle = [
                rank
                for rank in request.ranks
                if rank not in busy and holders[rank] == {request_id}
            ]
            pause_until = long_prompt["pause_until"]
            if pause_until is None:
                if len(idle) < len(request.ranks):
                    paused.append(request_id)
                    continue
                # Open a window over the remaining budget, only when the waiting
                # units can give every lender GPU a step of work.
                waiting_tokens = sum(unit["tokens"] for unit in waiting)
                can_pause = (
                    long_prompt["paused"] < self.resume_every_steps - 1
                    and waiting_tokens >= self.step_tokens * len(request.ranks)
                )
                if can_pause and self._lend_gpus(
                    request_id,
                    request.ranks,
                    idle,
                    now,
                    now + long_prompt["budget"],
                    free_at,
                    waiting,
                    kv_tokens,
                    steps,
                    admit,
                    started,
                ):
                    long_prompt["paused"] += 1
                    long_prompt["pause_start"] = now
                    long_prompt["pause_until"] = now + long_prompt["budget"]
                    paused.append(request_id)
                    continue
            else:
                # Plan the open window again.
                lent = now < pause_until and self._lend_gpus(
                    request_id,
                    request.ranks,
                    idle,
                    now,
                    pause_until,
                    free_at,
                    waiting,
                    kv_tokens,
                    steps,
                    admit,
                    started,
                )
                if lent or len(idle) < len(request.ranks):
                    # Borrowers run on its GPUs.
                    paused.append(request_id)
                    continue
                # Closing the window leaves budget pause_until - now, so lending
                # again now would plan the same deadline; resume instead.
                long_prompt["budget"] -= now - long_prompt["pause_start"]
                long_prompt["pause_until"] = None
            long_prompt["paused"] = 0
            end = min(len(request.prompt.token_ids), request.end + long_prompt["chunk"])
            steps.append(PromptStep(request_id, request.ranks, end))
        return paused

    def _lend_gpus(
        self,
        lender,
        ranks,
        idle,
        now,
        deadline,
        free_at,
        waiting,
        kv_tokens,
        steps,
        admit,
        started,
    ):
        """Plan the shortest waiting units back to back on the lender's GPUs until
        deadline, each never chunked; start those whose GPUs are idle now and
        return how many started.

        Each GPU's calendar begins when it frees up: now if idle, else when its
        borrower ends. A unit takes the profiled size and the earliest-free GPUs
        that finish it first, the fewest GPUs on a tie; units planned to start
        later are planned again by the next call.
        """
        self.lend_calls += 1
        calendar = {
            rank: now if rank in idle else max(now, free_at[rank]) for rank in ranks
        }
        launched = 0
        for unit in list(waiting):
            # Packed groups of short prompts stay on one GPU.
            sizes = (
                [unit["size"]]
                if unit["tokens"] < self.step_tokens
                else [size for size in self.models if size <= len(ranks)]
            )
            best = None
            for size in sizes:
                if self._preemptible(unit, size):
                    continue
                # On a tie, idle GPUs first: a borrower past its predicted end
                # still holds its GPU.
                gpus = sorted(
                    ranks, key=lambda rank: (calendar[rank], rank not in idle, rank)
                )[:size]
                start = max(calendar[rank] for rank in gpus)
                finish = start + self._unit_seconds(unit, size)
                if finish <= deadline and (best is None or finish < best[0]):
                    best = (finish, start, size, gpus)
            if best is None:
                continue
            finish, start, size, gpus = best
            # Start now only on GPUs idle now and not taken by an earlier unit.
            if start <= now and all(rank in idle for rank in gpus):
                original_size, unit["size"] = unit["size"], size
                # The lender's ranks are held, so they are not in any free list.
                if not self._start_unit(
                    unit, [], kv_tokens, steps, admit, ranks=tuple(sorted(gpus))
                ):
                    unit["size"] = original_size
                    continue
                unit["lent_by"] = lender
                started.append(unit)
                waiting.remove(unit)
                launched += 1
            # Reserve the GPUs whether the unit starts now or later.
            for rank in gpus:
                calendar[rank] = finish
        return launched

    # ===== Latency model =====

    def _operation_seconds(self, size, pieces):
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
            self._operation_seconds(size, [(begin, min(chunk, length - begin))])
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
        pieces = [(0, len(request.prompt.token_ids)) for request in unit["requests"]]
        return self._operation_seconds(size, pieces)

    def _prompt_seconds(self, length, size):
        if math.ceil(length / (self.step_tokens * size)) > self.long_steps:
            return self._chunked_seconds(length, size, 0)
        return self._operation_seconds(size, [(0, length)])

    def _best_size(self, length, gpus):
        """Profiled degree within gpus that finishes the prompt first, fewest GPUs
        on a tie."""
        degrees = [degree for degree in self.models if degree <= max(1, gpus)]
        return min(
            degrees,
            key=lambda degree: (self._prompt_seconds(length, degree), degree),
        )

    # ===== KV placement =====

    def _kv_fits(self, requests, ranks, kv_tokens):
        extra = [0] * self.config.world_size
        for owners, request in self._owners(requests, ranks):
            length = len(request.prompt.token_ids)
            for rank in owners:
                extra[rank] += self._local_kv_tokens(length, owners, rank)
        return all(
            kv_tokens[rank] + extra[rank] <= self.config.max_local_tokens
            for rank in ranks
        )

    @staticmethod
    def _owners(requests, ranks):
        # Rotate owner order so per-request remainders spread over the group.
        offset = 0
        for request in requests:
            rotation = offset % len(ranks)
            yield ranks[rotation:] + ranks[:rotation], request
            offset += len(request.prompt.token_ids)

    @staticmethod
    def _local_kv_tokens(length, ranks, rank):
        return len(local_positions(length, ranks, rank))
