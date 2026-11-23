# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Roofline latency prediction and weighted remaining-time allocation."""

import itertools
import json
from pathlib import Path
import math
import time
from functools import lru_cache

from .plan import PrefillPlan, PromptStep, local_positions


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


class RooflineLatency:
    """Dense layer rooflines plus the production ring's hop dependencies.

    RingWorkspace starts one exchange alongside each attention hop, then waits
    for both before the next hop. The last hop has no outgoing exchange.
    Counts use a balanced critical-rank approximation. Effective attention
    rates absorb tiling, softmax, indexed KV access, and compulsory-byte reuse.
    """

    def __init__(self, profile, degree, ranks=None):
        self.degree = degree
        self.ranks = tuple(range(degree)) if ranks is None else tuple(ranks)
        self.geometry = profile["geometry"]
        self.hardware = profile["hardware"]
        self.attention = profile["attention"][str(degree)]
        self.network = profile["network"][str(degree)]
        self.links = profile.get("links", {})
        g = self.geometry
        for key in ("hidden_size", "intermediate_size", "num_layers", "query_heads",
                    "kv_heads", "head_dim", "vocab_size", "element_bytes"):
            if type(g[key]) is not int or g[key] <= 0:
                raise ValueError(f"Invalid geometry {key}")
        measured_ranks = self.ranks if self.links else range(len(self.hardware))
        self.compute = min(
            positive(self.hardware[r]["compute_flops_s"], "compute rate")
            for r in measured_ranks
        )
        self.memory = min(
            positive(self.hardware[r]["memory_bytes_s"], "memory rate")
            for r in measured_ranks
        )
        for key in ("compute_efficiency", "memory_efficiency"):
            if not 0 < positive(self.attention[key], key) <= 1:
                raise ValueError(f"{key} must be <= 1")
        nonnegative(self.attention["launch_seconds"], "attention launch")
        nonnegative(self.network["startup_seconds"], "network startup")
        positive(self.network["bytes_per_second"], "network bandwidth")
        self.dense_launch = nonnegative(profile["dense_launch_seconds"], "dense launch")
        self.step_launch = nonnegative(profile["step_launch_seconds"], "step launch")

    def exchange_seconds(self, byte_count):
        if self.degree == 1:
            return 0.0
        # Optional directed link measurements replace the prefix-group estimate.
        # A partially supplied matrix is rejected rather than silently mixed.
        if self.links:
            links = [self.links[f"{r}:{self.ranks[(i + 1) % self.degree]}"]
                     for i, r in enumerate(self.ranks)]
        else:
            links = [self.network]
        return max(nonnegative(x["startup_seconds"], "link startup")
                   + byte_count / positive(x["bytes_per_second"], "link bandwidth")
                   for x in links)

    def attention_seconds(self, history, chunk):
        g, s = self.geometry, self.degree
        # Useful causal pairs per busiest rank, with cyclic query ownership.
        pairs = []
        for residue in range(s):
            begin = history + ((residue - history) % s)
            n = max(0, (history + chunk - 1 - begin) // s + 1)
            pairs.append(n * (begin + 1) + s * n * (n - 1) // 2)
        flops = 4 * g["query_heads"] * g["head_dim"] * max(pairs)
        qlocal = math.ceil(chunk / s)
        shard = math.ceil((history + chunk) / s)
        kv_bytes = 2 * shard * g["kv_heads"] * g["head_dim"] * g["element_bytes"]
        # Each hop reads one KV shard and updates the query/output accumulator.
        query_bytes = 2 * qlocal * g["query_heads"] * g["head_dim"] * g["element_bytes"]
        compute = flops / s / (self.compute * self.attention["compute_efficiency"])
        memory = (kv_bytes + query_bytes) / (self.memory * self.attention["memory_efficiency"])
        hop = self.attention["launch_seconds"] + max(compute, memory)
        exchange = self.exchange_seconds(kv_bytes)
        # Pack reads and writes local KV; treat preparation conservatively as
        # exposed. The calibrated attention rates describe prepare+attention.
        pack = 2 * kv_bytes / self.memory if s > 1 else 0.0
        return pack + (s - 1) * max(hop, exchange) + hop

    def dense_seconds(self, chunk):
        g = self.geometry
        n = math.ceil(chunk / self.degree)
        h, f, e = g["hidden_size"], g["intermediate_size"], g["element_bytes"]
        qh, kv = g["query_heads"] * g["head_dim"], g["kv_heads"] * g["head_dim"]
        # QKV, attention output, gate/up, and down projections. Weights are
        # replicated, so each rank pays the full matrix read for each operation.
        shapes = ((h, qh + 2 * kv), (qh, h), (h, 2 * f), (f, h))
        seconds = 0.0
        for inner, out in shapes:
            flops = 2 * n * inner * out
            byte_count = e * (inner * out + n * (inner + out))
            seconds += self.dense_launch + max(flops / self.compute, byte_count / self.memory)
        # Two RMSNorms, rotary embedding, SiLU gating, and residual operations.
        seconds += 5 * self.dense_launch + e * n * (12 * h + 4 * f) / self.memory
        return seconds

    @lru_cache(maxsize=8192)
    def seconds(self, history, chunk, complete=False):
        if history < 0 or chunk <= 0:
            raise ValueError("Require nonnegative history and positive chunk")
        g = self.geometry
        result = self.step_launch + g["num_layers"] * (
            self.dense_seconds(chunk) + self.attention_seconds(history, chunk)
        )
        if complete:
            # Only the last-token output head is needed for prefill completion.
            flops = 2 * g["hidden_size"] * g["vocab_size"]
            byte_count = g["element_bytes"] * (g["hidden_size"] * g["vocab_size"]
                                               + g["hidden_size"] + g["vocab_size"])
            result += self.dense_launch + max(flops / self.compute, byte_count / self.memory)
        return result


class SPSRPTScheduler:
    def __init__(
        self, config, profile, quantum_seconds=0.2,
        migration_seconds_per_token=0.0, migration_startup_seconds=0.0,
        prediction_relative_tolerance=0.01, prediction_max_panels=128,
    ):
        if profile.get("schema") not in ("sp-roofline-v1", "spsrptroof-v1"):
            raise ValueError("spsrpt requires a roofline profile, not a Tetris profile")
        degrees = [int(d) for d in profile["attention"] if int(d) <= config.world_size]
        if len(profile["hardware"]) != config.world_size:
            raise ValueError("Roofline hardware rank count does not match world size")
        if profile.get("links"):
            for source in range(config.world_size):
                for target in range(config.world_size):
                    if source != target:
                        link = profile["links"][f"{source}:{target}"]
                        positive(link["bytes_per_second"], "link bandwidth")
                        nonnegative(link["startup_seconds"], "link startup")
        models = {d: RooflineLatency(profile, d) for d in degrees}
        if config.world_size > 8:
            raise ValueError("SPSRPT supports at most eight workers")
        self.quantum_seconds = positive(quantum_seconds, "chunk duration target")
        self.prediction_relative_tolerance = positive(
            prediction_relative_tolerance, "prediction relative tolerance")
        if self.prediction_relative_tolerance >= 1:
            raise ValueError("Prediction relative tolerance must be below one")
        if type(prediction_max_panels) is not int or prediction_max_panels < 2:
            raise ValueError("Prediction panel limit must be an integer >= 2")
        self.prediction_max_panels = prediction_max_panels
        self.prediction_calls = self.prediction_budget_hits = 0
        self.prediction_max_relative_change = 0.0

        self.migration_seconds_per_token = nonnegative(migration_seconds_per_token, "migration cost")
        self.migration_startup_seconds = nonnegative(migration_startup_seconds, "migration startup")
        self.config, self.profile, self.models = config, profile, models
        if 1 not in models or any(degree < 1 for degree in models):
            raise ValueError("SPSRPT requires positive SP degrees and an SP1 reference")
        self.placements = [ranks for degree in sorted(models)
                           for ranks in itertools.combinations(range(config.world_size), degree)]
        self.max_profiled_chunk = {
            degree: int(profile.get("fit_diagnostics", {}).get(str(degree), {})
                        .get("length_range", [1, config.max_batch_tokens])[1])
            for degree in models
        }
        if any(limit <= 0 for limit in self.max_profiled_chunk.values()):
            raise ValueError("Profile chunk limits must be positive")
        self.last_decision = {}
        self._estimates = {}
        self.decision_seconds = 0.0
        self.decisions = 0
        self._placement_models = {}
        self._references = {}
        from .remaining_time import RemainingTimeAllocator

        # Build/load before warmup and measured arrivals, never in next_plan.
        self._allocator = RemainingTimeAllocator()
        self._candidate_cache = {}
        self._last_state_key = None

    @classmethod
    def from_file(cls, config, path, quantum_seconds=0.2, **kwargs):
        return cls(
            config, json.loads(Path(path).read_text()), quantum_seconds, **kwargs
        )

    @staticmethod
    def mask(ranks):
        return sum(1 << rank for rank in set(ranks))

    def migration_seconds(self, history, old, new):
        if not history or not old or old == new:
            return 0.0
        if not self.migration_seconds_per_token:
            return math.inf
        period = math.lcm(len(old), len(new))
        cycles, tail = divmod(history, period)
        moved = sum(
            (cycles + (position < tail))
            for position in range(period)
            if old[position % len(old)] != new[position % len(new)]
        )
        return (
            self.migration_startup_seconds + moved * self.migration_seconds_per_token
            if moved
            else 0.0
        )

    def estimate_operation_seconds(self, plan):
        return sum(self._estimates.get(step.request_id, 0.0) for step in plan.steps)

    def latency_model(self, ranks):
        ranks = tuple(ranks)
        if not self.profile.get("links"):
            return self.models[len(ranks)]
        if ranks not in self._placement_models:
            self._placement_models[ranks] = RooflineLatency(self.profile, len(ranks), ranks)
        return self._placement_models[ranks]

    @staticmethod
    def chunk_tokens(model, history, budget, limit):
        # The stage-wise roofline is monotone but piecewise, not quadratic.
        low, high = 1, limit
        while low < high:
            middle = (low + high + 1) // 2
            if model.seconds(history, middle) <= budget:
                low = middle
            else:
                high = middle - 1
        return low

    def reference_seconds(self, length):
        if length not in self._references:
            history, seconds = 0, 0.0
            while history < length:
                end = min(length, history + self.max_profiled_chunk[1])
                seconds += self.models[1].seconds(history, end - history, end == length)
                history = end
            self._references[length] = seconds
        return self._references[length]

    @staticmethod
    def _efficient_chunk(model, degree, history, length, limit):
        """Maximize tokens/second over executable chunk sizes.

        On each residue class modulo SP degree, the roofline is convex in
        chunk length: affine traffic and quadratic causal work enter positive
        sums and maxima. Consequently q/T(q) is unimodal on each such grid.
        Search small unaligned chunks by residue, and large aligned chunks on
        their 64*degree grid. A completing chunk is checked separately.
        """
        alignment = 64 * degree
        best, best_rate = 1, -math.inf

        def rate(q):
            return q / model.seconds(history, q, history + q == length)

        def consider(q):
            nonlocal best, best_rate
            value = rate(q)
            if value > best_rate or (value == best_rate and q < best):
                best, best_rate = q, value

        def grid(first, stride, last):
            if first > last:
                return
            low, high = 0, (last - first) // stride
            while low < high:
                mid = (low + high) // 2
                q = first + mid * stride
                if rate(q) < rate(q + stride):
                    low = mid + 1
                else:
                    high = mid
            consider(first + low * stride)

        # Keep the completion head outside the convex nonfinal grids.
        nonfinal = min(limit, length - history - 1)
        for residue in range(1, degree + 1):
            grid(residue, degree, min(nonfinal, alignment - 1))
        grid(alignment, alignment, nonfinal)
        if limit == length - history:
            consider(limit)
        return best

    @lru_cache(maxsize=8192)
    def _next_chunk(self, model, degree, history, length):
        limit = min(
            length - history,
            self.max_profiled_chunk[degree],
            max(1, self.config.max_batch_tokens * degree // self.config.world_size),
        )
        if model.seconds(history, 1, history + 1 == length) > self.quantum_seconds:
            return self._efficient_chunk(model, degree, history, length, limit)
        chunk = self.chunk_tokens(model, history, self.quantum_seconds, limit)
        if (chunk == length - history and chunk > 1
                and model.seconds(history, chunk, True) > self.quantum_seconds):
            chunk -= 1
        alignment = 64 * degree
        if chunk < length - history and chunk >= alignment:
            chunk = chunk // alignment * alignment
        return chunk

    @lru_cache(maxsize=4096)
    def _remaining_seconds(self, model, degree, history, length):
        """Price the next step exactly, refine interior service/token to tolerance.

        Nested-resolution midpoint estimates stop on relative convergence or
        an explicit numerical work limit. Their difference is a convergence
        indicator, not a rigorous bound on discrete scheduling or model error.
        """
        self.prediction_calls += 1
        chunk = self._next_chunk(model, degree, history, length)
        seconds = model.seconds(history, chunk, history + chunk == length)
        history += chunk
        if history >= length:
            return seconds
        cap = min(self.max_profiled_chunk[degree],
                  max(1, self.config.max_batch_tokens * degree // self.config.world_size))
        terminal_size = self._next_chunk(model, degree, length - 1, length - 1 + cap)
        tail = min(length - history, terminal_size)
        stop = length - tail
        seconds += model.seconds(stop, tail, True)
        span = stop - history
        if span == 0:
            return seconds

        def integral(panels):
            total = 0.0
            for panel in range(panels):
                left = history + span * panel // panels
                right = history + span * (panel + 1) // panels
                midpoint = (left + right) // 2
                q = self._next_chunk(model, degree, midpoint, midpoint + cap)
                total += (right - left) * model.seconds(midpoint, q) / q
            return total

        panels = 1
        previous = integral(panels)
        limit = min(span, self.prediction_max_panels)
        if limit == 1:
            return seconds + previous
        while panels < limit:
            panels = min(2 * panels, limit)
            current = integral(panels)
            relative_change = abs(current - previous) / (seconds + current)
            if relative_change <= self.prediction_relative_tolerance or panels == span:
                return seconds + current
            previous = current
        self.prediction_budget_hits += 1
        self.prediction_max_relative_change = max(
            self.prediction_max_relative_change, relative_change)
        return seconds + previous

    def _candidates(self, request):
        """Cache per-prompt predictions independently of other requests' progress."""
        rid = request.prompt.request_id
        length = len(request.prompt.token_ids)
        history = (request.reserved_end if request.in_flight
                   and request.reserved_end is not None else request.end)
        key = (length, history, tuple(request.ranks))
        cached = self._candidate_cache.get(rid)
        if cached is not None and cached[0] == key:
            return cached[1]
        if history >= length:
            return []
        reference = self.reference_seconds(length)
        old_counts = {r: len(local_positions(length, request.ranks, r))
                      for r in request.ranks}
        predictions = []
        for ranks in self.placements:
            if self.config.max_batch_tokens * len(ranks) // self.config.world_size < 1:
                continue
            migration = self.migration_seconds(history, request.ranks, ranks)
            if not math.isfinite(migration):
                continue
            model = self.latency_model(ranks)
            duration = migration + self._remaining_seconds(model, len(ranks), history, length)
            if not math.isfinite(duration) or duration <= 0:
                continue
            # Full-prompt reservations, retaining all other resident requests.
            delta = tuple((r, len(local_positions(length, ranks, r)) - old_counts.get(r, 0))
                          for r in ranks)
            predictions.append({
                "ranks": ranks, "mask": self.mask((*request.ranks, *ranks)),
                "duration": duration, "migration_seconds": migration,
                "reference_seconds": reference, "capacity_delta": delta,
                "history": history,
                "chunk": self._next_chunk(model, len(ranks), history, length),
            })
        self._candidate_cache[rid] = (key, predictions)
        return predictions

    def next_plan(self, state):
        started = time.perf_counter()
        busy = self.mask(state.busy_ranks)
        state_key = (state.version, busy, tuple(
            (r.prompt.request_id, len(r.prompt.token_ids), r.end, r.reserved_end,
             tuple(r.ranks), r.in_flight) for r in state.requests
        ))
        # Scheduling is event driven. Clock-only polling cannot change a plan.
        if state_key == self._last_state_key:
            return None
        self._last_state_key = state_key
        decisions = []

        def finish(steps=(), admit=(), release=(), candidate_seconds=0.0, solver_seconds=0.0):
            elapsed = time.perf_counter() - started
            self.decision_seconds += elapsed
            self.decisions += 1
            self.last_decision = {
                "version": state.version, "seconds": elapsed,
                "policy": "weighted-remaining-time-v1",
                "candidate_seconds": candidate_seconds, "solver_seconds": solver_seconds,
                "objective": "sum 1 / (reference_seconds * (wait + migration + remaining_seconds))",
                "degree_choices": decisions,
                "steps": [{"request_id": x.request_id, "ranks": x.ranks, "end": x.end}
                          for x in steps],
            }
            if not steps and not release:
                return None
            return PrefillPlan(state.version, tuple(steps), tuple(admit), tuple(release))

        release = tuple(r.prompt.request_id for r in state.requests
                        if r.complete and r.ranks and not r.in_flight
                        and not self.mask(r.ranks) & busy)
        if release:
            for rid in release:
                self._estimates.pop(rid, None)
                self._candidate_cache.pop(rid, None)
            return finish(release=release)
        live = {r.prompt.request_id for r in state.requests}
        for rid in self._candidate_cache.keys() - live:
            del self._candidate_cache[rid]
        kv = [0] * self.config.world_size
        residents = [r for r in state.requests if r.ranks]
        for r in residents:
            for rank in r.ranks:
                kv[rank] += len(local_positions(len(r.prompt.token_ids), r.ranks, rank))
        slots = max(0, self.config.max_running_requests - len(residents))
        requests = sorted(
            (r for r in state.requests if not r.complete and not (
                r.in_flight and r.reserved_end == len(r.prompt.token_ids))),
            key=lambda r: (r.submitted_at, r.prompt.request_id),
        )
        groups, admission = [], []
        for request in requests:
            # Same lock mask has the same conflicts and admission cost. Only
            # its highest index can participate in an optimal allocation.
            best = {}
            if request.ranks or slots:
                for prediction in self._candidates(request):
                    if any(kv[r] + delta > self.config.max_local_tokens
                           for r, delta in prediction["capacity_delta"]):
                        continue
                    mask = prediction["mask"]
                    wait = max((max(0.0, state.worker_available_at[r] - state.now)
                                if state.worker_available_at else self.quantum_seconds
                                for r in range(self.config.world_size)
                                if mask & busy & (1 << r)), default=0.0)
                    index = 1.0 / (prediction["reference_seconds"]
                                   * (wait + prediction["duration"]))
                    candidate = dict(prediction, index=index, wait_seconds=wait)
                    previous = best.get(mask)
                    if previous is None or index > previous["index"]:
                        best[mask] = candidate
            groups.append(list(best.values()))
            admission.append(int(not request.ranks))
        candidate_seconds = time.perf_counter() - started
        solve_started = time.perf_counter()
        selected = self._allocator.allocate(self.config.world_size, slots, groups, admission)
        solver_seconds = time.perf_counter() - solve_started
        steps, admit = [], []
        tokens = self.config.max_batch_tokens
        for request, candidate in zip(requests, selected):
            if candidate is None:
                continue
            ranks = candidate["ranks"]
            blocked = bool(candidate["mask"] & busy) or request.in_flight
            decisions.append({
                "request_id": request.prompt.request_id, "ranks": ranks,
                "processed_tokens": candidate["history"],
                "remaining_tokens": len(request.prompt.token_ids) - candidate["history"],
                "reason": "maximum_total_weighted_remaining_time_index",
                "index": candidate["index"],
                "reference_seconds": candidate["reference_seconds"],
                "predicted_remaining_seconds": candidate["duration"],
                "migration_seconds": candidate["migration_seconds"],
                "wait_seconds": candidate["wait_seconds"],
                "dispatch": not blocked,
                "chunk_target_infeasible": self.latency_model(ranks).seconds(
                    candidate["history"], 1,
                    candidate["history"] + 1 == len(request.prompt.token_ids)
                ) > self.quantum_seconds,
                "blocked_ranks": tuple(r for r in range(self.config.world_size)
                                       if candidate["mask"] & busy & (1 << r)),
            })
            # A selected blocked candidate reserves its entire operation mask.
            # The exact solver has already excluded conflicting dispatches.
            if blocked:
                continue
            chunk = candidate["chunk"]
            if chunk > tokens:
                raise RuntimeError("Disjoint proportional chunk limits exceeded the batch budget")
            tokens -= chunk
            end = request.end + chunk
            if not request.ranks:
                admit.append(request.prompt)
            steps.append(PromptStep(request.prompt.request_id, ranks, end))
            self._estimates[request.prompt.request_id] = (
                candidate["migration_seconds"]
                + self.latency_model(ranks).seconds(request.end, chunk, end == len(request.prompt.token_ids))
            )
        if not steps and not busy and requests:
            raise RuntimeError("No remaining-time allocation fits KV capacity")
        return finish(steps, admit, candidate_seconds=candidate_seconds, solver_seconds=solver_seconds)

    def metadata(self):
        return {
            "quantum_seconds": self.quantum_seconds,
            "degrees": sorted(self.models),
            "max_profiled_chunk": self.max_profiled_chunk,
            "chunk_alignment_per_worker": 64,
            "infeasible_target_policy": "maximum_predicted_tokens_per_second",
            "remaining_time_estimator": "relative-midpoint-terminal-v2",
            "prediction_relative_tolerance": self.prediction_relative_tolerance,
            "prediction_max_panels": self.prediction_max_panels,
            "prediction_calls": self.prediction_calls,
            "prediction_budget_hits": self.prediction_budget_hits,
            "prediction_max_unconverged_relative_change": self.prediction_max_relative_change,
            "placement": "all subsets; old and new owners locked during migration",
            "migration_seconds_per_token": self.migration_seconds_per_token,
            "migration_startup_seconds": self.migration_startup_seconds,
            "decision_seconds": self.decision_seconds,
            "decisions": self.decisions,
            "last_decision": self.last_decision,
            "policy": "weighted-remaining-time-v1",
            "priority": "inverse original reference time times inverse predicted remaining time",
            "allocation": "exact compiled multiple-choice GPU-mask packing; wait is allowed",
            "objective": "sum 1 / (reference_seconds * (wait + migration + remaining_seconds))",
            "optimality_scope": "per-decision index objective, not global online mean slowdown",
            "preemption": "completed forward passes; no timed leases",
            "reference": "fixed original SP1 roofline service time",
            "latency_model": "stage-wise compute/HBM roofline with ring hop dependencies",
            "roofline_profile_source": self.profile.get("source"),
            "placement_calibration": self.profile["placement_calibration"],
        }
