# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""LoongServe global manager (Sec. 5) over elastic sequence-parallel instances.

Every scheduling call runs the paper's four steps:
  Sec. 5.1 dispatching            -> _dispatch
  Sec. 5.2 elastic instance alloc -> _preempt_for_memory, _expand
  Sec. 5.3 batching               -> BatchPlanner (Eq. 5, Eq. 6 in csrc/loongserve.cpp)
  Sec. 5.4 elastic scaling plan   -> choose_decode_instance
with the Sec. 5.5 analytical model (Eq. 7) in predict.

Without attached decode state the instances only prefill, and Sec. 5.1's
preemption terms and Sec. 5.4 have nothing to act on. With attach_decode,
every worker is also a TP1 decode instance (elastic mode): prefill may preempt
an instance's decoding batch, which then pauses for that prefill iteration.
Deviations: a preempted batch pauses instead of migrating its KV (Eq. 4 still
prices the migration it would need), and multi-master decoding is not
implemented, so a request must fit one decode instance.
"""

import ctypes
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from sglang.srt.dynamic_sp.plan import PrefillPlan, PromptStep


class BatchPlanner:
    """Compiled contiguous partition DP; compiled before measured arrivals."""

    def __init__(self, models, monotone=True):
        self.monotone = monotone
        self.models = np.zeros((len(models) + 1, 3), dtype=np.float64)
        for degree, values in models.items():
            self.models[int(degree)] = values
        source = Path(__file__).parent / "csrc/loongserve.cpp"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
        cache = Path(
            os.environ.get(
                "SGLANG_SP_BUILD_DIR", str(source.parent.parent / "work/build")
            )
        )
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"loongserve-{digest}.so"
        with (cache / "loongserve.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=cache) as temporary:
                    output = Path(temporary) / "scheduler.so"
                    subprocess.run(
                        [
                            os.environ.get("CXX", "c++"),
                            "-std=c++17",
                            "-O3",
                            "-shared",
                            "-fPIC",
                            str(source),
                            "-o",
                            str(output),
                        ],
                        check=True,
                    )
                    output.replace(target)
        self.library = ctypes.CDLL(str(target))
        self.solve = self.library.loongserve_partition
        self.solve.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        ]
        self.solve.restype = ctypes.c_int

    def __deepcopy__(self, memo):
        return self  # Immutable coefficients and stateless compiled function.

    def partition(self, lengths, capacities):
        """Sec. 5.3: split sorted requests and ordered instances into batches."""
        n, m = len(lengths), len(capacities)
        if not n or not m:
            return []
        if m >= len(self.models):
            raise ValueError("Missing LoongServe degree models")
        splits = np.empty((min(n, m), 4), dtype=np.int32)
        count = self.solve(
            n,
            m,
            np.asarray(lengths, dtype=np.int64),
            np.asarray(capacities, dtype=np.int64),
            self.models,
            int(self.monotone),
            splits,
        )
        if count == -3:
            raise ValueError(
                "LoongServe latency extrapolation is nonpositive; recalibrate the SIB"
            )
        if count < 0:
            raise ValueError("No memory-feasible LoongServe batch partition")
        return [tuple(map(int, row)) for row in splits[:count][::-1]]


class LoongServeScheduler:
    def __init__(self, config, profile, max_prefill_seconds=None, monotone=True):
        self.config, self.profile = config, profile
        if (
            profile.get("schema_version") != 1
            or profile.get("algorithm") != "loongserve"
        ):
            raise ValueError("Expected a LoongServe SIB profile")
        self.nodes = tuple(profile["nodes"])
        self.models = {int(s): tuple(v) for s, v in profile["models"].items()}
        if len(self.nodes) != config.world_size or set(self.models) != set(
            range(1, config.world_size + 1)
        ):
            raise ValueError("LoongServe requires measured models for every SP degree")
        if any(
            len(v) != 3 or any(not math.isfinite(x) for x in v)
            for v in self.models.values()
        ):
            raise ValueError("Invalid LoongServe coefficients")
        # Sec. 5.1 compute check: the profiled memory-to-compute-bound tipping point.
        self.max_prefill_seconds = (
            profile["max_prefill_seconds"]
            if max_prefill_seconds is None
            else max_prefill_seconds
        )
        if not math.isfinite(self.max_prefill_seconds) or self.max_prefill_seconds <= 0:
            raise ValueError("LoongServe prefill time limit must be positive")
        self.planner = BatchPlanner(self.models, monotone)
        self.decode = None
        self.last_decision = {}

    @classmethod
    def from_file(cls, config, path, max_prefill_seconds=None, monotone=True):
        return cls(
            config, json.loads(Path(path).read_text()), max_prefill_seconds, monotone
        )

    def attach_decode(self, coordinator, kv_bytes_per_token, bandwidth, batch_limit):
        """Elastic mode: every worker also decodes; see the module docstring.

        bandwidth prices Eq. (4) migrations in bytes/s; batch_limit is the Sec. 5.4
        compute-bound decode batch size beyond which an instance is scaled up.
        """
        self.decode = coordinator
        self.kv_bytes_per_token = kv_bytes_per_token
        self.bandwidth = bandwidth
        self.batch_limit = batch_limit
        coordinator.choose = self.choose_decode_instance

    def predict(self, degree, lengths):
        """Sec. 5.5 Eq. (7): T = alpha + beta * sum(len) + gamma * sum(len^2)."""
        a, b, c = self.models[degree]
        result = a + b * sum(lengths) + c * sum(n * n for n in lengths)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(
                "LoongServe latency model is nonpositive for this batch; recalibrate"
            )
        return result

    def estimate_operation_seconds(self, plan):
        if not plan.steps:
            return 0.0
        # A LoongServe operation is one full-prefill batch on one owner set.
        return self.predict(len(plan.steps[0].ranks), [s.end for s in plan.steps])

    def metadata(self):
        return {
            "profile": self.profile,
            "profile_sha256": hashlib.sha256(
                json.dumps(self.profile, sort_keys=True).encode()
            ).hexdigest(),
            "max_prefill_seconds": self.max_prefill_seconds,
            "dp": "monotone" if self.planner.monotone else "exact",
            "elastic_decode": self.decode is not None,
            "last_decision": self.last_decision,
        }

    # ---- decode-instance state (empty without attach_decode) ----

    def _decoding(self, rank):
        return self.decode is not None and (
            bool(self.decode.running[(rank,)]) or self.decode.reserved[(rank,)] > 0
        )

    def _unused_slots(self, rank):
        if self.decode is None:
            return self.config.max_local_tokens
        return self.decode.capacity - self.decode.reserved[(rank,)]

    def _batch_seconds(self, rank, now):
        """B.exec_time: how long this instance's decoding batch has been running."""
        started = [self.decode.started[rid] for rid in self.decode.running[(rank,)]]
        return now - min(started) if started else 0.0

    def next_plan(self, state):
        busy = set(state.busy_ranks)
        release = tuple(
            r.prompt.request_id
            for r in state.requests
            if r.complete and not r.in_flight and not busy.intersection(r.ranks)
        )
        # Finished batches retain their workers until their independent cleanup
        # commits. Unrelated idle workers can admit new work in the same poll.
        occupied = busy | {rank for r in state.requests for rank in r.ranks}
        free = set(range(self.config.world_size)) - occupied
        pending = sorted(
            (
                r
                for r in state.requests
                if not r.ranks and not r.in_flight and not r.complete
            ),
            key=lambda r: r.submitted_at,
        )
        for r in pending:
            if (
                len(r.prompt.token_ids)
                > self.config.world_size * self.config.max_local_tokens
            ):
                raise ValueError(
                    f"Prompt {r.prompt.request_id} exceeds aggregate KV capacity"
                )
        # Sec. 5.2: idle instances go to the prefill batch first.
        instances = [rank for rank in free if not self._decoding(rank)]
        chosen, preempted = self._dispatch(pending, instances, free, state.now)
        if chosen:
            preempted += self._expand(chosen, instances, free)
        steps, batches = self._batch(chosen, instances)
        self.last_decision = {
            "version": state.version,
            "batches": batches,
            "preempted_decode_instances": preempted,
        }
        if not steps and not release:
            return None
        return PrefillPlan(
            state.version, tuple(steps), tuple(r.prompt for r in chosen), release
        )

    def _dispatch(self, pending, instances, free, now):
        """Sec. 5.1: FCFS admission under memory, compute and preemption checks.

        Grows `instances` in place when admitting a request preempts decoding
        instances; returns the admitted requests and the preempted ranks.
        """
        chosen, lengths, preempted = [], [], []
        for request in pending:
            length = len(request.prompt.token_ids)
            group = self._preempt_for_memory(lengths + [length], instances, free)
            if group is None:
                break  # FCFS, never skip an inconvenient long request.
            if group and not self._worth_preempting(
                lengths + [length], instances, group, [length], now
            ):
                break
            candidate = instances + group
            # Compute check: stop once the batch passes the tipping point.
            if chosen and self.predict(len(candidate), lengths + [length]) > (
                self.max_prefill_seconds
            ):
                break
            instances.extend(group)
            preempted.extend(group)
            chosen.append(request)
            lengths.append(length)
        return chosen, preempted

    def _fits(self, lengths, instances):
        """Memory check: prefill KV, plus each prompt's future decode KV."""
        if sum(lengths) > len(instances) * self.config.max_local_tokens:
            return False
        if self.decode is None:
            return True
        need = max(lengths) + self.decode.decode_tokens - 1
        return any(
            self._unused_slots(rank) >= need for rank in range(self.config.world_size)
        )

    def _preempt_for_memory(self, lengths, instances, free):
        """Sec. 5.2: when idle instances lack slots, take decoding instances with
        the most unused slots. Returns the added ranks, or None if nothing fits."""
        group = []
        candidates = sorted(
            (rank for rank in free if rank not in instances),
            key=lambda rank: (-self._unused_slots(rank), rank),
        )
        while not (instances or group) or not self._fits(lengths, instances + group):
            if not candidates:
                return None
            group.append(candidates.pop(0))
        return group

    def _worth_preempting(self, lengths, instances, group, added, now):
        """Sec. 5.1 Eq. (1) cost versus Eq. (2) gain of preempting `group`.

        Cost: every decoding request on a preempted instance stalls for the new
        prefill iteration, normalized by its output length. Gain: the added
        requests skip waiting for those batches, normalized by input length.
        """
        stall = self.predict(len(instances) + len(group), lengths)
        cost = sum(
            stall / self.decode.decode_tokens
            for rank in group
            for _ in self.decode.running[(rank,)]
        )
        waited = min(self._batch_seconds(rank, now) for rank in group)
        gain = sum(max(0.0, self.decode.average_seconds - waited) / n for n in added)
        return gain > cost

    def _expand(self, chosen, instances, free):
        """Sec. 5.2: add e_min, the instance with the fewest used KV slots, while
        the Eq. (3) latency gain exceeds the Eq. (4) migration cost."""
        lengths = [len(r.prompt.token_ids) for r in chosen]
        added = []
        while True:
            rest = [rank for rank in free if rank not in instances]
            if not rest:
                return added
            e_min = max(rest, key=lambda rank: (self._unused_slots(rank), -rank))
            before = self.predict(len(instances), lengths)
            after = self.predict(len(instances) + 1, lengths)
            gain = sum((before - after) / n for n in lengths)  # Eq. (3)
            moved = 0.0
            if self.decode is not None:
                moved = self.decode.reserved[(e_min,)] * self.kv_bytes_per_token
            cost = sum(moved / (self.bandwidth * n) for n in lengths)  # Eq. (4)
            if gain <= cost:
                return added
            instances.append(e_min)
            if self._decoding(e_min):
                added.append(e_min)

    def _batch(self, chosen, instances):
        """Sec. 5.3: longest requests first, instances by location then unused
        slots ascending; the DP splits both into contiguous batches."""
        if not chosen:
            return [], []
        chosen.sort(key=lambda r: len(r.prompt.token_ids), reverse=True)
        lengths = [len(r.prompt.token_ids) for r in chosen]
        workers = sorted(
            instances, key=lambda rank: (self.nodes[rank], self._unused_slots(rank))
        )
        steps, batches = [], []
        for begin, end, first, last in self.planner.partition(
            lengths, [self.config.max_local_tokens] * len(workers)
        ):
            ranks = tuple(workers[first:last])
            offset = 0
            for request, length in zip(chosen[begin:end], lengths[begin:end]):
                # Stripe the concatenated batch, so per-request remainders
                # do not all consume an extra slot on the same worker.
                rotation = offset % len(ranks)
                owners = ranks[rotation:] + ranks[:rotation]
                steps.append(PromptStep(request.prompt.request_id, owners, length))
                offset += length
            batches.append(
                {
                    "requests": [r.prompt.request_id for r in chosen[begin:end]],
                    "ranks": ranks,
                    "predicted_seconds": self.predict(len(ranks), lengths[begin:end]),
                }
            )
        return steps, batches

    def choose_decode_instance(self, length, ranks, need):
        """Sec. 5.4 elastic scaling plan for one finished prefill.

        Scale down to the minimum degree that holds the KV: one instance, since
        multi-master decoding is not implemented. Prefer the request's own
        prefill instances, which keep the stripe they computed (proactive
        migration). Scale up past an instance whose batch reached the profiled
        compute-bound size by choosing another instance. None waits for slots.
        """
        fits = [
            rank
            for rank in range(self.config.world_size)
            if self._unused_slots(rank) >= need
        ]
        if not fits:
            if self.decode.capacity < need:
                raise ValueError("Request needs multi-master decoding (unsupported)")
            return None
        return (
            min(
                fits,
                key=lambda rank: (
                    len(self.decode.running[(rank,)]) >= self.batch_limit,
                    rank not in ranks,
                    -self._unused_slots(rank),
                    rank,
                ),
            ),
        )
