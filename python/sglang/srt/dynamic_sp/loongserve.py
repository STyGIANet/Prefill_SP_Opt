# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""LoongServe Sections 5.1-5.5 specialized to a prefill-only instance pool."""

import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from sglang.srt.dynamic_sp.plan import PrefillPlan, PromptStep


class BatchPlanner:
    """Compiled exact contiguous partition DP; compiled before measured arrivals."""

    def __init__(self, models):
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
            np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        ]
        self.solve.restype = ctypes.c_int

    def __deepcopy__(self, memo):
        return self  # Immutable coefficients and stateless compiled function.

    def partition(self, lengths, capacities):
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
    def __init__(self, config, profile, max_prefill_seconds=None):
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
        self.max_prefill_seconds = (
            profile["max_prefill_seconds"]
            if max_prefill_seconds is None
            else max_prefill_seconds
        )
        if not math.isfinite(self.max_prefill_seconds) or self.max_prefill_seconds <= 0:
            raise ValueError("LoongServe prefill time limit must be positive")
        self.planner = BatchPlanner(self.models)
        self.last_decision = {}

    @classmethod
    def from_file(cls, config, path, max_prefill_seconds=None):
        return cls(config, json.loads(Path(path).read_text()), max_prefill_seconds)

    def predict(self, degree, lengths):
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
            "last_decision": self.last_decision,
        }

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
        workers = sorted(
            set(range(self.config.world_size)) - occupied,
            key=lambda rank: (self.nodes[rank], rank),
        )
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
        chosen, lengths = [], []
        for request in pending:
            length = len(request.prompt.token_ids)
            if (
                not workers
                or sum(lengths) + length > len(workers) * self.config.max_local_tokens
            ):
                break  # FCFS, never skip an inconvenient long request.
            candidate = lengths + [length]
            if (
                chosen
                and self.predict(len(workers), candidate) > self.max_prefill_seconds
            ):
                break
            chosen.append(request)
            lengths.append(length)
        chosen.sort(key=lambda r: len(r.prompt.token_ids), reverse=True)
        lengths = [len(r.prompt.token_ids) for r in chosen]
        steps, batches = [], []
        if chosen:
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
                        "predicted_seconds": self.predict(
                            len(ranks), lengths[begin:end]
                        ),
                    }
                )
        self.last_decision = {"version": state.version, "batches": batches}
        if not steps and not release:
            return None
        return PrefillPlan(
            state.version, tuple(steps), tuple(r.prompt for r in chosen), release
        )
