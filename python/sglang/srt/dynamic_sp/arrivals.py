# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Reproducible Poisson arrivals and an independent benchmark request producer."""

import math
import queue
import random
import threading
import time


def poisson_arrival_times(count: int, rate: float, seed: int = 0):
    """Arrival offsets in seconds, including the first exponential interarrival."""
    if count < 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Poisson arrival rate must be finite and positive")
    rng = random.Random(seed)
    now = 0.0
    offsets = []
    for _ in range(count):
        now += rng.expovariate(rate)
        offsets.append(now)
    return offsets


def request_rate_for_token_load(prompt_lengths, offered_tokens_per_second):
    """Convert expected input token load to a Poisson request rate.

    For a workload with mean length E[L], lambda * E[L] is the expected
    input token arrival rate. This does not estimate GPU utilization.
    """
    if not math.isfinite(offered_tokens_per_second) or offered_tokens_per_second <= 0:
        raise ValueError("Offered token load must be finite and positive")
    lengths = tuple(prompt_lengths)
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("Prompt lengths must be nonempty and positive")
    return offered_tokens_per_second / (sum(lengths) / len(lengths))


def trace_arrival_times(timestamps, prompt_lengths, offered_tokens_per_second):
    """Offsets of real timestamps, time-compressed to the offered token load.

    Gaps keep their recorded proportions; the span is set to total tokens over
    the load, the expected span of Poisson arrivals at the same load. Offsets
    follow the input order, which need not be sorted.
    """
    if not math.isfinite(offered_tokens_per_second) or offered_tokens_per_second <= 0:
        raise ValueError("Offered token load must be finite and positive")
    first, last = min(timestamps), max(timestamps)
    if last <= first:
        raise ValueError("Trace timestamps must span a positive interval")
    scale = sum(prompt_lengths) / offered_tokens_per_second / (last - first)
    return [(t - first) * scale for t in timestamps], scale


class ArrivalSource:
    """Queue arrivals while workers execute; only rank zero owns this source.

    The producer does not inspect model state or submit GPU work. The scheduler
    drains the queue at execution boundaries. Recorded arrival timestamps are
    scheduled monotonic times, so producer delays do not hide queueing latency.
    """

    def __init__(self, arrivals, origin):
        self._arrivals = tuple(arrivals)
        self._origin = origin
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._produce, daemon=True, name="dynamic-sp-arrivals"
        )
        self._thread.start()

    def _produce(self):
        try:
            for offset, prompt in self._arrivals:
                timestamp = self._origin + offset
                if self._stop.wait(max(0.0, timestamp - time.perf_counter())):
                    break
                self._queue.put((timestamp, prompt, time.perf_counter() - timestamp))
        except Exception as exc:  # noqa: BLE001 - forward producer failures to workers
            self._error = exc
        finally:
            self._done.set()

    def drain(self):
        if self._error is not None:
            raise RuntimeError("Arrival producer failed") from self._error
        available = []
        while True:
            try:
                available.append(self._queue.get_nowait())
            except queue.Empty:
                return available

    @property
    def exhausted(self):
        # The producer never enqueues after setting done. Read done first.
        done = self._done.is_set()
        if self._error is not None:
            raise RuntimeError("Arrival producer failed") from self._error
        return done and self._queue.empty()

    def close(self):
        self._stop.set()
        self._thread.join()
