# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Shared wall throughput boundaries for online SP and native DP runs."""

import math


def throughput_summary(records):
    """Use completed requests, each counted once, on one monotonic clock.

    Timestamps are offsets from each run's origin, not clocks from GPU workers.
    Includes queueing and idle gaps. Does not estimate saturated device capacity.
    """
    if not records:
        return {
            "completed_prompts": 0,
            "completed_input_tokens": 0,
            "throughput_start_seconds": None,
            "throughput_end_seconds": None,
            "prefill_wall_seconds": 0.0,
            "useful_tokens_per_second": 0.0,
        }
    for r in records:
        start, end = r["submission_time_seconds"], r["completion_time_seconds"]
        if not all(math.isfinite(v) for v in (start, end)) or end <= start:
            raise ValueError("Completion must follow submission on the same clock")
        if type(r["input_tokens"]) is not int or r["input_tokens"] <= 0:
            raise ValueError("Completed input token counts must be positive integers")
    start = min(r["submission_time_seconds"] for r in records)
    end = max(r["completion_time_seconds"] for r in records)
    tokens = sum(r["input_tokens"] for r in records)
    return {
        "completed_prompts": len(records),
        "completed_input_tokens": tokens,
        "throughput_start_seconds": start,
        "throughput_end_seconds": end,
        "prefill_wall_seconds": end - start,
        "useful_tokens_per_second": tokens / (end - start),
        "throughput_definition": "Completed input tokens / (last completion - first actual submission); includes queueing and idle gaps",
        "timestamp_origin": "Seconds relative to benchmark start on the coordinator monotonic clock",
    }
