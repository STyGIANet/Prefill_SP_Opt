# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Native SGLang DP baseline with the same scheduled tokenized arrivals as SP."""

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from sglang.srt.dynamic_sp.metrics import throughput_summary


def read_workload(path):
    requests = json.loads(Path(path).read_text())
    if not isinstance(requests, list) or not requests:
        raise ValueError("Workload must be a nonempty request array")
    seen = set()
    for request in requests:
        rid = request["request_id"]
        if not isinstance(rid, str) or rid in seen:
            raise ValueError("Request IDs must be unique strings")
        seen.add(rid)
        tokens = request["token_ids"]
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Each prompt must contain nonnegative integer token IDs")
        offset = request.setdefault("arrival_time_seconds", 0.0)
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("Arrival offsets must be finite and nonnegative")
    return sorted(requests, key=lambda r: r["arrival_time_seconds"])


async def benchmark(engine, requests):
    """Open-loop arrivals: submit without waiting for earlier completions."""
    origin = time.perf_counter()

    async def submit(request):
        scheduled = origin + request["arrival_time_seconds"]
        dispatched = time.perf_counter()
        result = await engine.async_generate(
            input_ids=request["token_ids"],
            sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
            rid=request["request_id"],
        )
        finished = time.perf_counter()
        if result.get("meta_info", {}).get("completion_tokens") != 1:
            raise RuntimeError(
                f"Request {request['request_id']} did not return one token"
            )
        ready = result["meta_info"].get("first_token_ready_time")
        if (
            ready is None
            or not math.isfinite(ready)
            or not dispatched <= ready <= finished
        ):
            raise RuntimeError(
                "Missing or invalid scheduler first-token timestamp; use this checkout's native Engine"
            )
        return {
            "request_id": request["request_id"],
            "input_tokens": len(request["token_ids"]),
            "arrival_time_seconds": request["arrival_time_seconds"],
            "submission_time_seconds": dispatched - origin,
            "dispatch_delay_seconds": dispatched - scheduled,
            "ttft_seconds": ready - scheduled,
            "response_ttft_seconds": finished - scheduled,
            "response_delivery_seconds": finished - ready,
            "completion_time_seconds": ready - origin,
            "response_completion_time_seconds": finished - origin,
            "output_ids": result.get("output_ids"),
        }

    tasks = []
    try:
        for request in requests:
            delay = origin + request["arrival_time_seconds"] - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(submit(request)))
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    latencies = sorted(r["ttft_seconds"] for r in results)
    response_latencies = sorted(r["response_ttft_seconds"] for r in results)
    response_records = [
        dict(r, completion_time_seconds=r["response_completion_time_seconds"])
        for r in results
    ]
    return {
        "scheduler": "sglang-dp",
        "dispatch_policy": "round_robin",
        "worker_schedule_policy": "fcfs",
        "results": results,
        "summary": {
            **throughput_summary(results),
            "ttft_p50_seconds": statistics.median(latencies),
            "ttft_p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "p95_method": "nearest rank",
            "ttft_definition": "Scheduled arrival to first sampled token ready on CPU; excludes response IPC and HTTP",
            "completion_boundary": "first_token_ready_before_response_ipc",
            "response_ttft_p50_seconds": statistics.median(response_latencies),
            "response_ttft_p95_seconds": response_latencies[
                math.ceil(0.95 * len(response_latencies)) - 1
            ],
            "response_throughput": throughput_summary(response_records),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16")
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-local-tokens", type=int, default=65536)
    parser.add_argument(
        "--max-running-requests", type=int, default=32, help="Per DP worker"
    )
    args = parser.parse_args()
    if (
        min(
            args.dp_size,
            args.chunk_size,
            args.max_local_tokens,
            args.max_running_requests,
        )
        <= 0
    ):
        parser.error("Worker count and token/request limits must be positive")
    if Path(args.output).exists():
        parser.error("Choose a fresh results path")
    requests = read_workload(args.workload)
    if max(len(r["token_ids"]) for r in requests) + 1 > args.max_local_tokens:
        parser.error("Each prompt plus its output token must fit on one worker")

    from sglang.srt.entrypoints.engine import Engine

    settings = {
        "model_path": args.model_path,
        "dp_size": args.dp_size,
        "tp_size": 1,
        "pp_size": 1,
        "dtype": args.dtype,
        "load_balance_method": "round_robin",
        "schedule_policy": "fcfs",
        "chunked_prefill_size": args.chunk_size,
        "max_total_tokens": args.max_local_tokens,
        "max_running_requests": args.max_running_requests,
        "disable_radix_cache": True,
        "skip_tokenizer_init": True,
        "disable_cuda_graph": True,
        "log_level": "error",
    }
    engine = Engine(**settings)
    try:
        results = engine.loop.run_until_complete(benchmark(engine, requests))
        results["engine_settings"] = settings
        results["workload_sha256"] = hashlib.sha256(
            Path(args.workload).read_bytes()
        ).hexdigest()
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        summary = results["summary"]
        print(
            f"[dynamic-sp] baseline=sglang-dp workers={args.dp_size} "
            f"prompts={summary['completed_prompts']} "
            f"window_start={summary['throughput_start_seconds']:.6f}s "
            f"window_end={summary['throughput_end_seconds']:.6f}s "
            f"window_seconds={summary['prefill_wall_seconds']:.6f} "
            f"throughput_tokens/s={summary['useful_tokens_per_second']:.1f} "
            f"TTFT_p50={summary['ttft_p50_seconds']:.3f}s "
            f"TTFT_p95={summary['ttft_p95_seconds']:.3f}s",
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
