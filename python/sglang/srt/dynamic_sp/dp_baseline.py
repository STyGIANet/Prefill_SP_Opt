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


async def benchmark(
    engine, requests, *, progress_interval=10.0, progress_throughput_target=0.0
):
    """Open-loop arrivals: submit without waiting for earlier completions."""
    origin = time.perf_counter()
    completed = []
    submitted = completed_tokens = 0
    peak_rate = 0.0

    def show_progress():
        nonlocal peak_rate
        elapsed = time.perf_counter() - origin
        duration = max(0.0, elapsed - requests[0]["arrival_time_seconds"])
        rate = completed_tokens / duration if duration > 0 else 0.0
        peak_rate = max(peak_rate, rate)
        scale = progress_throughput_target or peak_rate
        reference = "offered_load" if progress_throughput_target else "observed_peak"

        def bar(fraction):
            filled = max(0, min(16, int(16 * fraction)))
            return "[" + "#" * filled + "-" * (16 - filled) + "]"

        latencies = sorted(r["ttft_seconds"] for r in completed)
        p50 = f"{statistics.median(latencies):.3f}s" if latencies else "n/a"
        p95 = (
            f"{latencies[math.ceil(0.95 * len(latencies)) - 1]:.3f}s"
            if latencies
            else "n/a"
        )
        print(
            f"[dynamic-sp] progress {bar(len(completed) / len(requests))} baseline=sglang-dp "
            f"prompts={len(completed)}/{len(requests)} elapsed={elapsed:.1f}s "
            f"outstanding={submitted - len(completed)} not_arrived={len(requests) - submitted} "
            f"completed_input_tokens={completed_tokens}",
            flush=True,
        )
        print(
            f"[dynamic-sp] throughput {bar(rate / scale if scale > 0 else 0)} "
            f"completed_input_tokens/sec={rate:.1f} {reference}_tokens/sec={scale:.1f} "
            f"TTFT_p50={p50} TTFT_p95={p95} completed_prompts={len(completed)}",
            flush=True,
        )

    async def report_progress():
        while True:
            show_progress()
            await asyncio.sleep(progress_interval)

    async def submit(request):
        nonlocal submitted, completed_tokens
        scheduled = origin + request["arrival_time_seconds"]
        dispatched = time.perf_counter()
        submitted += 1
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
        record = {
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

        completed.append(record)
        completed_tokens += record["input_tokens"]
        return record

    tasks = []
    reporter = asyncio.create_task(report_progress()) if progress_interval > 0 else None
    try:
        for request in requests:
            delay = origin + request["arrival_time_seconds"] - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(submit(request)))
        results = await asyncio.gather(*tasks)
        if reporter is not None:
            show_progress()
    finally:
        if reporter is not None:
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)
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


def prepare_runtime_workload(engine, requests, args):
    """Filter whole records using actual per-replica limits before arrivals start."""
    infos = engine._scheduler_init_result.scheduler_infos
    # Native admission rejects input length >= max_req_input_len. Include the
    # tokenizer-side limit and every reported replica, not just the requested cap.
    limits = [args.max_local_tokens - 1, engine.tokenizer_manager.max_req_input_len - 1]
    for info in infos:
        limits.extend((info["max_req_input_len"] - 1, info["max_total_num_tokens"] - 1))
    maximum = min(limits)
    kept, excluded = [], []
    for request in requests:
        count = len(request["token_ids"])
        if count <= maximum:
            kept.append(request)
        else:
            excluded.append(
                {
                    "request_id": request["request_id"],
                    "input_tokens": count,
                    "reason": "exceeds_runtime_dp_capacity",
                }
            )
    output = Path(args.output)
    workload_path = output.with_name(output.stem + ".workload.json")
    report = {
        "max_accepted_input_tokens": maximum,
        "requested_kv_tokens_per_worker": args.max_local_tokens,
        "source_prompts": len(requests),
        "evaluated_prompts": len(kept),
        "excluded_prompts": len(excluded),
        "excluded_records": excluded,
        "evaluated_workload": str(workload_path),
        "arrival_policy": "Original scheduled arrival offsets preserved",
        "source_workload_sha256": hashlib.sha256(
            Path(args.workload).read_bytes()
        ).hexdigest(),
    }
    data = (
        (json.dumps(kept) + "\n").encode()
        if excluded
        else Path(args.workload).read_bytes()
    )
    workload_path.write_bytes(data)
    report["evaluated_workload_sha256"] = hashlib.sha256(data).hexdigest()
    output.with_name(output.stem + ".capacity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"[dynamic-sp] DP runtime input limit={maximum} tokens; "
        f"accepted={len(kept)}/{len(requests)} excluded={len(excluded)}; "
        f"no truncation; comparison workload: {workload_path}",
        flush=True,
    )
    if not kept:
        raise ValueError(
            "No complete prompt fits the native DP runtime capacity; see capacity report"
        )
    return kept, report


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
    parser.add_argument("--max-local-tokens", type=int, default=600000)
    parser.add_argument(
        "--max-running-requests", type=int, default=32, help="Per DP worker"
    )
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--progress-throughput-target", type=float, default=0.0)
    args = parser.parse_args()
    if any(
        not math.isfinite(v) or v < 0
        for v in (args.progress_interval, args.progress_throughput_target)
    ):
        parser.error("Progress interval and target must be finite and nonnegative")
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
        requests, capacity = prepare_runtime_workload(engine, requests, args)
        results = engine.loop.run_until_complete(
            benchmark(
                engine,
                requests,
                progress_interval=args.progress_interval,
                progress_throughput_target=args.progress_throughput_target,
            )
        )
        results["capacity_filter"] = capacity
        results["source_workload_sha256"] = capacity["source_workload_sha256"]
        results["engine_settings"] = settings
        results["workload_sha256"] = capacity["evaluated_workload_sha256"]
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
