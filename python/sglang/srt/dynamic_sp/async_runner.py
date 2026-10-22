# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Event-driven online runner for the independently scheduled SP groups."""

import copy
import json
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch.distributed as dist
from sglang.srt.dynamic_sp.arrivals import ArrivalSource
from sglang.srt.dynamic_sp.async_executor import AsyncExecutor
from sglang.srt.dynamic_sp.async_scheduler import AsyncScheduler
from sglang.srt.dynamic_sp.engine import PlanRejected
from sglang.srt.dynamic_sp.metrics import throughput_summary
from sglang.srt.dynamic_sp.scheduler import GreedyScheduler, SchedulerConfig


def run_async(args, model, control, scheduler, arrivals, plans):
    options = {
        "max_local_tokens": args.max_local_tokens,
        "memory_execution_tokens": args.max_batch_tokens,
        "kv_block_size": args.kv_block_size,
        "query_block_size": args.query_block_size,
        "attention_backend": args.attention_backend,
        "logits_mode": "all" if args.verify else "completed",
        "kv_transport": args.kv_transport,
        "shared_attention": args.shared_attention,
        "migration_overlap": getattr(args, "migration_overlap", True),
    }
    executor = AsyncExecutor(
        model,
        control,
        options,
        verify=args.verify,
        atol=args.atol,
        rtol=args.rtol,
        timeout=args.async_timeout,
    )
    if dist.get_rank() != 0:
        executor.serve()
        return
    warmup_count = getattr(args, "warmup_requests", 0)
    if warmup_count:
        from sglang.srt.dynamic_sp.benchmark import warmup_arrivals

        if plans is not None:
            raise ValueError(
                "Warmup requires an online workload, not a replay schedule"
            )
        warmup = warmup_arrivals(
            arrivals, warmup_count, token_budget=getattr(args, "warmup_tokens", 8192)
        )
        print(
            f"[dynamic-sp] warmup prompts={len(warmup)} scheduler={args.scheduler}",
            flush=True,
        )
        # One prompt at a time: the next warmup starts only after the previous
        # request and its KV have been released. Full measured prompts are untouched.
        for arrival in warmup:
            _run_phase(
                args,
                model,
                copy.deepcopy(scheduler),
                [arrival],
                None,
                executor,
                warmup=True,
            )
            executor.reset()
        print(
            "[dynamic-sp] warmup complete; KV slots released; starting measured arrivals",
            flush=True,
        )
    executor.reset()
    _run_phase(args, model, scheduler, arrivals, plans, executor)


def _run_phase(args, model, scheduler, arrivals, plans, executor, *, warmup=False):
    config = SchedulerConfig(
        dist.get_world_size(),
        args.max_local_tokens,
        args.chunk_size,
        args.max_running_requests,
        args.max_batch_tokens,
    )
    policy = scheduler.policy if scheduler is not None else GreedyScheduler(config)
    driver = AsyncScheduler(policy, config, model)
    print(
        f"[dynamic-sp] execution=async workers={config.world_size} scheduler={args.scheduler} verify={args.verify}",
        flush=True,
    )
    origin = time.perf_counter()
    source = ArrivalSource(arrivals, origin) if plans is None else None
    timings, pending, arrival_records, results, executed, decisions = (
        [],
        {},
        [],
        [],
        [],
        {},
    )
    replay_index = 0
    progress_interval = getattr(args, "progress_interval", 10.0)
    next_progress = origin
    processed_tokens = 0
    first_arrival = None
    peak_rate = 0.0
    last_reported_completed = -1
    total_prompts = len(arrivals) if plans is None else sum(len(p.admit) for p in plans)
    try:
        while True:
            events = executor.poll(timeout=0)
            for operation, responses in events:
                old_ends = {
                    prompt.request_id: end for prompt, _, end in operation.previous
                }
                driver.completed(operation, responses)
                if responses[0]["status"] != "committed":
                    raise PlanRejected("; ".join(r["error"] for r in responses))
                leader = next(r for r in responses if r["rank"] == operation.members[0])
                for rid, token in leader["output_ids"].items():
                    timing = pending.pop(rid)
                    timing["completion_time_seconds"] = leader["ready_at"] - origin
                    timing["response_observed_time_seconds"] = (
                        time.perf_counter() - origin
                    )
                    if not (
                        timing["arrival_time_seconds"]
                        <= timing["submission_time_seconds"]
                        < timing["completion_time_seconds"]
                        <= timing["response_observed_time_seconds"]
                    ):
                        raise RuntimeError(
                            "Invalid prompt timestamp ordering on shared clock"
                        )
                    timing["ttft_seconds"] = (
                        timing["completion_time_seconds"]
                        - timing["arrival_time_seconds"]
                    )
                    timing["response_delivery_seconds"] = (
                        timing["response_observed_time_seconds"]
                        - timing["completion_time_seconds"]
                    )
                    timing["output_ids"] = [token]
                    timings.append(timing)
                record = {
                    "ticket": operation.ticket,
                    "members": operation.members,
                    "new_tokens": sum(
                        s.end - old_ends.get(s.request_id, 0)
                        for s in operation.plan.steps
                    ),
                    "worker_start_seconds": min(r["started_at"] for r in responses)
                    - origin,
                    "worker_end_seconds": max(r["ready_at"] for r in responses)
                    - origin,
                    "seconds": max(r["seconds"] for r in responses),
                    "worker_metrics": [
                        r["metrics"] for r in sorted(responses, key=lambda r: r["rank"])
                    ],
                    "max_abs_logit_error": leader["errors"],
                    "verified": args.verify,
                    "requests": driver.snapshot(),
                    "scheduling": decisions.pop(operation.ticket),
                }
                results.append(record)
                processed_tokens += record["new_tokens"]
            now = time.perf_counter()
            if source is not None:
                for timestamp, prompt, delay in source.drain():
                    driver.submit(prompt, timestamp)
                    pending[prompt.request_id] = {
                        "request_id": prompt.request_id,
                        "input_tokens": len(prompt.token_ids),
                        "arrival_time_seconds": timestamp - origin,
                        "submission_time_seconds": timestamp + delay - origin,
                    }
                    arrival_records.append(
                        {
                            "request_id": prompt.request_id,
                            "arrival_time_seconds": timestamp - origin,
                            "producer_delay_seconds": delay,
                            "scheduler_observation_delay_seconds": now - timestamp,
                        }
                    )
            if (
                not warmup
                and progress_interval > 0
                and (
                    now >= next_progress
                    or (
                        len(timings) == total_prompts
                        and total_prompts > 0
                        and last_reported_completed != total_prompts
                    )
                )
            ):
                if first_arrival is None and pending:
                    first_arrival = min(
                        t["arrival_time_seconds"] for t in pending.values()
                    )
                if timings:
                    first_completed_arrival = min(
                        t["arrival_time_seconds"] for t in timings
                    )
                    first_arrival = (
                        first_completed_arrival
                        if first_arrival is None
                        else min(first_arrival, first_completed_arrival)
                    )
                elapsed = now - origin
                duration = (
                    max(0.0, elapsed - first_arrival)
                    if first_arrival is not None
                    else 0.0
                )
                rate = processed_tokens / duration if duration > 0 else 0.0
                peak_rate = max(peak_rate, rate)
                target = getattr(args, "progress_throughput_target", 0.0)
                scale = target if target > 0 else peak_rate

                def bar(fraction):
                    filled = min(16, max(0, int(16 * fraction)))
                    return "[" + "#" * filled + "-" * (16 - filled) + "]"

                active = [r for r in driver.requests.values() if not r.complete]
                running = sum(r.prompt.request_id in driver.inflight for r in active)
                latencies = sorted(t["ttft_seconds"] for t in timings)
                p50 = f"{statistics.median(latencies):.3f}s" if latencies else "n/a"
                p95 = (
                    f"{latencies[math.ceil(0.95 * len(latencies)) - 1]:.3f}s"
                    if latencies
                    else "n/a"
                )
                denominator = target if target > 0 else peak_rate
                reference = "offered_load" if target > 0 else "observed_peak"
                print(
                    f"[dynamic-sp] progress {bar(len(timings) / max(1, total_prompts))} "
                    f"prompts={len(timings)}/{total_prompts} elapsed={elapsed:.1f}s "
                    f"queued={len(active) - running} running={running} busy_gpus={len(executor.busy)} "
                    f"prefill_tokens={processed_tokens}",
                    flush=True,
                )
                print(
                    f"[dynamic-sp] throughput {bar(rate / scale if scale > 0 else 0)} "
                    f"prefill_tokens/sec={rate:.1f} {reference}_tokens/sec={denominator:.1f} "
                    f"TTFT_p50={p50} TTFT_p95={p95} completed_prompts={len(latencies)}",
                    flush=True,
                )
                last_reported_completed = len(timings)
                next_progress = now + progress_interval
            scheduling_start = time.perf_counter()
            if plans is not None:
                # Replay explicitly preserves plan boundaries. Independent
                # components within each recorded plan still execute concurrently.
                if executor.inflight:
                    plan = None
                elif replay_index < len(plans):
                    original = plans[replay_index]
                    for prompt in original.admit:
                        driver.submit(prompt, now)
                        pending[prompt.request_id] = {
                            "request_id": prompt.request_id,
                            "input_tokens": len(prompt.token_ids),
                            "arrival_time_seconds": now - origin,
                            "submission_time_seconds": now - origin,
                        }
                    plan = replace(original, version=driver.version)
                    replay_index += 1
                else:
                    plan = None
            else:
                plan = driver.next_plan(now, executor.busy)
            if plan is not None:
                operations = driver.operations(plan, now, executor.busy)
                decision = {
                    "seconds": time.perf_counter() - scheduling_start,
                    "policy": getattr(policy, "last_decision", None),
                }
                for operation in operations:
                    decisions[operation.ticket] = decision
                    executed.append(
                        {
                            **asdict(operation.plan),
                            "ticket": operation.ticket,
                            "members": operation.members,
                            "dispatch_time_seconds": time.perf_counter() - origin,
                        }
                    )
                    executor.submit(operation)
                continue
            exhausted = (
                source.exhausted if source is not None else replay_index == len(plans)
            )
            if exhausted and not executor.inflight:
                if source is not None and driver.requests:
                    raise RuntimeError(
                        "Async policy made no progress with pending requests and no leased workers"
                    )
                break
            # No global barrier: every poll processes whichever group completes.
            # Arrivals are drained at most 1 ms later while GPUs remain active.
            time.sleep(0.001)
        if not warmup:
            executor.close()
    except BaseException:
        # Drain healthy groups before shutdown. If transport itself is broken,
        # let torchrun terminate the job instead of entering a world barrier.
        try:
            while executor.inflight:
                executor.poll(timeout=0.01)
            executor.close()
        except BaseException:  # noqa: BLE001 - transport failure requires fail-stop
            args.async_failed = True
            for connection in executor.connections.values():
                connection.close()
        raise
    finally:
        if source is not None:
            source.close()
    if warmup:
        return
    wall = throughput_summary(timings)
    latencies = sorted(
        t["completion_time_seconds"] - t["arrival_time_seconds"] for t in timings
    )
    summary = {
        **wall,
        "prefill_tokens": sum(r["new_tokens"] for r in results),
        "wall_tokens_per_second": wall["useful_tokens_per_second"],
        "ttft_p50_seconds": statistics.median(latencies) if latencies else None,
        "ttft_p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1]
        if latencies
        else None,
        "p95_method": "nearest rank",
        "completion_boundary": "first_token_ready_before_response_ipc",
        "ttft_definition": "Scheduled arrival to committed sampled token ready on group leader; excludes response IPC and HTTP",
        "clock": "Linux CLOCK_MONOTONIC, shared boot and time namespace verified",
        "clock_domain": executor.clock_domain,
        "clock_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        "allocator_retries": sum(
            m.get("allocator_retries", 0) for r in results for m in r["worker_metrics"]
        ),
        "allocator_ooms": sum(
            m.get("allocator_ooms", 0) for r in results for m in r["worker_metrics"]
        ),
        "peak_allocated_bytes": max(
            (
                m.get("peak_allocated_bytes", 0)
                for r in results
                for m in r["worker_metrics"]
            ),
            default=0,
        ),
        "peak_reserved_bytes": max(
            (
                m.get("peak_reserved_bytes", 0)
                for r in results
                for m in r["worker_metrics"]
            ),
            default=0,
        ),
        "kv_storage": "native_sglang_slot_pool",
        "verification_enabled": args.verify,
        "warmup_requests": min(
            getattr(args, "warmup_requests", 0),
            len(arrivals),
            getattr(args, "warmup_tokens", 8192),
        ),
        "allow_profile_backend_mismatch": getattr(
            args, "allow_profile_backend_mismatch", False
        ),
        "execution_seconds_definition": "Concurrent operation durations are not summed into a throughput denominator",
    }
    print(
        f"[dynamic-sp] execution=async prefill_tokens={summary['prefill_tokens']} "
        f"window_start={wall['throughput_start_seconds']}s "
        f"window_end={wall['throughput_end_seconds']}s "
        f"window_seconds={wall['prefill_wall_seconds']:.6f} throughput_tokens/s={wall['useful_tokens_per_second']:.1f}",
        flush=True,
    )
    if latencies:
        print(
            f"[dynamic-sp] per-prompt TTFT: p50={summary['ttft_p50_seconds']:.3f}s "
            f"p95={summary['ttft_p95_seconds']:.3f}s prompts={len(latencies)}; excludes response IPC and HTTP",
            flush=True,
        )
    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "execution_mode": "async",
                    "world_size": config.world_size,
                    "model": args.model_path,
                    "dtype": args.dtype,
                    "scheduler": args.scheduler if plans is None else "replay",
                    "scheduler_settings": {
                        **vars(config),
                        "tetris": policy.metadata()
                        if hasattr(policy, "metadata")
                        else None,
                    },
                    "arrival_records": arrival_records,
                    "request_timings": timings,
                    "executed_plans": executed,
                    "results": results,
                    "summary": summary,
                },
                indent=2,
            )
            + "\n"
        )
