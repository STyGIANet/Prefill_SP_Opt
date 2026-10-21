# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Event-driven online runner for the independently scheduled SP groups."""

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
        "verification_enabled": args.verify,
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
