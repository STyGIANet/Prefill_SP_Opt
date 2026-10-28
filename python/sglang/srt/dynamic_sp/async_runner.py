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
from sglang.srt.dynamic_sp.async_executor import AsyncExecutor, prewarm_memberships
from sglang.srt.dynamic_sp.async_scheduler import AsyncScheduler
from sglang.srt.dynamic_sp.decode import (
    DecodeCoordinator,
    DecodeStep,
    KVTransfer,
    decode_instances,
)
from sglang.srt.dynamic_sp.engine import PlanRejected
from sglang.srt.dynamic_sp.metrics import throughput_summary
from sglang.srt.dynamic_sp.scheduler import GreedyScheduler, SchedulerConfig


def run_async(args, model, control, scheduler, arrivals, plans):
    options = {
        "max_local_tokens": args.max_local_tokens,
        "memory_execution_tokens": (
            args.max_local_tokens
            if args.scheduler == "loongserve"
            else args.max_batch_tokens
        ),
        "kv_block_size": args.kv_block_size,
        "query_block_size": args.query_block_size,
        "attention_backend": args.attention_backend,
        "logits_mode": "all" if args.verify else "completed",
        "kv_transport": args.kv_transport,
        "shared_attention": args.shared_attention,
        "migration_overlap": getattr(args, "migration_overlap", True),
    }
    prefill_world = dist.get_world_size() - args.decode_gpus
    instances = _decode_instances(args)
    prewarm = prewarm_memberships(prefill_world, args.prewarm_groups)
    if prewarm:
        # Decode TP groups, then every owner-to-instance transfer group.
        extra = set(instances) | {
            tuple(sorted({owner, *instance}))
            for owner in range(prefill_world)
            for instance in instances
        }
        prewarm += tuple(sorted(extra - set(prewarm), key=lambda g: (len(g), g)))
    started = time.perf_counter()
    executor = AsyncExecutor(
        model,
        control,
        options,
        verify=args.verify,
        atol=args.atol,
        rtol=args.rtol,
        timeout=args.async_timeout,
        prewarm=prewarm,
        decode_instances=instances,
        decode_kv_tokens=args.decode_kv_tokens,
    )
    if dist.get_rank() != 0:
        executor.serve()
        return
    if args.prewarm_groups != "none":
        print(
            f"[dynamic-sp] prewarmed {args.prewarm_groups} groups "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
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
        dist.get_world_size() - args.decode_gpus,
        args.max_local_tokens,
        args.chunk_size,
        args.max_running_requests,
        args.max_batch_tokens,
    )
    policy = scheduler.policy if scheduler is not None else GreedyScheduler(config)
    driver = AsyncScheduler(policy, config, model)
    decode = (
        DecodeCoordinator(
            _decode_instances(args),
            args.decode_kv_tokens,
            args.decode_tokens,
            args.decode_max_batch,
        )
        if _decode_instances(args)
        else None
    )
    # LoongServe elastic mode: its global manager also places decoding (Sec.
    # 5.4) and plans prefill before decode steps claim the idle instances, so it
    # can weigh preempting them (Sec. 5.1-5.2).
    elastic = args.loongserve_elastic_decode
    if elastic:
        policy.attach_decode(
            decode,
            kv_bytes_per_token=2
            * model.num_layers
            * model.num_kv_heads
            * model.head_dim
            * model.dtype.itemsize,
            bandwidth=args.loongserve_migration_bandwidth,
            batch_limit=args.loongserve_decode_batch_limit,
        )
    records, decode_results = {}, []
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
                if isinstance(operation, (KVTransfer, DecodeStep)):
                    _decode_completed(
                        decode, driver, operation, responses, records, origin
                    )
                    decode_results.append(
                        {
                            "kind": type(operation).__name__,
                            "ticket": operation.ticket,
                            "members": operation.members,
                            "requests": [operation.request_id]
                            if isinstance(operation, KVTransfer)
                            else list(operation.run),
                            "worker_start_seconds": min(
                                r["started_at"] for r in responses
                            )
                            - origin,
                            "seconds": max(r["seconds"] for r in responses),
                        }
                    )
                    continue
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
                    timing["slo_violated"] = (
                        (timing["ttft_seconds"] > timing["ttft_slo_seconds"])
                        if timing.get("ttft_slo_seconds") is not None
                        else None
                    )
                    timing["output_ids"] = [token]
                    timings.append(timing)
                    if decode is not None:
                        request = driver.requests[rid]
                        decode.add(
                            rid, len(request.prompt.token_ids), request.ranks, token
                        )
                        records[rid] = timing
                record = {
                    "ticket": operation.ticket,
                    "members": operation.members,
                    "new_tokens": sum(
                        s.end
                        - old_ends.get(
                            s.request_id,
                            driver.requests[s.request_id].prompt.prefix_length,
                        )
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
                        "input_tokens": len(prompt.token_ids) - prompt.prefix_length,
                        "ttft_slo_seconds": prompt.ttft_slo_seconds,
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
            if decode is not None and not elastic:
                for operation in decode.operations(executor.busy, driver.ticket):
                    executor.submit(operation)
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
                            "input_tokens": len(prompt.token_ids)
                            - prompt.prefix_length,
                            "arrival_time_seconds": now - origin,
                            "submission_time_seconds": now - origin,
                        }
                    plan = replace(original, version=driver.version)
                    replay_index += 1
                else:
                    plan = None
            else:
                if hasattr(policy, "arrivals_exhausted"):
                    policy.arrivals_exhausted = source.exhausted
                plan = driver.next_plan(now, executor.busy)
            if plan is not None and decode is not None:
                # A finished prompt stays on its prefill owners until its KV ships.
                release = tuple(r for r in plan.release if not decode.holds(r))
                plan = replace(plan, release=release)
                if not plan.steps and not plan.release:
                    plan = None
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
            if decode is not None and elastic:
                for operation in decode.operations(executor.busy, driver.ticket):
                    executor.submit(operation)
            if plan is not None:
                continue
            exhausted = (
                source.exhausted if source is not None else replay_index == len(plans)
            )
            if exhausted and not executor.inflight and (decode is None or decode.idle):
                if driver.requests and getattr(policy, "wake_at", None) is not None:
                    time.sleep(
                        min(0.001, max(0.0, policy.wake_at - time.perf_counter()))
                    )
                    continue
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
        "slo_requests": sum(t.get("slo_violated") is not None for t in timings),
        "slo_violations": sum(t.get("slo_violated") is True for t in timings),
        "prefill_graph_replays": sum(
            m.get("prefill_graph_replays", 0)
            for r in results
            for m in r["worker_metrics"]
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
    if decode is not None:
        summary.update(_decode_summary(timings, args))
        print(
            f"[dynamic-sp] decode gpus={args.decode_gpus} tp={args.decode_tp} tokens={args.decode_tokens} "
            f"handoff_p50={summary['kv_handoff_p50_seconds']:.3f}s "
            f"TPOT_p50={summary['tpot_p50_seconds'] * 1e3:.1f}ms "
            f"TPOT_p95={summary['tpot_p95_seconds'] * 1e3:.1f}ms "
            f"E2E_p50={summary['e2e_p50_seconds']:.3f}s",
            flush=True,
        )
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
                    "decode_results": decode_results,
                    "summary": summary,
                },
                indent=2,
            )
            + "\n"
        )


def _decode_completed(decode, driver, operation, responses, records, origin):
    if responses[0]["status"] != "committed":
        raise RuntimeError(f"Decode operation failed: {responses}")
    ready = max(r["ready_at"] for r in responses) - origin
    done = decode.completed(operation)
    for rid in done["shipped"]:
        driver.hand_off(rid)
        records[rid]["kv_ready_time_seconds"] = ready
    if isinstance(operation, DecodeStep):
        leader = responses[0]
        for rid, token in leader["output_ids"].items():
            records[rid]["output_ids"].append(token)
    for rid in done["finished"]:
        timing = records.pop(rid)
        timing["decode_done_time_seconds"] = ready
        timing["kv_handoff_seconds"] = (
            timing["kv_ready_time_seconds"] - timing["completion_time_seconds"]
        )
        timing["tpot_seconds"] = (ready - timing["kv_ready_time_seconds"]) / (
            len(timing["output_ids"]) - 1
        )
        timing["e2e_seconds"] = ready - timing["arrival_time_seconds"]


def _decode_summary(timings, args):
    def percentiles(key):
        values = sorted(t[key] for t in timings)
        return values[len(values) // 2], values[math.ceil(0.95 * len(values)) - 1]

    if any("e2e_seconds" not in t for t in timings):
        raise RuntimeError("A request finished prefill but never finished decoding")
    summary = {
        "decode_gpus": args.decode_gpus,
        "decode_tp": args.decode_tp,
        "decode_tokens": args.decode_tokens,
    }
    for key in ("kv_handoff", "tpot", "e2e"):
        summary[f"{key}_p50_seconds"], summary[f"{key}_p95_seconds"] = percentiles(
            f"{key}_seconds"
        )
    return summary


def _decode_instances(args):
    world = dist.get_world_size()
    if args.loongserve_elastic_decode:
        # LoongServe: every elastic instance also decodes, one TP1 replica each.
        return tuple((rank,) for rank in range(world))
    if not args.decode_gpus:
        return ()
    return decode_instances(world - args.decode_gpus, world, args.decode_tp)
