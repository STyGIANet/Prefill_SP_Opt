# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""torchrun entry point for native SGLang dynamic SP prefill validation."""

import argparse
import importlib
import json
import math
import os
import socket
import statistics
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sglang.srt.dynamic_sp.arrivals import ArrivalSource
from sglang.srt.dynamic_sp.engine import DynamicSPPrefillEngine
from sglang.srt.dynamic_sp.kv_storage import NativeKVStorage
from sglang.srt.dynamic_sp.llama import load_native_llama
from sglang.srt.dynamic_sp.metrics import throughput_summary
from sglang.srt.dynamic_sp.plan import PrefillPlan, Prompt, PromptStep
from sglang.srt.dynamic_sp.scheduler import (
    GreedyScheduler,
    OnlineScheduler,
    SchedulerConfig,
)


@torch.inference_mode()
def reference_logits(model, ids):
    """Unsharded, unchunked causal SDPA reference using the same native weights."""
    ids = torch.tensor(ids, device=model.device, dtype=torch.long)
    positions = torch.arange(len(ids), device=model.device)
    x = model.embed(ids)
    for layer in range(model.num_layers):
        q, k, v = model.prepare(layer, x, positions)
        repeat = model.num_heads // model.num_kv_heads
        attention = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).repeat_interleave(repeat, 0).unsqueeze(0),
                v.transpose(0, 1).repeat_interleave(repeat, 0).unsqueeze(0),
                is_causal=True,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        x = model.complete(layer, x, attention)
    return model.logits(x[-1:]).flatten()


def default_schedule(world_size, vocab_size, prompt_length, chunk_size):
    """Exercise every degree and reordered memberships, not a scheduling policy."""
    prompts = [
        Prompt("a", tuple(i % vocab_size for i in range(prompt_length))),
        Prompt("b", tuple((3 * i + 1) % vocab_size for i in range(prompt_length))),
    ]
    plans = []
    degrees = list(range(1, world_size + 1)) + list(range(world_size, 0, -1))
    for index, end in enumerate(
        range(chunk_size, prompt_length + chunk_size, chunk_size)
    ):
        degree = degrees[index % len(degrees)]
        other = world_size + 1 - degree
        ranks = tuple((index + r) % world_size for r in range(degree))
        ranks_b = tuple((world_size - 1 - index - r) % world_size for r in range(other))
        plans.append(
            PrefillPlan(
                index,
                (
                    PromptStep("a", ranks, min(end, prompt_length)),
                    PromptStep("b", ranks_b, min(end, prompt_length)),
                ),
                tuple(prompts) if index == 0 else (),
            )
        )
    # Change ownership of complete KV without recomputing the prompt.
    for degree in degrees:
        ranks = tuple(reversed(range(degree)))
        plans.append(PrefillPlan(len(plans), (PromptStep("a", ranks, prompt_length),)))
    plans.append(PrefillPlan(len(plans), release=("a", "b")))
    return plans


def read_schedule(path):
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw["executed_plans"]
    if not isinstance(raw, list):
        raise TypeError("Schedule must be a JSON array of plans or recorded results")
    return [
        PrefillPlan(
            version=p.get("version", i),
            steps=tuple(
                PromptStep(s["request_id"], tuple(s["ranks"]), s["end"])
                for s in p.get("steps", [])
            ),
            admit=tuple(
                Prompt(a["request_id"], tuple(a["token_ids"]))
                for a in p.get("admit", [])
            ),
            release=tuple(p.get("release", [])),
        )
        for i, p in enumerate(raw)
    ]


# Tetris Implementation
def _parse_ints(text):
    return tuple(int(v) for v in text.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--execution-mode", choices=("async", "sync"), default="async")
    parser.add_argument("--async-timeout", type=float, default=600)
    parser.add_argument(
        "--prewarm-groups",
        choices=("none", "contiguous", "all"),
        default="none",
        help="Build these rank groups and connect their NCCL paths before measuring",
    )
    parser.add_argument(
        "--decode-gpus",
        type=int,
        default=0,
        help="Trailing workers that run TP1 decode replicas; the rest prefill",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=128,
        help="Output tokens per request, including the one sampled by prefill",
    )
    parser.add_argument("--decode-max-batch", type=int, default=128)
    parser.add_argument(
        "--decode-tp",
        type=int,
        default=0,
        help="TP degree of each decode instance (default: one instance over all decode GPUs)",
    )
    parser.add_argument(
        "--decode-kv-tokens",
        type=int,
        default=262144,
        help="KV slots of each decode instance; every TP rank holds its KV-head slice",
    )
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=8192)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Async progress refresh seconds; 0 disables progress",
    )
    parser.add_argument(
        "--progress-throughput-target",
        type=float,
        default=0.0,
        help="Throughput bar reference in input tokens/sec; 0 uses observed peak",
    )
    parser.add_argument("--allow-profile-backend-mismatch", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--schedule", help="Replay a recorded plan trace instead of online scheduling"
    )
    inputs.add_argument(
        "--workload",
        help="JSON requests with token_ids and optional arrival_time_seconds",
    )
    parser.add_argument(
        "--scheduler",
        default="greedy",
        help="greedy, request, dynamic, tetris, loongserve, or module:factory taking SchedulerConfig",
    )
    # Tetris Implementation
    parser.add_argument(
        "--dynamic-profile",
        help="Latency profile (Tetris profile JSON) for --scheduler dynamic predictions",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        help="Seconds per compute-bound step for --scheduler dynamic (default: from --dynamic-profile)",
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument(
        "--step-tokens",
        type=int,
        default=256,
        help="Tokens one GPU prefills in one compute-bound step",
    )
    parser.add_argument(
        "--long-steps",
        type=int,
        default=4,
        help="Prompts taking more steps than this run one step at a time and are preemptible",
    )
    parser.add_argument(
        "--resume-every-steps",
        type=int,
        default=2,
        help="A preempted long prompt runs at least once every k steps",
    )
    parser.add_argument(
        "--preempt-ratio",
        type=float,
        default=0.2,
        help="Total preemption of a long prompt, as a fraction of its own duration",
    )
    parser.add_argument(
        "--sp-sizes",
        type=_parse_ints,
        help="Comma-separated aligned block sizes for --scheduler request (default: powers of two)",
    )
    parser.add_argument(
        "--size-thresholds",
        type=_parse_ints,
        help="Comma-separated prompt lengths switching to the next --sp-sizes block",
    )
    parser.add_argument(
        "--tetris-profile",
        help="Fitted latency models and simulated load thresholds JSON",
    )
    # Tetris Implementation
    parser.add_argument(
        "--tetris-chunk-solver",
        choices=("newton", "bisection"),
        default="newton",
        help="Chunk length solver for Tetris Alg. 3 (paper: Newton's method)",
    )
    parser.add_argument(
        "--loongserve-profile", help="LoongServe batch latency SIB JSON"
    )
    parser.add_argument("--loongserve-max-prefill-seconds", type=float)
    parser.add_argument(
        "--loongserve-dp",
        choices=("monotone", "exact"),
        default="monotone",
        help="Sec. 5.3 DP: Eq. (6) split monotonicity (paper) or exhaustive splits",
    )
    parser.add_argument(
        "--loongserve-elastic-decode",
        action="store_true",
        help="Every worker also decodes; LoongServe preempts and places decoding (Sec. 5.1-5.4)",
    )
    parser.add_argument(
        "--loongserve-decode-batch-limit",
        type=int,
        default=256,
        help="Sec. 5.4 scale-up threshold: decode batch size where a step turns compute-bound "
        "(measured roofline ridge, about 300 tokens on RTX PRO 6000 Blackwell)",
    )
    parser.add_argument(
        "--loongserve-migration-bandwidth",
        type=float,
        default=25e9,
        help="Bytes/s pricing Eq. (4) KV migration; assumed PCIe 5.0 x16 peer bandwidth",
    )
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-local-tokens", type=int, default=8192)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--attention-backend", choices=("ring", "gathered", "streaming"), default="ring"
    )
    parser.add_argument(
        "--kv-transport", choices=("targeted", "all_gather"), default="targeted"
    )
    parser.add_argument("--shared-attention", choices=("auto", "sdpa"), default="auto")
    parser.add_argument(
        "--migration-overlap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--kv-block-size", type=int, default=256)
    parser.add_argument("--query-block-size", type=int, default=128)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--output", help="Save JSON validation/timing results on worker zero"
    )
    args = parser.parse_args()
    if any(
        not math.isfinite(v) or v < 0
        for v in (args.progress_interval, args.progress_throughput_target)
    ):
        parser.error(
            "Progress interval and throughput target must be finite and nonnegative"
        )
    if args.warmup_tokens <= 0:
        parser.error("warmup-tokens must be positive")
    if args.warmup_requests < 0:
        parser.error("warmup-requests must be nonnegative")
    if args.warmup_requests and (args.execution_mode != "async" or args.schedule):
        parser.error(
            "Warmup requires async online execution; use --warmup-requests 0 otherwise"
        )
    if args.prewarm_groups != "none" and args.execution_mode != "async":
        parser.error("Group prewarm requires async execution")
    if args.decode_gpus < 0 or args.decode_tokens < 2 or args.decode_max_batch <= 0:
        parser.error("Need decode-gpus >= 0, decode-tokens >= 2, decode-max-batch > 0")
    args.decode_tp = args.decode_tp or args.decode_gpus
    if args.decode_gpus and (
        args.decode_tp <= 0
        or args.decode_gpus % args.decode_tp
        or args.decode_kv_tokens <= 0
    ):
        parser.error(
            "decode-tp must divide decode-gpus; decode-kv-tokens must be positive"
        )
    if args.loongserve_elastic_decode and (
        args.scheduler != "loongserve" or args.decode_gpus
    ):
        parser.error(
            "--loongserve-elastic-decode needs --scheduler loongserve and no decode GPUs"
        )
    if (args.decode_gpus or args.loongserve_elastic_decode) and (
        args.execution_mode != "async" or args.schedule
    ):
        parser.error("Disaggregated decode requires async online execution")
    if args.chunk_size <= 0 or args.prompt_length <= 0:
        parser.error("prompt-length and chunk-size must be positive")
    if any(not math.isfinite(t) or t < 0 for t in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    if args.scheduler == "tetris" and not args.tetris_profile and not args.schedule:
        parser.error("--scheduler tetris requires --tetris-profile")
    if (
        args.scheduler == "dynamic"
        and not (args.dynamic_profile or args.step_seconds)
        and not args.schedule
    ):
        parser.error("--scheduler dynamic requires --dynamic-profile or --step-seconds")
    if args.scheduler == "loongserve":
        if not args.loongserve_profile and not args.schedule:
            parser.error("--scheduler loongserve requires --loongserve-profile")
        if args.execution_mode != "async":
            parser.error(
                "LoongServe requires async execution for independently completing batches"
            )
    if not math.isfinite(args.async_timeout) or args.async_timeout <= 0:
        parser.error("async-timeout must be finite and positive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group(
        "nccl" if args.device == "cuda" else "gloo", timeout=timedelta(minutes=10)
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))
    arrival_source = None
    try:
        model = load_native_llama(
            args.model_path, dtype=args.dtype, device=args.device, local_rank=local_rank
        )
        kv_pool = (
            NativeKVStorage(
                model,
                args.max_local_tokens,
                dist.get_world_size(),
                args.attention_backend,
                execution_tokens=args.max_local_tokens,
            )
            if args.execution_mode == "sync"
            else None
        )
        engine = DynamicSPPrefillEngine(
            model,
            control_group=control,
            kv_pool=kv_pool,
            max_local_tokens=args.max_local_tokens,
            kv_block_size=args.kv_block_size,
            query_block_size=args.query_block_size,
            attention_backend=args.attention_backend,
            logits_mode="all" if args.verify else "completed",
            kv_transport=args.kv_transport,
            shared_attention=args.shared_attention,
            migration_overlap=args.migration_overlap,
        )

        # Profiles describe the prefill workers only; decode replicas follow them.
        prefill_world = dist.get_world_size() - args.decode_gpus
        profile_gpu_names = (
            engine._all_objects(torch.cuda.get_device_name(local_rank))[:prefill_world]
            if args.scheduler in ("tetris", "loongserve") and args.device == "cuda"
            else None
        )

        profile_nodes = None
        if args.scheduler in ("tetris", "loongserve"):
            hosts = engine._all_objects(socket.gethostname())
            unique_hosts = list(dict.fromkeys(hosts))
            profile_nodes = [unique_hosts.index(host) for host in hosts]
            profile_nodes = profile_nodes[:prefill_world]

        def load_input():
            if dist.get_rank() != 0:
                return None
            if args.schedule:
                return read_schedule(args.schedule), None, []
            config = SchedulerConfig(
                prefill_world,
                args.max_local_tokens,
                args.chunk_size,
                args.max_running_requests,
                args.max_batch_tokens,
            )
            if args.scheduler == "greedy":
                policy = GreedyScheduler(config)
            elif args.scheduler == "request":
                from sglang.srt.dynamic_sp.dynamic_scheduler import RequestSpScheduler

                policy = RequestSpScheduler(config, args.sp_sizes, args.size_thresholds)
            elif args.scheduler == "dynamic":
                from sglang.srt.dynamic_sp.dynamic_scheduler import (
                    DynamicBatchScheduler,
                )

                options = {
                    "window_seconds": args.window_seconds,
                    "step_tokens": args.step_tokens,
                    "long_steps": args.long_steps,
                    "resume_every_steps": args.resume_every_steps,
                    "preempt_ratio": args.preempt_ratio,
                }
                if args.step_seconds:
                    policy = DynamicBatchScheduler(
                        config, step_seconds=args.step_seconds, **options
                    )
                else:
                    policy = DynamicBatchScheduler.from_profile(
                        config,
                        json.loads(Path(args.dynamic_profile).read_text()),
                        **options,
                    )
            elif args.scheduler in ("tetris", "loongserve"):
                if args.scheduler == "tetris":
                    from sglang.srt.dynamic_sp.tetris import TetrisScheduler

                    policy = TetrisScheduler.from_file(
                        config, args.tetris_profile, args.tetris_chunk_solver
                    )
                else:
                    from sglang.srt.dynamic_sp.loongserve import LoongServeScheduler

                    policy = LoongServeScheduler.from_file(
                        config,
                        args.loongserve_profile,
                        args.loongserve_max_prefill_seconds,
                        monotone=args.loongserve_dp == "monotone",
                    )
                if policy.profile["nodes"] != profile_nodes:
                    raise ValueError(
                        "Scheduler profile topology does not match the current rank-to-node mapping"
                    )
                metadata = policy.profile.get("metadata", {})
                for key, value in (
                    ("model_path", os.path.realpath(args.model_path)),
                    ("execution_mode", args.execution_mode),
                    ("gpu_names", profile_gpu_names),
                    ("world_size", config.world_size),
                    ("dtype", args.dtype),
                    ("attention_backend", args.attention_backend),
                    ("kv_transport", args.kv_transport),
                    ("shared_attention", args.shared_attention),
                    ("migration_overlap", args.migration_overlap),
                ):
                    if args.allow_profile_backend_mismatch and key in (
                        "attention_backend",
                        "kv_transport",
                    ):
                        continue
                    if metadata.get(key) != value:
                        raise ValueError(
                            f"Scheduler profile {key}={metadata.get(key)!r} does not match {value!r}"
                        )
            else:
                module, name = args.scheduler.rsplit(":", 1)
                policy = getattr(importlib.import_module(module), name)(config)
            raw = (
                json.loads(Path(args.workload).read_text())
                if args.workload
                else [
                    {
                        "request_id": rid,
                        "token_ids": [
                            (factor * i + 1) % model.vocab_size
                            for i in range(args.prompt_length)
                        ],
                    }
                    for rid, factor in (("a", 1), ("b", 3))
                ]
            )
            arrivals, seen = [], set()
            for row in raw:
                prompt = Prompt(row["request_id"], tuple(row["token_ids"]))
                arrival = float(row.get("arrival_time_seconds", 0.0))
                if not math.isfinite(arrival) or arrival < 0:
                    raise ValueError("Arrival times must be finite and nonnegative")
                if not prompt.request_id or prompt.request_id in seen:
                    raise ValueError("Workload request IDs must be unique and nonempty")
                if not prompt.token_ids or len(prompt.token_ids) > model.context_length:
                    raise ValueError("Prompt length is outside model context")
                if len(prompt.token_ids) > config.world_size * config.max_local_tokens:
                    raise ValueError("Prompt exceeds aggregate KV capacity")
                if any(
                    type(t) is not int or not 0 <= t < model.vocab_size
                    for t in prompt.token_ids
                ):
                    raise ValueError("Invalid token ID in workload")
                seen.add(prompt.request_id)
                arrivals.append((arrival, prompt))
            arrivals.sort(key=lambda item: item[0])
            return None, OnlineScheduler(policy), arrivals

        loaded = engine._checked("scheduler initialization", load_input)
        plans, scheduler, arrivals = loaded if loaded is not None else (None, None, [])
        if args.execution_mode == "async":
            from sglang.srt.dynamic_sp.async_runner import run_async

            run_async(args, model, control, scheduler, arrivals, plans)
            return
        index = 0
        arrival_records = []
        executed_plans = []
        results = []
        admitted_at = {}
        prompt_latencies = []
        pending_timings = {}
        request_timings = []
        if dist.get_rank() == 0:
            print(
                f"[dynamic-sp] workers={dist.get_world_size()} dtype={args.dtype} "
                f"backend={args.attention_backend} transport={args.kv_transport} "
                f"shared_attention={args.shared_attention} "
                f"scheduler={args.scheduler if not args.schedule else 'replay'} verify={args.verify}",
                flush=True,
            )
            print(
                "[dynamic-sp] step  new_tokens  seconds   tokens/s  KV_comm  migrations  checks  max_status_wait_s",
                flush=True,
            )
        benchmark_start = time.perf_counter()
        arrival_source = engine._checked(
            "arrival producer initialization",
            lambda: (
                ArrivalSource(arrivals, benchmark_start)
                if dist.get_rank() == 0 and scheduler is not None
                else None
            ),
        )
        while True:
            scheduling_started = time.perf_counter()

            def decide(index=index):
                if dist.get_rank() != 0:
                    return None
                if plans is not None:
                    return (
                        ("execute", plans[index])
                        if index < len(plans)
                        else ("stop", None)
                    )
                available = arrival_source.drain()
                now = time.perf_counter()
                for timestamp, prompt, producer_delay in available:
                    scheduler.submit(prompt, timestamp)
                    admitted_at[prompt.request_id] = timestamp
                    pending_timings[prompt.request_id] = {
                        "request_id": prompt.request_id,
                        "input_tokens": len(prompt.token_ids),
                        "arrival_time_seconds": timestamp - benchmark_start,
                        "submission_time_seconds": timestamp
                        + producer_delay
                        - benchmark_start,
                    }
                    arrival_records.append(
                        {
                            "request_id": prompt.request_id,
                            "arrival_time_seconds": timestamp - benchmark_start,
                            "producer_delay_seconds": producer_delay,
                            "scheduler_observation_delay_seconds": now - timestamp,
                        }
                    )
                plan = scheduler.next_plan(now)
                if plan is not None:
                    return "execute", plan
                if not arrival_source.exhausted:
                    return "wait", 0.01
                return "stop", None

            decision = engine._checked("online scheduling", decide)
            command = [decision]
            dist.broadcast_object_list(command, src=0, group=control)
            action, plan = command[0]
            if action == "stop":
                break
            if action == "wait":
                time.sleep(plan)
                continue
            scheduling_seconds = time.perf_counter() - scheduling_started
            if args.device == "cuda":
                torch.cuda.synchronize()
            new_tokens = sum(
                step.end
                - (
                    engine._states[step.request_id].end
                    if step.request_id in engine._states
                    else 0
                )
                for step in plan.steps
            )
            start = time.perf_counter()
            if dist.get_rank() == 0:
                for prompt in plan.admit:
                    admitted_at.setdefault(prompt.request_id, start)
                    pending_timings.setdefault(
                        prompt.request_id,
                        {
                            "request_id": prompt.request_id,
                            "input_tokens": len(prompt.token_ids),
                            "arrival_time_seconds": start - benchmark_start,
                            "submission_time_seconds": start - benchmark_start,
                        },
                    )
            logits = engine.execute(plan if dist.get_rank() == 0 else None)
            finished = time.perf_counter()
            completed = {}
            output_ids = {}
            if dist.get_rank() == 0:
                for rid in logits:
                    state = engine._states[rid]
                    if (
                        state.end == len(state.prompt.token_ids)
                        and rid in pending_timings
                    ):
                        # Greedy top-k=1, matching the native DP benchmark.
                        output_ids[rid] = [int(torch.argmax(logits[rid]).item())]
                        ready = time.perf_counter()
                        completed[rid] = ready - admitted_at.pop(rid)
                        timing = pending_timings.pop(rid)
                        timing["logits_ready_time_seconds"] = finished - benchmark_start
                        timing["completion_time_seconds"] = ready - benchmark_start
                        timing["output_ids"] = output_ids[rid]
                        request_timings.append(timing)
                prompt_latencies.extend(completed.values())

            elapsed = max(engine._all_objects(finished - start))
            worker_metrics = engine._all_objects(dict(engine.metrics))

            def observe(plan=plan, elapsed=elapsed, worker_metrics=worker_metrics):
                if dist.get_rank() == 0 and scheduler is not None:
                    scheduler.committed(
                        plan, engine.snapshot(), elapsed, worker_metrics
                    )

            engine._checked("scheduler commit observation", observe)
            if dist.get_rank() == 0:
                executed_plans.append(asdict(plan))
            index += 1

            def verify(logits=logits):
                if dist.get_rank() != 0:
                    return None
                errors = {}
                if args.verify:
                    for rid, actual in logits.items():
                        state = engine._states[rid]
                        expected = reference_logits(
                            model, state.prompt.token_ids[: state.end]
                        )
                        torch.testing.assert_close(
                            actual, expected, atol=args.atol, rtol=args.rtol
                        )
                        errors[rid] = float(
                            (actual.float() - expected.float()).abs().max()
                        )
                return errors

            errors = (
                engine._checked("reference validation", verify) if args.verify else {}
            )
            if dist.get_rank() == 0:
                record = {
                    "version": engine.version,
                    "seconds": elapsed,
                    "scheduling_seconds": scheduling_seconds,
                    "tetris_decision": scheduler.policy.last_decision
                    if scheduler is not None and args.scheduler in ("tetris", "dynamic")
                    else None,
                    "new_tokens": new_tokens,
                    "tokens_per_second": new_tokens / elapsed if new_tokens else None,
                    "prompt_ttft_seconds": completed,
                    "output_ids": output_ids,
                    "worker_metrics": worker_metrics,
                    "max_abs_logit_error": errors,
                    "verified": args.verify,
                    "requests": engine.snapshot(),
                }
                results.append(record)
                metrics = worker_metrics[0]
                rate = f"{new_tokens / elapsed:10.1f}" if new_tokens else "         -"
                wait = max(m["status_consensus_seconds"] for m in worker_metrics)
                print(
                    f"[dynamic-sp] {engine.version:4d} {new_tokens:11d} {elapsed:8.3f} {rate} "
                    f"{metrics['kv_collectives']:8d} {metrics['migration_collectives']:11d} "
                    f"{metrics['checked_phases']:7d} {wait:18.3f}",
                    flush=True,
                )
                for rid, latency in completed.items():
                    print(
                        f"[dynamic-sp] prompt={rid} TTFT={latency:.3f}s (arrival to first token ready, before response IPC)",
                        flush=True,
                    )
        summary = {}
        if dist.get_rank() == 0:
            last_prefill = max(
                (i for i, r in enumerate(results) if r["new_tokens"]), default=-1
            )
            prefill_seconds = sum(r["seconds"] for r in results[: last_prefill + 1])
            wall_metrics = throughput_summary(request_timings)
            wall_seconds = wall_metrics["prefill_wall_seconds"]
            tokens = sum(r["new_tokens"] for r in results)
            rate = tokens / prefill_seconds if prefill_seconds else 0.0
            summary = {
                "prefill_tokens": tokens,
                "prefill_seconds": prefill_seconds,
                "prefill_tokens_per_second": rate,
                "prefill_wall_seconds": wall_seconds,
                "wall_tokens_per_second": tokens / wall_seconds
                if wall_seconds
                else 0.0,
                **wall_metrics,
                "ttft_definition": "Scheduled arrival to first sampled token ready on CPU; excludes response IPC and HTTP",
                "completion_boundary": "first_token_ready_before_response_ipc",
                "ttft_p50_seconds": statistics.median(prompt_latencies)
                if prompt_latencies
                else None,
                "ttft_p95_seconds": sorted(prompt_latencies)[
                    math.ceil(0.95 * len(prompt_latencies)) - 1
                ]
                if prompt_latencies
                else None,
                "p95_method": "nearest rank",
            }
            print(
                f"[dynamic-sp] prefill_tokens={tokens} execution_seconds={prefill_seconds:.3f} "
                f"execution_tokens/s={rate:.1f} wall_tokens/s={summary['wall_tokens_per_second']:.1f} "
                f"window_start={summary['throughput_start_seconds']}s "
                f"window_end={summary['throughput_end_seconds']}s "
                f"window_seconds={wall_seconds:.6f} "
                f"throughput_tokens/s={summary['useful_tokens_per_second']:.1f}",
                flush=True,
            )
            if prompt_latencies:
                latencies = sorted(prompt_latencies)
                print(
                    f"[dynamic-sp] per-prompt TTFT: p50={statistics.median(latencies):.3f}s "
                    f"p95={latencies[math.ceil(0.95 * len(latencies)) - 1]:.3f}s "
                    f"prompts={len(latencies)}; excludes response IPC and HTTP",
                    flush=True,
                )
        if args.output:
            engine._checked(
                "result writing",
                lambda: (
                    Path(args.output).write_text(
                        json.dumps(
                            {
                                "execution_mode": "sync",
                                "world_size": dist.get_world_size(),
                                "model": args.model_path,
                                "dtype": args.dtype,
                                "scheduler": args.scheduler
                                if not args.schedule
                                else "replay",
                                "scheduler_settings": {
                                    "tetris": scheduler.policy.metadata()
                                    if args.scheduler in ("tetris", "dynamic")
                                    else None,
                                    "chunk_tokens": args.chunk_size,
                                    "max_running_requests": args.max_running_requests,
                                    "max_batch_tokens": args.max_batch_tokens,
                                    "max_local_tokens": args.max_local_tokens,
                                }
                                if not args.schedule
                                else None,
                                "arrival_records": arrival_records,
                                "request_timings": request_timings,
                                "executed_plans": executed_plans,
                                "results": results,
                                "summary": summary,
                            },
                            indent=2,
                        )
                    )
                    if dist.get_rank() == 0
                    else None
                ),
            )
    finally:
        if arrival_source is not None:
            arrival_source.close()
        if not getattr(args, "async_failed", False):
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
