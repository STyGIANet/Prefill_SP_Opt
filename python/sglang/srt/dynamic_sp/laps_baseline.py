# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Released LAPS scheduling on this checkout's native SGLang Engine replicas."""

import argparse
import json
import math
import os
import random
from pathlib import Path

from sglang.srt.dynamic_sp.dp_baseline import (
    benchmark,
    prepare_runtime_workload,
    read_workload,
)
from sglang.srt.dynamic_sp.laps_released.routing import MiniLoadBalancerPolicy

UPSTREAM = (
    "https://github.com/Jianshu-She/LAPS/tree/aa29962d501236ecd2a0c7a8470c5977df3fe057"
)
DEFAULTS = {
    "tp_size": 1,
    "length_threshold": 256,
    "rebalance_interval_seconds": 5.0,
    "rebalance_ratio": 2.0,
    "routing_seed": 0,
    "attention_backend": "flashinfer",
    "mem_fraction_static": 0.80,
    "graph_token_buckets": None,
    "piecewise_compiler": "eager",
    "batch_prefill_batch_sizes": [1, 2, 4, 8],
    "batch_prefill_seq_lengths": [16, 32, 64, 128, 256],
}


def read_config(path):
    supplied = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(supplied, dict) or set(supplied) - DEFAULTS.keys():
        raise ValueError(
            "Unknown LAPS config fields; use the native laps_native.json schema"
        )
    config = DEFAULTS | supplied
    for key in ("tp_size", "length_threshold"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if 8 % config["tp_size"]:
        raise ValueError("tp_size must divide eight GPUs")
    for key in ("rebalance_interval_seconds", "rebalance_ratio", "mem_fraction_static"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["rebalance_ratio"] <= 1 or config["mem_fraction_static"] >= 1:
        raise ValueError("Require rebalance_ratio > 1 and mem_fraction_static < 1")
    for key in (
        "graph_token_buckets",
        "batch_prefill_batch_sizes",
        "batch_prefill_seq_lengths",
    ):
        buckets = config[key]
        if key == "graph_token_buckets" and buckets is None:
            continue
        if (
            not isinstance(buckets, list)
            or not buckets
            or any(type(x) is not int or x <= 0 for x in buckets)
            or sorted(set(buckets)) != buckets
        ):
            raise ValueError(f"{key} must be increasing positive integers")
    if config["piecewise_compiler"] not in ("eager", "inductor"):
        raise ValueError("piecewise_compiler must be eager or inductor")
    if config["attention_backend"] != "flashinfer":
        raise ValueError("Released LAPS uses FlashInfer attention")
    return config


class ReleasedLAPSRouter(MiniLoadBalancerPolicy):
    """Released MiniLB methods with local prefill replica addresses."""

    def __init__(self, replicas, config):
        self.prefill_urls = list(range(replicas))
        self.prefill_bootstrap_ports = [None] * replicas
        # Preserve the released second random draw without a decode worker.
        self.decode_urls = [None]
        mid = replicas // 2
        self._laps_short_group = list(range(mid))
        self._laps_long_group = list(range(mid, replicas))
        self._laps_short_pending = self._laps_long_pending = 0
        self._laps_last_rebalance = 0.0
        self._laps_threshold = config["length_threshold"]
        self._laps_rebalance_interval = config["rebalance_interval_seconds"]
        self._laps_rebalance_ratio = config["rebalance_ratio"]
        random.seed(config["routing_seed"])
        self.assignments = {}
        self.moves = []

    @property
    def groups(self):
        return {
            "short": list(self._laps_short_group),
            "long": list(self._laps_long_group),
        }

    def select(self, rid, input_ids):
        previous = set(self._laps_short_group)
        rank, _, _, category = self.select_pair_laps({"input_ids": input_ids})
        current = set(self._laps_short_group)
        for moved in previous ^ current:
            self.moves.append(
                {
                    "rank": moved,
                    "to": "short" if moved in current else "long",
                    "before_request": rid,
                }
            )
        self.assignments[rid] = rank
        return rank, category

    def complete(self, category):
        self.laps_request_done(category)


class RoutedEngine:
    def __init__(self, engine, router):
        self.engine, self.router = engine, router

    async def async_generate(self, *, input_ids, rid, **kwargs):
        rank, category = self.router.select(rid, input_ids)
        try:
            return await self.engine.async_generate(
                input_ids=input_ids, rid=rid, routed_dp_rank=rank, **kwargs
            )
        finally:
            self.router.complete(category)


async def warmup(engine, requests, count, token_budget, replicas):
    if not count:
        return 0
    # Cover every replica, including replicas not currently receiving this class.
    count = max(count, replicas)
    if token_budget < count:
        raise ValueError(
            "warmup-tokens must cover at least one token per warmup request"
        )
    ordered = sorted(requests, key=lambda r: len(r["token_ids"]))
    for i in range(count):
        request = ordered[round(i * (len(ordered) - 1) / max(count - 1, 1))]
        result = await engine.async_generate(
            input_ids=request["token_ids"][: token_budget // count],
            sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
            rid=f"laps-warmup-{i}",
            routed_dp_rank=i % replicas,
        )
        if result.get("meta_info", {}).get("completion_tokens") != 1:
            raise RuntimeError("Native LAPS warmup did not produce one token")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--laps-config")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-local-tokens", type=int, default=600000)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=8192)
    parser.add_argument("--progress-interval", type=float, default=10)
    parser.add_argument("--progress-throughput-target", type=float, default=0)
    args = parser.parse_args()
    config = read_config(args.laps_config)
    if (
        min(
            args.chunk_size,
            args.max_local_tokens,
            args.max_running_requests,
            args.max_batch_tokens,
            args.warmup_tokens,
        )
        <= 0
        or args.warmup_requests < 0
    ):
        parser.error(
            "Token/request limits must be positive; warmup count must be nonnegative"
        )
    if any(
        not math.isfinite(v) or v < 0
        for v in (args.progress_interval, args.progress_throughput_target)
    ):
        parser.error("Progress interval and load must be finite and nonnegative")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Choose a fresh output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    requests = read_workload(args.workload)
    if any(r.get("prefix_request_id") or r.get("retain_kv") for r in requests):
        parser.error(
            "LAPS native benchmark accepts independent complete prompts, not custom retained-prefix chains"
        )
    # Prefer the installed toolkit matching PyTorch for native JIT kernels.
    # An explicit CUDA_HOME remains authoritative; no packages are installed.
    import torch

    if not os.environ.get("CUDA_HOME") and torch.version.cuda:
        toolkit = Path("/usr/local") / f"cuda-{torch.version.cuda}"
        if (toolkit / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = str(toolkit)
            os.environ["PATH"] = (
                str(toolkit / "bin") + os.pathsep + os.environ.get("PATH", "")
            )
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig

    if config["graph_token_buckets"] is None:
        # Released ServerArgs._generate_piecewise_cuda_graph_tokens, with
        # the ordinary GQA ceiling inherited from chunked_prefill_size.
        sizes = (
            list(range(4, 33, 4))
            + list(range(48, 257, 16))
            + list(range(288, 513, 32))
            + list(range(576, 1025, 64))
            + list(range(1280, 4097, 256))
            + list(range(4608, args.chunk_size + 1, 512))
        )
        config["graph_token_buckets"] = [n for n in sizes if n <= args.chunk_size]
        if not config["graph_token_buckets"]:
            raise ValueError("Released piecewise graph grid requires chunk-size >= 4")
    replicas = 8 // config["tp_size"]
    settings = {
        "model_path": args.model_path,
        "dp_size": replicas,
        "tp_size": config["tp_size"],
        "pp_size": 1,
        "dtype": args.dtype,
        "load_balance_method": "round_robin",
        "schedule_policy": "fcfs",
        "enable_laps_scheduler": True,
        "laps_length_threshold": config["length_threshold"],
        "laps_graph_config": json.dumps(config),
        "chunked_prefill_size": args.chunk_size,
        "max_total_tokens": args.max_local_tokens,
        "max_running_requests": args.max_running_requests,
        "max_prefill_tokens": args.max_batch_tokens,
        "mem_fraction_static": config["mem_fraction_static"],
        "attention_backend": config["attention_backend"],
        "disable_radix_cache": True,
        "skip_tokenizer_init": True,
        "log_level": "warning",
    }
    if max(config["batch_prefill_batch_sizes"]) > args.max_running_requests:
        raise ValueError(
            "max-running-requests must cover the released batch graph grid"
        )
    graphs = CudaGraphConfig(
        decode=PhaseConfig(backend="disabled"),
        prefill=PhaseConfig(backend="breakable", bs=config["graph_token_buckets"]),
    )
    from dataclasses import asdict

    (output.parent / "laps-settings.json").write_text(
        json.dumps(
            {
                "upstream": UPSTREAM,
                "engine": settings,
                "graphs": asdict(graphs),
                "router": config,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[dynamic-sp] scheduler=laps runtime=native replicas={replicas} tp={config['tp_size']} prefill_gpus=8 graphs=laps-released attention={config['attention_backend']}",
        flush=True,
    )
    engine = Engine(**settings, cuda_graph_config=graphs)
    try:
        requests, capacity = prepare_runtime_workload(
            engine, requests, args, baseline="LAPS"
        )
        count = engine.loop.run_until_complete(
            warmup(engine, requests, args.warmup_requests, args.warmup_tokens, replicas)
        )
        if not engine.flush_cache().success:
            raise RuntimeError("Native LAPS could not clear KV state after warmup")
        print("[dynamic-sp] warmup complete; starting measured arrivals", flush=True)
        router = ReleasedLAPSRouter(replicas, config)
        result = engine.loop.run_until_complete(
            benchmark(
                RoutedEngine(engine, router),
                requests,
                baseline="laps",
                progress_interval=args.progress_interval,
                progress_throughput_target=args.progress_throughput_target,
            )
        )
        result.update(
            scheduler="laps",
            dispatch_policy="released_laps_pending_ratio",
            worker_schedule_policy="released_laps_short_first",
            dtype=args.dtype,
            engine_settings=settings,
            graph_settings=asdict(graphs),
            upstream=UPSTREAM,
            router_settings=config,
            final_pools=router.groups,
            reassignments=router.moves,
            capacity_filter=capacity,
            workload_sha256=capacity["evaluated_workload_sha256"],
        )
        result["request_timings"] = result.pop("results")
        for record in result["request_timings"]:
            record["dp_rank"] = router.assignments[record["request_id"]]
        summary = result["summary"]
        summary.update(
            warmup_requests=count,
            verification_enabled=False,
            prefill_graph_replays=None,
            allocator_retries=None,
            allocator_ooms=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            allocator_statistics="Not collected by the native Engine benchmark",
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        # Single runs also produce a clean CSV with the same schema as sweeps.
        from sglang.srt.dynamic_sp.benchmark import measured_metrics, write_csv

        manifest_path = output.parent / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        row = dict(
            dataset=manifest.get("dataset", ""),
            split=manifest.get("split", ""),
            model=manifest.get("model", args.model_path),
            dtype=args.dtype,
            offered_load_tokens_per_second=args.progress_throughput_target,
            arrival_pattern=manifest.get("arrival_pattern", ""),
            arrival_seed=manifest.get("arrival_seed", ""),
            workload_sha256=capacity["evaluated_workload_sha256"],
            scheduler="laps",
            attention_backend=config["attention_backend"],
            kv_transport="none",
            effective_attention_transport="native_dp",
            repeat=1,
            status="ok",
            excluded_capacity_records=capacity["excluded_prompts"],
            excluded_context_records=manifest.get("excluded_context_records", 0),
            result_path=str(output),
            **measured_metrics(result, requests),
        )
        write_csv(output.parent / "metrics.csv", [row])
        print(
            f"[dynamic-sp] baseline=laps throughput_tokens/s={summary['useful_tokens_per_second']:.1f} TTFT_p50={summary['ttft_p50_seconds']:.3f}s TTFT_p95={summary['ttft_p95_seconds']:.3f}s prompts={len(requests)}; first token ready, excludes response IPC and HTTP",
            flush=True,
        )
        print(
            f"[dynamic-sp] Benchmark CSV: {output.parent / 'metrics.csv'}", flush=True
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
