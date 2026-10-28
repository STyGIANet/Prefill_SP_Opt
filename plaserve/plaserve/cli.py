"""Dependency-free CLI; torch and transformers are imported only for GPU runs."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .model import LatencyModel, SimConfig
from .scheduler import POLICIES
from .simulator import simulate
from .workload import generate, load_trace, save_trace


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path: str | None, cls, **overrides):
    raw = json.loads(Path(path).read_text()) if path else {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    if cls is SimConfig:
        for key in ("graph_lengths", "graph_depths"):
            if key in raw:
                raw[key] = tuple(raw[key])
    return cls(**raw)


def summary(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        m = report["metrics"]
        rows.append({"policy": report["policy"], "execution": report["execution"],
                     "requests": m["requests"], "rps": m["rps"],
                     "mean_prefill_ms": m["mean_prefill_ms"],
                     "p90_prefill_ms": m["p90_prefill_ms"],
                     "p99_prefill_ms": m["p99_prefill_ms"],
                     "slo_violation_rate": m["slo_violation_rate"],
                     "graph_batch_fraction": m["graph_batch_fraction"],
                     "padding_fraction": m["padding_fraction"],
                     "migrations": sum(e["event"] == "applied" for e in report["migrations"])})
    return rows


def save_reports(reports: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for report in reports:
        write_json(directory / f"{report['policy']}.json", report)
    rows = summary(reports)
    write_json(directory / "summary.json", rows)
    with (directory / "summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_workload(parser, gpu=False):
    parser.add_argument("--trace", help="JSONL request counts and arrivals")
    parser.add_argument("--requests", type=int, default=100 if gpu else 1000)
    parser.add_argument("--rate", type=float, default=100 if gpu else 500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--short-fraction", type=float, default=0.8)
    parser.add_argument("--slo", type=float, default=0.4)
    parser.add_argument("--offline", action="store_true", help="omit request deadlines")
    parser.add_argument("--no-history", action="store_true", help="first-turn prefills only")


def get_workload(args, gpu=False):
    if args.trace:
        return load_trace(args.trace)
    return generate(args.requests, args.rate, args.seed, args.short_fraction,
                    None if args.offline else args.slo, not (gpu or args.no_history))


def parser():
    root = argparse.ArgumentParser(description="PLAserve: CPU simulation and cluster GPU reproduction")
    commands = root.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("trace", help="create a seeded synthetic JSONL workload")
    add_workload(trace)
    trace.add_argument("--output", default="results/trace.jsonl")
    for name in ("simulate", "compare"):
        sub = commands.add_parser(name, help="simulate one policy" if name == "simulate" else "compare all four ablations")
        add_workload(sub)
        sub.add_argument("--workers", type=int)
        sub.add_argument("--config", help="SimConfig JSON overrides")
        sub.add_argument("--latency-model", help="LatencyModel JSON coefficients in seconds")
        sub.add_argument("--output", default=f"results/{name}")
        if name == "simulate":
            sub.add_argument("--policy", choices=POLICIES, default="full")
    gpu = commands.add_parser("gpu", help="single-GPU Qwen prefill experiment (optional dependencies)")
    add_workload(gpu, gpu=True)
    gpu.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    gpu.add_argument("--revision", default="main", help="model revision; resolved commit is recorded")
    gpu.add_argument("--policy", choices=(*POLICIES, "all"), default="all")
    gpu.add_argument("--config", help="override the conservative single-GPU configuration")
    gpu.add_argument("--output", default="results/single_gpu")
    gpu.add_argument("--device", type=int, default=0)
    gpu.add_argument("--max-graphs", type=int, default=18)
    gpu.add_argument("--graph-memory-mb", type=int, default=2048)
    gpu.add_argument("--require-graphs", action="store_true", help="fail rather than fall back after capture failure")
    gpu.add_argument("--dry-run", action="store_true", help="print the plan without downloading or importing GPU packages")
    cluster = commands.add_parser("cluster", help="run the authors' SGLang benchmark on a GPU cluster node")
    cluster.add_argument("--profile", choices=("h200", "h100-4", "a100-2"), default="h200")
    cluster.add_argument("--models", nargs="+", help="model sizes (default: 7b; A100 profile runs its full suite)")
    cluster.add_argument("--upstream", default="vendor/LAPS")
    cluster.add_argument("--output", default="results/cluster")
    cluster.add_argument("--dry-run", action="store_true", help="show commands without needing the runtime or GPUs")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "trace":
            workload = get_workload(args)
            save_trace(workload, args.output)
            print(f"Saved {len(workload)} requests to {args.output}")
            return 0
        if args.command == "cluster":
            from .cluster import run_cluster
            run_cluster(args)
            return 0
        if args.command == "gpu":
            from .gpu import run_gpu
            reports = run_gpu(args)
            if reports is None:
                return 0
        else:
            config = load_config(args.config, SimConfig, workers=args.workers)
            model = load_config(args.latency_model, LatencyModel)
            workload = get_workload(args)
            policies = POLICIES if args.command == "compare" else (args.policy,)
            reports = [simulate(workload, config, model, policy) for policy in policies]
            save_trace(workload, Path(args.output) / "trace.jsonl")
        save_reports(reports, Path(args.output))
        print(json.dumps(summary(reports), indent=2, allow_nan=False))
        print(f"Saved raw requests, batches, and summary to {args.output}")
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
