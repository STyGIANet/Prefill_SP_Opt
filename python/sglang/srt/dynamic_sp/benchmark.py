# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Matched SP benchmark matrix, same-process warmup selection and CSV reporting."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from sglang.srt.dynamic_sp.metrics import throughput_summary

DATASETS = {
    "pg19": ("emozilla/pg19", "default", "text"),
    "arxiv": ("ccdv/arxiv-summarization", "document", "article"),
    "govreport": ("ccdv/govreport-summarization", "document", "report"),
}
FIELDS = [
    "dataset",
    "split",
    "model",
    "dtype",
    "offered_load_tokens_per_second",
    "arrival_pattern",
    "arrival_seed",
    "workload_sha256",
    "scheduler",
    "attention_backend",
    "kv_transport",
    "effective_attention_transport",
    "repeat",
    "status",
    "completed_prompts",
    "input_tokens",
    "excluded_context_records",
    "window_seconds",
    "throughput_tokens_per_second",
    "ttft_p50_seconds",
    "ttft_p95_seconds",
    "warmup_requests",
    "allocator_retries",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "verification_enabled",
    "completion_boundary",
    "tetris_profile",
    "loongserve_profile",
    "profile_attention_backend",
    "profile_kv_transport",
    "result_path",
    "error",
]


def warmup_arrivals(arrivals, count, token_budget=8192):
    """Deterministic length quantiles, including both extremes, with fresh IDs."""
    ordered = sorted(arrivals, key=lambda item: len(item[1].token_ids))
    count = min(count, len(ordered))
    if count <= 0:
        return []
    if token_budget <= 0:
        raise ValueError("Warmup token budget must be positive")
    count = min(count, token_budget)
    indices = [round(i * (len(ordered) - 1) / max(count - 1, 1)) for i in range(count)]
    per_prompt = token_budget // count
    return [
        (
            0.0,
            replace(
                ordered[j][1],
                request_id=f"warmup-{i}",
                token_ids=ordered[j][1].token_ids[:per_prompt],
            ),
        )
        for i, j in enumerate(indices)
    ]


def measured_metrics(result, workload):
    if result.get("summary", {}).get("allocator_retries", 0) or result.get(
        "summary", {}
    ).get("allocator_ooms", 0):
        raise ValueError("Allocator retries/OOM invalidate a measured trial")
    records = result["request_timings"]
    expected = {r["request_id"]: r for r in workload}
    if len(records) != len(expected) or {r["request_id"] for r in records} != set(
        expected
    ):
        raise ValueError(
            "Measured completions do not match the entire prepared workload"
        )
    latencies = []
    for r in records:
        original = expected[r["request_id"]]
        if r["input_tokens"] != len(original["token_ids"]) or not math.isclose(
            r["arrival_time_seconds"],
            original.get("arrival_time_seconds", 0),
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Measured tokens or arrivals differ from prepared workload"
            )
        latency = r["completion_time_seconds"] - r["arrival_time_seconds"]
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Invalid TTFT")
        latencies.append(latency)
    wall = throughput_summary(records)
    latencies.sort()
    return {
        "completed_prompts": len(records),
        "input_tokens": wall["completed_input_tokens"],
        "window_seconds": wall["prefill_wall_seconds"],
        "throughput_tokens_per_second": wall["useful_tokens_per_second"],
        "ttft_p50_seconds": statistics.median(latencies),
        "ttft_p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "completion_boundary": result["summary"]["completion_boundary"],
        "warmup_requests": result["summary"].get("warmup_requests", 0),
        "allocator_retries": result.get("summary", {}).get("allocator_retries", 0),
        "peak_allocated_bytes": result["summary"].get("peak_allocated_bytes", 0),
        "peak_reserved_bytes": result["summary"].get("peak_reserved_bytes", 0),
        "verification_enabled": result["summary"]["verification_enabled"],
    }


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--datasets")
    parser.add_argument("--dataset", "--eval-dataset", dest="dataset")
    parser.add_argument("--dataset-config", "--eval-dataset-config", default="default")
    parser.add_argument("--text-column", "--eval-text-column", default="text")
    parser.add_argument("--split", "--eval-split", default="test")
    parser.add_argument("--scheduler", default="all")
    parser.add_argument("--attention-backends")
    parser.add_argument("--attention-backend")
    parser.add_argument("--kv-transports")
    parser.add_argument("--kv-transport")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument("--workload-dir")
    parser.add_argument("--recalibrate-loongserve", action="store_true")
    parser.add_argument("--loongserve-profile")
    parser.add_argument("--loongserve-samples")
    parser.add_argument("--recalibrate-tetris", action="store_true")
    parser.add_argument("--tetris-profile")
    parser.add_argument("--tetris-samples")
    parser.add_argument("--fit-dataset")
    parser.add_argument("--fit-dataset-config", default="default")
    parser.add_argument("--fit-text-column", default="text")
    parser.add_argument("--fit-split", default="train")
    args, common = parser.parse_known_args()
    if args.recalibrate_loongserve and (
        args.loongserve_profile or args.loongserve_samples
    ):
        parser.error(
            "Choose recalibration or an existing LoongServe profile/samples, not both"
        )
    if args.recalibrate_tetris and (args.tetris_profile or args.tetris_samples):
        parser.error(
            "Choose Tetris recalibration or an existing profile/samples, not both"
        )
    if "--" in common:
        parser.error(
            "Matrix mode accepts named launcher options, not raw runner overrides"
        )
    schedulers = (
        ["greedy", "tetris", "loongserve"]
        if args.scheduler == "all"
        else [args.scheduler]
    )
    if "sglang-dp" in schedulers:
        parser.error("This benchmark compares SP schedulers only")
    backends = (
        args.attention_backends or args.attention_backend or "ring gathered streaming"
    ).split()
    transports = (
        args.kv_transports or args.kv_transport or "targeted all_gather"
    ).split()
    if not set(backends) <= {"ring", "gathered", "streaming"} or not set(
        transports
    ) <= {"targeted", "all_gather"}:
        parser.error("Invalid backend/transport")
    if args.repeats < 1 or not backends or not transports:
        parser.error("Repeats and matrix dimensions must be positive")
    if args.dataset and args.datasets:
        parser.error("Choose --dataset or --datasets")
    names = args.datasets.split() if args.datasets else list(DATASETS)
    if args.dataset:
        datasets = [("custom", args.dataset, args.dataset_config, args.text_column)]
    else:
        if not set(names) <= DATASETS.keys():
            parser.error("Dataset presets: pg19 arxiv govreport")
        datasets = [(n, *DATASETS[n]) for n in names]
    if args.workload_dir and len(datasets) != 1:
        parser.error("--workload-dir requires selecting one dataset")
    launcher = str(Path(args.launcher).resolve())
    root = Path(launcher).parent
    work = root / "python/sglang/srt/dynamic_sp/work"
    work.mkdir(parents=True, exist_ok=True)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="benchmark-", dir=work))
    )
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Choose a fresh output directory")
    rows, commands = [], []
    frozen_profile = (
        Path(args.tetris_profile).resolve() if args.tetris_profile else None
    )

    loongserve_profiles = {}

    def run(flags, log):
        command = ["bash", launcher, *common, *flags]
        commands.append(command)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with log.open("w") as stream:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            try:
                for line in proc.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                return proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise

    combinations = list(itertools.product(schedulers, backends, transports))
    for name, dataset, config, column in datasets:
        directory = output / name
        directory.mkdir()
        prepared = directory / "prepared"
        flags = [
            "--scheduler",
            "greedy",
            "--prepare-only",
            "--dataset",
            dataset,
            "--dataset-config",
            config,
            "--text-column",
            column,
            "--split",
            args.split,
            "--output-dir",
            str(prepared),
        ]
        if args.workload_dir:
            flags += ["--workload-dir", args.workload_dir]
        code = run(flags, directory / "prepare.log")
        manifest = (
            json.loads((prepared / "manifest.json").read_text()) if code == 0 else {}
        )
        workload = (
            json.loads((prepared / "workload.json").read_text()) if code == 0 else []
        )
        digest = (
            hashlib.sha256((prepared / "workload.json").read_bytes()).hexdigest()
            if code == 0
            else ""
        )
        for trial in range(1, args.repeats + 1):
            # Rotate run order to avoid always favoring the first configuration.
            offset = (trial - 1) % len(combinations)
            for scheduler, backend, transport in (
                combinations[offset:] + combinations[:offset]
            ):
                target = (
                    directory
                    / f"{scheduler.replace(':', '_').replace('/', '_')}-{backend}-{transport}-{trial}"
                )
                target.mkdir()
                row = {
                    "dataset": dataset,
                    "split": args.split,
                    "model": manifest.get("model", ""),
                    "offered_load_tokens_per_second": manifest.get(
                        "offered_load_input_tokens_per_second", ""
                    ),
                    "arrival_pattern": manifest.get("arrival_pattern", ""),
                    "arrival_seed": manifest.get("arrival_seed", ""),
                    "workload_sha256": digest,
                    "scheduler": scheduler,
                    "attention_backend": backend,
                    "kv_transport": transport,
                    "effective_attention_transport": (
                        "neighbor_p2p"
                        if backend == "ring"
                        else "broadcast"
                        if backend == "streaming"
                        else "alltoallv"
                        if transport == "targeted"
                        else "all_gather"
                    ),
                    "repeat": trial,
                    "status": "failed",
                    "excluded_context_records": manifest.get(
                        "excluded_context_records", ""
                    ),
                    "result_path": str(target / "results.json"),
                }
                try:
                    if code:
                        raise RuntimeError(
                            f"Dataset preparation failed; see {directory / 'prepare.log'}"
                        )
                    flags = [
                        "--scheduler",
                        scheduler,
                        "--attention-backend",
                        backend,
                        "--kv-transport",
                        transport,
                        "--workload-dir",
                        str(prepared),
                        "--output-dir",
                        str(target),
                    ]
                    if scheduler == "loongserve":
                        profile = (
                            Path(args.loongserve_profile).resolve()
                            if args.loongserve_profile
                            else loongserve_profiles.get((backend, transport))
                        )
                        if profile:
                            flags += ["--loongserve-profile", str(profile)]
                        elif args.loongserve_samples:
                            flags += ["--loongserve-samples", args.loongserve_samples]
                        elif args.recalibrate_loongserve:
                            flags += ["--recalibrate-loongserve"]
                    if scheduler == "tetris":
                        profile = frozen_profile
                        flags += ["--allow-profile-backend-mismatch"]
                        if profile:
                            flags += ["--tetris-profile", str(profile)]
                        else:
                            flags += [
                                "--fit-dataset",
                                args.fit_dataset or "emozilla/pg19",
                                "--fit-dataset-config",
                                args.fit_dataset_config,
                                "--fit-text-column",
                                args.fit_text_column,
                                "--fit-split",
                                args.fit_split,
                            ]
                            if args.tetris_samples:
                                flags += ["--tetris-samples", args.tetris_samples]
                            elif args.recalibrate_tetris:
                                flags += ["--recalibrate-tetris"]
                    print(
                        f"[dynamic-sp] benchmark dataset={name} scheduler={scheduler} backend={backend} transport={transport} repeat={trial}",
                        flush=True,
                    )
                    status = run(flags, target / "benchmark.log")
                    if scheduler == "loongserve":
                        fitted_loongserve = target / "loongserve/profile.json"
                        if fitted_loongserve.exists():
                            loongserve_profiles[(backend, transport)] = (
                                fitted_loongserve
                            )
                        selected_profile = (
                            Path(args.loongserve_profile).resolve()
                            if args.loongserve_profile
                            else loongserve_profiles.get((backend, transport))
                        )
                        if selected_profile:
                            calibration = json.loads(selected_profile.read_text())[
                                "metadata"
                            ]
                            row.update(
                                loongserve_profile=str(selected_profile),
                                profile_attention_backend=calibration.get(
                                    "attention_backend"
                                ),
                                profile_kv_transport=calibration.get("kv_transport"),
                            )
                    fitted = target / "tetris/profile.json"
                    if fitted.exists():
                        frozen_profile = fitted
                    if scheduler == "tetris" and frozen_profile is not None:
                        calibration = json.loads(frozen_profile.read_text())["metadata"]
                        row.update(
                            tetris_profile=str(frozen_profile),
                            profile_attention_backend=calibration.get(
                                "attention_backend"
                            ),
                            profile_kv_transport=calibration.get("kv_transport"),
                        )
                    if status:
                        raise RuntimeError(
                            f"Trial exited {status}; see {target / 'benchmark.log'}"
                        )
                    result = json.loads((target / "results.json").read_text())
                    row.update(
                        measured_metrics(result, workload),
                        status="ok",
                        dtype=result.get("dtype", ""),
                    )
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    row["error"] = str(exc)
                rows.append(row)
                write_csv(output / "metrics.csv", rows)
                if row["status"] == "ok":
                    print(
                        f"[dynamic-sp] result {name}/{scheduler}/{backend}/{transport} repeat={trial} throughput={row['throughput_tokens_per_second']:.1f} tokens/s TTFT_p50={row['ttft_p50_seconds']:.3f}s TTFT_p95={row['ttft_p95_seconds']:.3f}s",
                        flush=True,
                    )
                else:
                    print(f"[dynamic-sp] FAILED: {row['error']}", flush=True)
    print(f"[dynamic-sp] Benchmark CSV: {output / 'metrics.csv'}", flush=True)
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
