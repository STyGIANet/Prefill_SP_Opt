# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Matched prefill experiments with frozen profiles and per-request measurements."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .benchmark import DATASETS as BASE_DATASETS, measured_metrics

DATASETS = {name: (*spec, "test", "text") for name, spec in BASE_DATASETS.items()}
DATASETS["longbench"] = (
    "zai-org/LongBench-v2",
    "default",
    "context",
    "train",
    "longbench-v2",
)

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "python/sglang/srt/dynamic_sp/work"
SCHEDULERS = {
    "laps": "laps",
    "tetris": "tetris",
    "loongserve": "loongserve",
    "ours": "dynamic",
    "spsrpt": "spsrpt",
    "sglang-dp": "sglang-dp",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def print_summary(rows):
    """Print tab-separated measurements; missing values remain explicit."""
    columns = [
        ("dataset", "dataset"),
        ("scheduler", "scheduler"),
        ("run", "repeat"),
        ("status", "status"),
        ("load_tok/s", "offered_load_tokens_per_second"),
        ("tokens/s", "throughput_tokens_per_second"),
        ("prompts", "ttft_all_count"),
        ("TTFT_mean_s", "ttft_all_mean_seconds"),
        ("TTFT_p50_s", "ttft_all_p50_seconds"),
        ("TTFT_p95_s", "ttft_all_p95_seconds"),
        ("slowdown_est", "slowdown_estimated_mean"),
        ("short_n", "ttft_short_count"),
        ("short_p50_s", "ttft_short_p50_seconds"),
        ("short_p95_s", "ttft_short_p95_seconds"),
        ("long_n", "ttft_long_count"),
        ("long_p50_s", "ttft_long_p50_seconds"),
        ("long_p95_s", "ttft_long_p95_seconds"),
    ]
    table = [[label for label, _ in columns]]
    for row in rows:
        values = [row.get(key) for _, key in columns]
        table.append(
            [
                "NA"
                if value is None
                else f"{value:.3f}"
                if isinstance(value, float)
                else str(value)
                for value in values
            ]
        )
    widths = [max(len(row[index]) for row in table) for index in range(len(columns))]
    print(
        "\n[experiments] Final summary (TTFT in seconds; slowdown_est uses a shared estimated SP1 reference)",
        flush=True,
    )
    for row in table:
        print(
            "\t".join(value.ljust(width) for value, width in zip(row, widths)),
            flush=True,
        )


def print_slowdown_tables(rows):
    """Compare schedulers within each matched workload and repetition."""
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in (
            "dataset", "model", "workload_sha256", "offered_load_tokens_per_second",
            "repeat", "short_threshold_tokens",
        ))
        groups.setdefault(key, []).append(row)
    labels = {"tetris": "Tetris", "laps": "LAPS", "spsrpt": "SPSRPT",
              "loongserve": "LoongServe", "ours": "Ours", "sglang-dp": "SGLang DP"}
    metrics = [
        ("Overall mean", "slowdown_estimated_mean"),
        ("Overall p50", "slowdown_estimated_p50"),
        ("Overall p95", "slowdown_estimated_p95"),
        ("Short mean", "slowdown_short_estimated_mean"),
        ("Short p95", "slowdown_short_estimated_p95"),
        ("Long mean", "slowdown_long_estimated_mean"),
        ("Long p95", "slowdown_long_estimated_p95"),
    ]
    for key, group in groups.items():
        group = sorted(group, key=lambda row: list(SCHEDULERS).index(row["scheduler"]))
        print(f"\n[experiments] Estimated slowdown: {key[0]}, run={key[4]}, "
              f"load={key[3]} tokens/sec, short <= {key[5]} tokens; shared estimated SP1 reference",
              flush=True)
        table = [["Slowdown", *[labels.get(r["scheduler"], r["scheduler"]) for r in group]]]
        for label, field in metrics:
            values = [r.get(field) if r.get("status") == "ok" else None for r in group]
            table.append([label, *[
                "NA" if v is None or not math.isfinite(v) else
                f"{v:.3f}"
                for v in values
            ]])
        widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
        for row in table:
            print(
                "\t".join(value.ljust(width) for value, width in zip(row, widths)),
                flush=True,
            )



def run(command, log):
    """Preserve exact commands and diagnostics, including preparation failures."""
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(item) for item in command]
    save_json(log.with_suffix(".command.json"), command)
    print(f"[experiments] {log.parent.name}: {log}", flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise subprocess.CalledProcessError(code, command)


def stats(values, prefix):
    values = sorted(values)
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean_seconds": statistics.mean(values) if values else None,
        f"{prefix}_p50_seconds": statistics.median(values) if values else None,
        f"{prefix}_p95_seconds": values[math.ceil(0.95 * len(values)) - 1]
        if values
        else None,
        f"{prefix}_p99_seconds": values[math.ceil(0.99 * len(values)) - 1]
        if values
        else None,
        f"{prefix}_max_seconds": max(values) if values else None,
    }


def count_or_all(value):
    if value == "all":
        return value
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a positive integer or all") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer or all")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--model", default="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=["pg19", "longbench"]
    )
    parser.add_argument("--split", help="Override each dataset's evaluation split")
    parser.add_argument(
        "--schedulers", nargs="+", choices=SCHEDULERS, default=list(SCHEDULERS)
    )
    parser.add_argument("--suite", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--num-prompts", type=count_or_all, help="Record limit or all; full suite defaults to all")
    parser.add_argument("--target-total-tokens", type=count_or_all, help="Aggregate token target or all for no target")
    parser.add_argument(
        "--eligible-max-tokens",
        type=int,
        help="Exclude larger prompts without shortening eligible records",
    )
    parser.add_argument("--spsrpt-profile", type=Path)
    parser.add_argument("--roofline-samples", type=Path)
    parser.add_argument("--spsrpt-quantum", type=float, default=0.2)
    parser.add_argument("--spsrpt-prediction-tolerance", type=float, default=0.01)
    parser.add_argument("--spsrpt-prediction-max-panels", type=int, default=128)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1048576,
        help="Truncate longer tokenized prompts to this length",
    )
    parser.add_argument("--max-source-rows", type=count_or_all, help="Source scan limit or all")
    parser.add_argument(
        "--short-threshold", type=int, default=1024,
        help="Short prompts have at most this many total input tokens (default: 1024)",
    )
    parser.add_argument("--offered-load", type=float, default=25000)
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--sp-kv-tokens", type=int, default=131072)
    parser.add_argument("--native-kv-tokens", type=int, default=600000)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=8192)
    parser.add_argument(
        "--attention-backend", default="ring", choices=("ring", "gathered", "streaming")
    )
    parser.add_argument(
        "--kv-transport", default="targeted", choices=("targeted", "all_gather")
    )
    parser.add_argument("--laps-config", type=Path)
    parser.add_argument("--tetris-profile", type=Path)
    parser.add_argument("--loongserve-profile", type=Path)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--fit-dataset", default="emozilla/pg19")
    parser.add_argument("--fit-dataset-config", default="default")
    parser.add_argument("--fit-text-column", default="text")
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--fit-num-prompts", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reuse-tetris-from", type=Path,
        help="Copy Tetris results, workloads, and frozen profiles from a prior experiment; rerun SPSRPT",
    )
    preliminary, _ = parser.parse_known_args()
    reused_configuration = None
    if preliminary.reuse_tetris_from:
        source = preliminary.reuse_tetris_from.resolve()
        reused_configuration = json.loads((source / "configuration.json").read_text())
        defaults = {}
        for action in parser._actions:
            if action.dest in reused_configuration and action.dest not in (
                "output_dir", "schedulers", "reuse_tetris_from", "recalibrate",
                "spsrpt_quantum",
            ):
                value = reused_configuration[action.dest]
                defaults[action.dest] = Path(value) if action.type is Path and value else value
        parser.set_defaults(**defaults, schedulers=["tetris", "spsrpt"])
    args = parser.parse_args()
    if reused_configuration is not None:
        args.reuse_tetris_from = args.reuse_tetris_from.resolve()
        if args.recalibrate:
            parser.error("--reuse-tetris-from cannot be combined with --recalibrate; reuse needs the saved profiles")
        if (len(set(args.schedulers)) != len(args.schedulers)
                or set(args.schedulers) not in ({"spsrpt"}, {"tetris", "spsrpt"})):
            parser.error("--reuse-tetris-from supports --schedulers spsrpt (Tetris is included automatically)")
        # Tetris participates in reporting but is never launched in reuse mode.
        args.schedulers = ["tetris", "spsrpt"]
        for key in (
            "model", "model_revision", "datasets", "split", "num_prompts",
            "target_total_tokens", "eligible_max_tokens", "max_prompt_tokens",
            "max_source_rows", "offered_load", "arrival_seed", "repeats",
            "chunk_tokens", "sp_kv_tokens", "warmup_requests", "warmup_tokens",
            "attention_backend", "kv_transport",
        ):
            if getattr(args, key) != reused_configuration[key]:
                parser.error(f"Cannot change {key} when reusing Tetris results")
        for label in ("tetris", "spsrpt"):
            frozen = args.reuse_tetris_from / "profiles" / f"{label}.json"
            if label == "spsrpt":
                # Historical sweeps used this filename for the roofline profile.
                candidates = (frozen, args.reuse_tetris_from / "profiles" / "spsrptroof.json")
                frozen = next((path for path in candidates if path.is_file()
                               and json.loads(path.read_text()).get("schema")
                               in ("sp-roofline-v1", "spsrptroof-v1")), frozen)
            if not frozen.is_file():
                parser.error(f"Missing frozen profile: {frozen}")
            supplied = getattr(args, f"{label}_profile")
            if supplied and digest(supplied) != digest(frozen):
                parser.error(f"Cannot replace the frozen {label} profile during reuse")
            setattr(args, f"{label}_profile", frozen)
        with (args.reuse_tetris_from / "summary.csv").open(newline="") as handle:
            saved_rows = list(csv.DictReader(handle))
        for name in args.datasets:
            directory = args.reuse_tetris_from / "workloads" / name
            manifest = json.loads((directory / "manifest.json").read_text())
            if digest(directory / "workload.json") != manifest["workload_sha256"]:
                parser.error(f"Saved workload changed: {directory}")
            for repeat in range(1, args.repeats + 1):
                result = args.reuse_tetris_from / "runs" / name / name / f"tetris-{repeat}" / "results.json"
                if not result.is_file():
                    parser.error(f"Missing saved Tetris result: {result}")
                matches = [row for row in saved_rows
                           if row["dataset"] == name and row["scheduler"] == "tetris"
                           and int(row["repeat"]) == repeat]
                if len(matches) != 1 or matches[0]["status"] != "ok":
                    parser.error(f"No successful Tetris trial for {name}, repeat {repeat}")
                if (matches[0]["workload_sha256"] != manifest["workload_sha256"]
                        or matches[0]["profile_sha256"] != digest(args.tetris_profile)):
                    parser.error(f"Saved Tetris workload/profile mismatch: {result}")
    if args.num_prompts is None:
        args.num_prompts = 24 if args.suite == "pilot" else "all"
    if args.target_total_tokens is None:
        args.target_total_tokens = 262144 if args.suite == "pilot" else "all"
    if args.max_source_rows is None:
        args.max_source_rows = 10000 if args.suite == "pilot" else "all"
    if args.eligible_max_tokens is None:
        args.eligible_max_tokens = (
            65536 if args.suite == "pilot" else args.max_prompt_tokens
        )
    if not math.isfinite(args.spsrpt_quantum) or args.spsrpt_quantum <= 0:
        parser.error("spsrpt-quantum must be finite and positive")
    for name in (
        "num_prompts",
        "target_total_tokens",
        "max_prompt_tokens",
        "eligible_max_tokens",
        "max_source_rows",
        "short_threshold",
        "repeats",
        "chunk_tokens",
        "sp_kv_tokens",
        "native_kv_tokens",
        "warmup_tokens",
        "fit_num_prompts",
    ):
        if getattr(args, name) != "all" and getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if (
        args.warmup_requests < 0
        or not math.isfinite(args.offered_load)
        or args.offered_load <= 0
    ):
        parser.error(
            "Warmup count must be nonnegative and offered loads finite and positive"
        )
    if len(set(args.datasets)) != len(args.datasets) or len(
        set(args.schedulers)
    ) != len(args.schedulers):
        parser.error("Datasets and schedulers must be unique")
    if args.recalibrate and (args.tetris_profile or args.loongserve_profile or args.spsrpt_profile):
        parser.error("Choose supplied profiles or --recalibrate")
    output = (
        args.output_dir or WORK / datetime.now().strftime("experiments-%Y%m%d-%H%M%S")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration.update(
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        scheduler_mapping=SCHEDULERS,
        short_definition="input_tokens <= short_threshold",
        selection="Source-order prefix, truncate at max_prompt_tokens, exclude above eligible_max_tokens, stop at count or token target",
        python=sys.executable,
    )
    save_json(output / "configuration.json", configuration)
    module = [sys.executable, "-m"]
    launcher = [
        "bash",
        ROOT / "launch_dynamic_sp_8gpu.sh",
        "--python",
        sys.executable,
        "--no-verify",
        "--dtype",
        "bfloat16",
        "--execution-mode",
        "async",
        "--chunk-tokens",
        args.chunk_tokens,
        "--warmup-requests",
        args.warmup_requests,
        "--warmup-tokens",
        args.warmup_tokens,
    ]
    sp = [
        "--attention-backend",
        args.attention_backend,
        "--kv-transport",
        args.kv_transport,
        "--max-local-tokens",
        args.sp_kv_tokens,
    ]
    workloads = []
    model = args.model
    # Tokenize each dataset once and share identical arrivals across policies.
    for name in args.datasets:
        dataset, config, column, default_split, prompt_format = DATASETS[name]
        directory = output / "workloads" / name
        if args.reuse_tetris_from:
            shutil.copytree(args.reuse_tetris_from / "workloads" / name, directory)
        else:
            run(
                module
                + [
                    "sglang.srt.dynamic_sp.prepare_workload",
                    "--model",
                    model,
                    "--model-revision",
                    args.model_revision,
                    "--dataset",
                    dataset,
                    "--dataset-config",
                    config,
                    "--text-column",
                    column,
                    "--split",
                    args.split or default_split,
                    "--num-prompts",
                    args.num_prompts,
                    "--input-tokens",
                    "all",
                    "--prompt-format",
                    prompt_format,
                    "--truncate-tokens",
                    args.max_prompt_tokens,
                    *([] if args.target_total_tokens == "all" else
                      ["--target-total-tokens", args.target_total_tokens]),
                    *([] if args.max_source_rows == "all" else
                      ["--max-source-rows", args.max_source_rows]),
                    "--max-prompt-tokens",
                    min(args.max_prompt_tokens, args.eligible_max_tokens),
                    "--offered-load",
                    args.offered_load,
                    "--arrival-seed",
                    args.arrival_seed,
                    "--chunk-tokens",
                    args.chunk_tokens,
                    "--output-dir",
                    directory,
                ],
                output / "preparation" / f"{name}.log",
            )
        manifest = json.loads((directory / "manifest.json").read_text())
        model = manifest["model_path"]
        base = json.loads((directory / "workload.json").read_text())
        lengths = sorted(len(record["token_ids"]) for record in base)
        short_count = sum(length <= args.short_threshold for length in lengths)
        print(
            f"[experiments] {name}: prompts={len(lengths)} tokens={sum(lengths)} "
            f"short={short_count} long={len(lengths) - short_count} "
            f"min_tokens={min(lengths)} max_tokens={max(lengths)} "
            f"excluded={len(manifest.get('excluded_rows', []))}",
            flush=True,
        )
        save_json(
            directory / "lengths.json",
            {
                "input_tokens": [len(record["token_ids"]) for record in base],
                "minimum": min(lengths),
                "median": statistics.median(lengths),
                "p95": lengths[math.ceil(0.95 * len(lengths)) - 1],
                "maximum": max(lengths),
                "total": sum(lengths),
            },
        )
        manifest["workload_sha256"] = digest(directory / "workload.json")
        save_json(directory / "manifest.json", manifest)
        workloads.append((name, args.offered_load, directory, base, manifest))
    # Reject known impossible shared workloads before loading any GPU model.
    # Native runtimes may expose a lower effective capacity at startup.
    for name, _, _, records, _ in workloads:
        longest = max(len(record["token_ids"]) for record in records)
        if any(label in args.schedulers for label in ("laps", "sglang-dp")) and longest + 1 > args.native_kv_tokens:
            raise ValueError(
                f"{name}: longest prompt has {longest} tokens, exceeding requested native "
                f"capacity {args.native_kv_tokens} including output. Set --eligible-max-tokens "
                f"below {args.native_kv_tokens}, or select only SP schedulers."
            )
        if any(label in args.schedulers for label in ("ours", "spsrpt", "tetris", "loongserve")) and longest > 8 * args.sp_kv_tokens:
            raise ValueError(
                f"{name}: {longest} tokens exceed the eight-worker SP KV capacity "
                f"{8 * args.sp_kv_tokens}; lower --eligible-max-tokens or raise --sp-kv-tokens"
            )
    profiles = {}
    if "spsrpt" in args.schedulers:
        if args.attention_backend != "ring" or args.kv_transport != "targeted":
            raise ValueError("spsrpt requires ring attention with targeted transport")
        frozen = output / "profiles" / "spsrpt.json"
        frozen.parent.mkdir(exist_ok=True)
        if args.spsrpt_profile:
            shutil.copyfile(args.spsrpt_profile, frozen)
        else:
            from .roofline_profile import prepare_profile

            prepare_profile(model, WORK, frozen, args.roofline_samples, args.recalibrate)
        profiles["spsrpt"] = {"path": str(frozen), "sha256": digest(frozen)}
    for algorithm in ("tetris", "loongserve"):
        if algorithm not in args.schedulers and not (
            algorithm == "tetris"
            and "ours" in args.schedulers
        ):
            continue
        frozen = output / "profiles" / f"{algorithm}.json"
        frozen.parent.mkdir(exist_ok=True)
        supplied = getattr(args, f"{algorithm}_profile")
        if supplied:
            shutil.copyfile(supplied, frozen)
        else:
            found = False
            if not args.recalibrate:
                try:
                    run(
                        module
                        + [
                            "sglang.srt.dynamic_sp.profile_cache",
                            "--algorithm",
                            algorithm,
                            "--work-dir",
                            WORK,
                            "--output",
                            frozen,
                            "--model-path",
                            model,
                            "--dtype",
                            "bfloat16",
                            "--execution-mode",
                            "async",
                            "--attention-backend",
                            args.attention_backend,
                            "--kv-transport",
                            args.kv_transport,
                            "--shared-attention",
                            "auto",
                            "--migration-overlap",
                        ],
                        output / "profiles" / f"{algorithm}-cache.log",
                    )
                    found = True
                except subprocess.CalledProcessError as error:
                    if error.returncode != 2:
                        raise
            if not found:
                directory = output / "calibration" / algorithm
                command = (
                    launcher
                    + sp
                    + [
                        "--scheduler",
                        algorithm,
                        "--profile-only",
                        f"--recalibrate-{algorithm}",
                        "--workload-dir",
                        workloads[0][2],
                        "--output-dir",
                        directory,
                    ]
                )
                if algorithm == "tetris":
                    command += [
                        "--fit-dataset",
                        args.fit_dataset,
                        "--fit-dataset-config",
                        args.fit_dataset_config,
                        "--fit-text-column",
                        args.fit_text_column,
                        "--fit-split",
                        args.fit_split,
                        "--fit-num-prompts",
                        args.fit_num_prompts,
                        "--tetris-loads",
                        str(args.offered_load),
                    ]
                run(command, output / "profiles" / f"{algorithm}-calibration.log")
                shutil.copyfile(directory / algorithm / "profile.json", frozen)
                cache = WORK / f"{algorithm}-profiles"
                cache.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(frozen, cache / f"{digest(frozen)}.json")
        profiles[algorithm] = {"path": str(frozen), "sha256": digest(frozen)}
    if profiles:
        import torch

        expected = dict(
            model_path=os.path.realpath(model),
            dtype="bfloat16",
            world_size=8,
            gpu_names=[torch.cuda.get_device_name(i) for i in range(8)],
            execution_mode="async",
            attention_backend=args.attention_backend,
            kv_transport=args.kv_transport,
            shared_attention="auto",
            migration_overlap=True,
        )
        for algorithm, profile in profiles.items():
            raw = json.loads(Path(profile["path"]).read_text())
            for key, value in expected.items():
                if raw.get("metadata", {}).get(key) != value:
                    raise ValueError(
                        f"{algorithm} profile {key} does not match this experiment"
                    )
            if raw.get("nodes") != [0] * 8:
                raise ValueError(
                    f"{algorithm} profile topology is not a single eight-GPU node"
                )
        save_json(output / "profiles" / "manifest.json", profiles)
    # Check all evaluation sources before launching any measured trial.
    if "tetris" in profiles:
        for name, load, directory, records, manifest in workloads:
            run(
                module
                + [
                    "sglang.srt.dynamic_sp.tetris_profile",
                    "check-data",
                    "--profile",
                    profiles["tetris"]["path"],
                    "--evaluation-manifest",
                    directory / "manifest.json",
                ],
                directory / "profile-check.log",
            )
    reference_model = None
    if "tetris" in profiles:
        reference_model = json.loads(Path(profiles["tetris"]["path"]).read_text())[
            "models"
        ]["1"]
    summaries, distribution = [], []
    failed = False
    roofline_reference = None
    if reference_model is None and "spsrpt" in profiles:
        from .scheduler import SchedulerConfig
        from .spsrpt import SPSRPTScheduler

        roofline_reference = SPSRPTScheduler.from_file(
            SchedulerConfig(8, args.sp_kv_tokens), profiles["spsrpt"]["path"]
        )
    for name, load, directory, records, manifest in workloads:
        for repeat in range(args.repeats):
            order = (
                args.schedulers[repeat % len(args.schedulers) :]
                + args.schedulers[: repeat % len(args.schedulers)]
            )
            for label in order:
                trial = (
                    output / "runs" / name / directory.name / f"{label}-{repeat + 1}"
                )
                identity = dict(
                    dataset=name,
                    split=manifest["split"],
                    excluded_prompts=len(manifest.get("excluded_rows", [])),
                    truncated_prompts=manifest.get("truncated_records", 0),
                    truncated_tokens=manifest.get("truncated_tokens", 0),
                    model=model,
                    scheduler=label,
                    implementation=SCHEDULERS[label],
                    repeat=repeat + 1,
                    offered_load_tokens_per_second=load,
                    workload_sha256=manifest["workload_sha256"],
                    short_threshold_tokens=args.short_threshold,
                    attention_backend=args.attention_backend
                    if label in ("ours", "spsrpt", "tetris", "loongserve")
                    else "native",
                    kv_transport=args.kv_transport
                    if label in ("ours", "spsrpt", "tetris", "loongserve")
                    else "native",
                    result_path=str(trial / "results.json"),
                )
                row = dict(identity, status="failed", error="")
                command = launcher + [
                    "--scheduler",
                    SCHEDULERS[label],
                    "--workload-dir",
                    directory,
                    "--output-dir",
                    trial,
                    "--offered-load",
                    load,
                ]
                if label in ("tetris", "loongserve", "ours", "spsrpt"):
                    command += sp
                    profile = profiles[
                        "tetris" if label == "ours" else label
                    ]
                    row["profile_sha256"] = profile["sha256"]
                    if label == "spsrpt":
                        command += ["--spsrpt-profile", profile["path"],
                                    "--spsrpt-quantum", args.spsrpt_quantum,
                                    "--spsrpt-prediction-tolerance", args.spsrpt_prediction_tolerance,
                                    "--spsrpt-prediction-max-panels", args.spsrpt_prediction_max_panels]
                    elif label == "ours":
                        command += ["--", "--dynamic-profile", profile["path"]]
                    else:
                        command += [f"--{label}-profile", profile["path"]]
                else:
                    command += ["--max-local-tokens", args.native_kv_tokens]
                    if label == "laps" and args.laps_config:
                        command += ["--laps-config", args.laps_config.resolve()]
                try:
                    if (
                        digest(directory / "workload.json")
                        != manifest["workload_sha256"]
                    ):
                        raise ValueError("Prepared workload changed during experiments")
                    for profile in profiles.values():
                        if digest(Path(profile["path"])) != profile["sha256"]:
                            raise ValueError(
                                "Frozen profile changed during experiments"
                            )
                    if label == "tetris" and args.reuse_tetris_from:
                        source_trial = args.reuse_tetris_from / "runs" / name / directory.name / f"{label}-{repeat + 1}"
                        trial.mkdir(parents=True, exist_ok=False)
                        shutil.copyfile(source_trial / "results.json", trial / "results.json")
                        row["reused_from"] = str(source_trial / "results.json")
                        identity["reused_from"] = row["reused_from"]
                        save_json(trial / "reuse.json", {
                            "source": row["reused_from"],
                            "sha256": digest(trial / "results.json"),
                            "source_git_revision": reused_configuration.get("git_revision"),
                        })
                        print(f"[experiments] Reusing Tetris: {source_trial}", flush=True)
                    else:
                        run(command, trial.parent / f"{label}-{repeat + 1}.log")
                    result = json.loads((trial / "results.json").read_text())
                    if "request_timings" not in result:
                        result["request_timings"] = result["results"]
                    row.update(measured_metrics(result, records))
                    timings = []
                    for record in result["request_timings"]:
                        latency = (
                            record["completion_time_seconds"]
                            - record["arrival_time_seconds"]
                        )
                        timings.append(
                            dict(
                                identity,
                                **{
                                    key: record[key]
                                    for key in (
                                        "request_id",
                                        "input_tokens",
                                        "arrival_time_seconds",
                                        "completion_time_seconds",
                                    )
                                },
                                ttft_seconds=latency,
                                length_class="short"
                                if record["input_tokens"] <= args.short_threshold
                                else "long",
                            )
                        )
                    if reference_model is not None or roofline_reference is not None:
                        ref = reference_model
                        for timing in timings:
                            length = timing["input_tokens"]
                            seconds = (roofline_reference.reference_seconds(length)
                                       if roofline_reference is not None else (
                                ref["a"]
                                + ref["b"] * length
                                + max(ref["c"] / 2, ref["d"]) * length * length
                            ))
                            timing["reference_model"] = "roofline-SP1" if roofline_reference else "tetris-SP1"
                            timing["reference_seconds_estimated"] = seconds
                            timing["slowdown_estimated"] = (
                                timing["ttft_seconds"] / seconds
                            )
                        for group in ("all", "short", "long"):
                            values = sorted(
                                item["slowdown_estimated"] for item in timings
                                if group == "all" or item["length_class"] == group
                            )
                            prefix = ("slowdown_estimated" if group == "all"
                                      else f"slowdown_{group}_estimated")
                            row[f"{prefix}_mean"] = statistics.mean(values) if values else None
                            row[f"{prefix}_p50"] = statistics.median(values) if values else None
                            row[f"{prefix}_p95"] = values[math.ceil(.95 * len(values)) - 1] if values else None
                    for group in ("all", "short", "long"):
                        row.update(
                            stats(
                                [
                                    item["ttft_seconds"]
                                    for item in timings
                                    if group == "all" or item["length_class"] == group
                                ],
                                f"ttft_{group}",
                            )
                        )
                    save_csv(trial / "ttft.csv", timings)
                    distribution.extend(timings)
                    row["status"] = "ok"
                    print(
                        f"[experiments] {name} {label} load={load:g} tokens/sec={row['throughput_tokens_per_second']:.1f} "
                        f"TTFT p50={row['ttft_p50_seconds']:.3f}s p95={row['ttft_p95_seconds']:.3f}s",
                        flush=True,
                    )
                    for group in ("short", "long"):
                        print(
                            f"[experiments] {group}: n={row[f'ttft_{group}_count']} "
                            f"p50={row[f'ttft_{group}_p50_seconds']}s p95={row[f'ttft_{group}_p95_seconds']}s",
                            flush=True,
                        )
                except (
                    subprocess.CalledProcessError,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as error:
                    row["error"] = str(error)
                    failed = True
                    print(f"[experiments] FAILED {name}/{label}: {error}", flush=True)
                summaries.append(row)
                save_csv(output / "summary.csv", summaries)
                if distribution:
                    save_csv(output / "ttft.csv", distribution)
                if failed:
                    print_summary(summaries)
                    print_slowdown_tables(summaries)
                    raise SystemExit(
                        f"Experiment stopped after failure; see {output / 'summary.csv'}"
                    )
    print_summary(summaries)
    print_slowdown_tables(summaries)
    print(f"[experiments] Summary: {output / 'summary.csv'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
