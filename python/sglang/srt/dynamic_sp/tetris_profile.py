# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Offline latency fitting and load-threshold search for Tetris prefill.

No GPU work occurs here. Measurement collection is a separate, explicit command
in tetris_calibrate.py. All simulation comparisons reuse identical arrivals.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from sglang.srt.dynamic_sp.reservations import ReservationCalendar
from sglang.srt.dynamic_sp.tetris import CDSPPlanner, LatencyModel


def fit_models(samples, method="nonnegative"):
    """Least squares for Eq. (1), with column scaling.

    Nonnegative least squares preserves the planner's monotone latency model
    under measurement noise. Strict ordinary least squares remains selectable.
    Require all four terms to be identifiable rather than fitting a rank-deficient
    collection (for example measurements with no historical KV).
    """
    import numpy as np
    from scipy.optimize import nnls

    groups = {}
    for row in samples:
        s, c, n, t = (row[k] for k in ("degree", "history", "length", "seconds"))
        if (
            any(type(v) is not int for v in (s, c, n))
            or s <= 0
            or c < 0
            or n <= 0
            or not math.isfinite(t)
            or t <= 0
        ):
            raise ValueError("Invalid latency sample")
        groups.setdefault(s, []).append((c, n, t))
    if not groups:
        raise ValueError("No latency samples")
    models, diagnostics = {}, {}
    for degree, rows in sorted(groups.items()):
        x = np.array([[1, n, c * n, n * n] for c, n, _ in rows], dtype=float)
        y = np.array([t for _, _, t in rows])
        scale = np.maximum(np.linalg.norm(x, axis=0), 1)
        normalized = x / scale
        if np.linalg.matrix_rank(normalized) != 4:
            raise ValueError(
                f"SP={degree}: samples must identify all four latency terms; vary history and chunk length"
            )
        ordinary = np.linalg.lstsq(normalized, y, rcond=None)[0]
        if method == "least-squares":
            coef = ordinary.copy()
            tolerance = 100 * np.finfo(float).eps * max(1.0, np.linalg.norm(coef))
            coef[np.abs(coef) < tolerance] = 0
        elif method == "nonnegative":
            coef, _ = nnls(normalized, y)
        else:
            raise ValueError("Unknown latency fitting method")
        coef /= scale
        if np.any(coef < 0):
            raise ValueError(
                f"SP={degree}: least-squares fit has negative coefficients {coef.tolist()}; improve calibration coverage/repetitions or explicitly select nonnegative fitting"
            )
        model = LatencyModel(*map(float, coef))
        predicted = x @ coef
        models[str(degree)] = vars(model)
        diagnostics[str(degree)] = {
            "samples": len(rows),
            "method": method,
            "ordinary_coefficients": (ordinary / scale).tolist(),
            "ordinary_rmse_seconds": float(
                np.sqrt(np.mean((normalized @ ordinary - y) ** 2))
            ),
            "zero_terms": [
                term for term, value in zip(("a", "b", "c", "d"), coef) if value == 0
            ],
            "rmse_seconds": float(np.sqrt(np.mean((predicted - y) ** 2))),
            "max_relative_error": float(np.max(np.abs(predicted - y) / y)),
            "history_range": [min(r[0] for r in rows), max(r[0] for r in rows)],
            "length_range": [min(r[1] for r in rows), max(r[1] for r in rows)],
            "condition_number": float(np.linalg.cond(normalized)),
        }
    return models, diagnostics


def arrival_trace(lengths, tokens_per_second, count, seed):
    if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError(
            "Length distribution must contain positive integer token counts"
        )
    if not math.isfinite(tokens_per_second) or tokens_per_second <= 0 or count <= 0:
        raise ValueError("Simulation load and sample count must be positive")
    rng = random.Random(seed)
    rate = tokens_per_second / statistics.mean(lengths)
    now, trace = 0.0, []
    for _ in range(count):
        now += rng.expovariate(rate)
        trace.append((now, rng.choice(lengths)))
    return trace


def simulate(planner, trace, improvement_rate):
    """Execute the same persistent dependency calendar used by the online policy."""
    calendar = ReservationCalendar(len(planner.nodes))
    arrivals, latencies = {}, {}

    def start_ready(now):
        for task in calendar.ready():
            calendar.start(task.ticket, now)

    def advance(until):
        while True:
            running = [t for t in calendar.tasks.values() if t.status == "running"]
            if not running:
                return
            finish = min(t.start + t.duration for t in running)
            if finish > until:
                return
            for task in running:
                if task.start + task.duration <= finish:
                    calendar.complete(task.ticket, finish)
            for rid, ids in tuple(calendar.requests.items()):
                if all(calendar.tasks[i].status == "completed" for i in ids):
                    latencies[rid] = finish - arrivals[rid]
                    calendar.retire(rid)
            start_ready(finish)

    for index, (arrival, length) in enumerate(trace):
        advance(arrival)
        available = calendar.predict(arrival)
        chunks = planner.plan(
            length,
            [max(0.0, t - arrival) for t in available],
            improvement_rate=improvement_rate,
        )
        if chunks is None:
            raise ValueError("No feasible simulated SP allocation")
        rid = str(index)
        arrivals[rid] = arrival
        calendar.reserve(rid, chunks, planner.models, arrival)
        start_ready(arrival)
    advance(float("inf"))
    if len(latencies) != len(trace):
        raise RuntimeError("Reservation simulation failed to drain")
    return [latencies[str(i)] for i in range(len(trace))]


def search_thresholds(
    profile,
    lengths,
    loads,
    rates,
    count=1000,
    seed=0,
    objective="mean",
    chunk_solver="newton",
):
    if not rates or any(not math.isfinite(r) or not 0 <= r < 1 for r in rates):
        raise ValueError("Search rates must lie in [0, 1)")
    if not loads or any(not math.isfinite(load) or load <= 0 for load in loads):
        raise ValueError("Search loads must be finite and positive")
    if objective not in ("mean", "p50", "p95"):
        raise ValueError("Objective must be mean, p50, or p95 TTFT")
    models = {int(s): LatencyModel(**values) for s, values in profile["models"].items()}
    planner = CDSPPlanner(
        models, profile["nodes"], profile.get("min_chunk_tokens", 1), chunk_solver
    )
    thresholds, report = [], []
    for load in sorted(set(loads)):
        trace = arrival_trace(lengths, load, count, seed)
        trials = []
        for rate in sorted(set(rates)):
            latencies = sorted(simulate(planner, trace, rate))
            stats = {
                "mean": statistics.mean(latencies),
                "p50": statistics.median(latencies),
                "p95": latencies[math.ceil(0.95 * len(latencies)) - 1],
            }
            trials.append({"improvement_rate": rate, **stats})
        best = min(trials, key=lambda t: (t[objective], t["improvement_rate"]))
        thresholds.append(
            {
                "tokens_per_second": load,
                "requests_per_second": load / statistics.mean(lengths),
                "improvement_rate": best["improvement_rate"],
            }
        )
        report.append({"tokens_per_second": load, "trials": trials})
    # Below the lowest profiled load use its optimal rate, without extrapolation.
    thresholds.insert(
        0, {**thresholds[0], "tokens_per_second": 0.0, "requests_per_second": 0.0}
    )
    return thresholds, {
        "objective": objective,
        "requests": count,
        "seed": seed,
        "results": report,
    }


def check_data_separation(training, evaluation):
    """Check recorded provenance, without using evaluation lengths for tuning.

    Different named partitions are treated as separate. This is not a semantic
    deduplication guarantee across arbitrary datasets or dataset revisions.
    """
    required = ("dataset", "dataset_config", "split", "source_rows", "workload_sha256")
    if any(key not in m for m in (training, evaluation) for key in required):
        raise ValueError(
            "Both datasets need preparation manifests with source-row provenance"
        )
    if training["workload_sha256"] == evaluation["workload_sha256"]:
        raise ValueError("Threshold fitting and evaluation reuse the same workload")
    same_dataset = all(
        training[k] == evaluation[k] for k in ("dataset", "dataset_config")
    )
    if not same_dataset:
        return
    train_split, eval_split = training["split"], evaluation["split"]
    if train_split.split("[")[0] != eval_split.split("[")[0]:
        return
    if train_split != eval_split:
        raise ValueError(
            "Cannot verify disjoint sliced splits; use separate named partitions or datasets"
        )
    train_rows = {row["row_index"] for row in training["source_rows"]}
    eval_rows = {row["row_index"] for row in evaluation["source_rows"]}
    if train_rows & eval_rows:
        raise ValueError(
            "Threshold fitting and evaluation contain overlapping source rows; choose a held-out split or dataset"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-data", help="Validate fitting/evaluation provenance")
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-manifest")
    source.add_argument("--profile")
    check.add_argument("--evaluation-manifest", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument(
        "--samples",
        required=True,
        help="Calibration JSON containing samples and metadata",
    )
    fit.add_argument("--output", required=True)
    fit.add_argument(
        "--method", choices=("least-squares", "nonnegative"), default="nonnegative"
    )
    fit.add_argument("--window-seconds", type=float, default=30.0)
    fit.add_argument("--min-chunk-tokens", type=int, default=1)
    search = sub.add_parser("search")
    search.add_argument("--profile", required=True)
    search.add_argument(
        "--training-manifest",
        help="Provenance of the independent threshold-fitting workload",
    )
    search.add_argument(
        "--workload",
        required=True,
        help="Prepared workload.json, or JSON list of token lengths",
    )
    loads = search.add_mutually_exclusive_group(required=True)
    loads.add_argument("--loads", type=float, nargs="+", help="Input tokens per second")
    # Tetris Implementation: the paper profiles request rates; convert them to
    # token loads with the fitting workload's mean prompt length.
    loads.add_argument(
        "--request-rates",
        type=float,
        nargs="+",
        help="Requests per second, e.g. 0.5 1 2 4 8 16",
    )
    search.add_argument(
        "--chunk-solver", choices=("newton", "bisection"), default="newton"
    )
    search.add_argument(
        "--rates", type=float, nargs="+", default=[i / 100 for i in range(5, 76, 5)]
    )
    search.add_argument("--requests", type=int, default=1000)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--objective", choices=["mean", "p50", "p95"], default="mean")
    search.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "check-data":
        if args.profile:
            profile = json.loads(Path(args.profile).read_text())
            training = profile.get("threshold_training", {}).get("manifest")
            if training is None:
                parser.error(
                    "Profile has no fitting-dataset provenance; rebuild thresholds with --training-manifest"
                )
        else:
            training = json.loads(Path(args.training_manifest).read_text())
        evaluation = json.loads(Path(args.evaluation_manifest).read_text())
        check_data_separation(training, evaluation)
        return
    if args.command == "fit":
        raw = json.loads(Path(args.samples).read_text())
        models, diagnostics = fit_models(raw["samples"], args.method)
        if (
            not math.isfinite(args.window_seconds)
            or args.window_seconds <= 0
            or args.min_chunk_tokens <= 0
        ):
            parser.error("window-seconds and min-chunk-tokens must be positive")
        profile = {
            "schema_version": 2,
            "models": models,
            "fit_diagnostics": diagnostics,
            "fit_method": args.method,
            "metadata": raw["metadata"],
            "nodes": raw["nodes"],
            "arrival_window_seconds": args.window_seconds,
            "threshold_update_seconds": 30.0,
            "reservation_policy": "persistent_append_only",
            "min_chunk_tokens": args.min_chunk_tokens,
            "thresholds": [],
            "threshold_status": "search_required",
        }
    else:
        profile = json.loads(Path(args.profile).read_text())
        workload = json.loads(Path(args.workload).read_text())
        if isinstance(workload, dict):
            workload = workload["requests"]
        lengths = [len(r["token_ids"]) if isinstance(r, dict) else r for r in workload]
        loads = args.loads or [
            rate * statistics.mean(lengths) for rate in args.request_rates
        ]
        profile["thresholds"], profile["simulation"] = search_thresholds(
            profile,
            lengths,
            loads,
            args.rates,
            args.requests,
            args.seed,
            args.objective,
            args.chunk_solver,
        )
        profile["threshold_status"] = "simulated"
        encoded = Path(args.workload).read_bytes()
        training = {
            "workload_sha256": hashlib.sha256(encoded).hexdigest(),
            "num_prompts": len(lengths),
            "mean_tokens": statistics.mean(lengths),
        }
        if args.training_manifest:
            manifest = json.loads(Path(args.training_manifest).read_text())
            if manifest.get("workload_sha256") != training["workload_sha256"]:
                raise ValueError(
                    "Training manifest does not match the threshold-fitting workload"
                )
            training["manifest"] = manifest
        profile["threshold_training"] = training
    Path(args.output).write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
