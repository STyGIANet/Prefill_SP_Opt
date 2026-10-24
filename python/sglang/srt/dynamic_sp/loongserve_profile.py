# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Fit LoongServe Eq. (7) from independently measured full-prefill batches."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def fit(raw, max_prefill_seconds=0.5, max_validation_error=0.25):
    models, diagnostics = {}, {}
    world = raw["metadata"]["world_size"]
    for degree in range(1, world + 1):
        rows = [
            r
            for r in raw["samples"]
            if r["degree"] == degree and r.get("role", "fit") == "fit"
        ]
        x = np.array(
            [[1, sum(r["lengths"]), sum(n * n for n in r["lengths"])] for r in rows],
            dtype=float,
        )
        y = np.array([r["seconds"] for r in rows])
        if (
            len(rows) < 3
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or (y <= 0).any()
        ):
            raise ValueError(
                f"Insufficient or invalid calibration samples for SP={degree}"
            )
        scale = np.linalg.norm(x, axis=0)
        coef, _, rank, singular = np.linalg.lstsq(x / scale, y, rcond=None)
        if rank != 3:
            raise ValueError(f"Rank-deficient calibration for SP={degree}")
        coef /= scale
        if not np.isfinite(coef).all() or (x @ coef <= 0).any():
            raise ValueError(f"Invalid fitted latencies for SP={degree}")
        models[str(degree)] = coef.tolist()
        diagnostics[str(degree)] = {
            "samples": len(rows),
            "rmse_seconds": float(np.sqrt(np.mean((x @ coef - y) ** 2))),
            "condition_number_scaled": float(singular[0] / singular[-1]),
            "max_total_tokens": max(sum(r["lengths"]) for r in rows),
            "max_prompt_tokens": max(max(r["lengths"]) for r in rows),
        }
        validation = [
            r
            for r in raw["samples"]
            if r["degree"] == degree and r.get("role") == "validation"
        ]
        errors = []
        for row in validation:
            n = row["lengths"]
            predicted = float(coef @ np.array([1, sum(n), sum(v * v for v in n)]))
            observed = row["seconds"]
            if not math.isfinite(observed) or observed <= 0 or predicted <= 0:
                raise ValueError(
                    f"Invalid held-out prediction or observation for SP={degree}"
                )
            errors.append(
                {
                    "lengths": n,
                    "observed_seconds": observed,
                    "predicted_seconds": predicted,
                    "relative_error": abs(predicted - observed) / observed,
                }
            )
        diagnostics[str(degree)]["validation"] = errors
        diagnostics[str(degree)]["max_validation_relative_error"] = max(
            (e["relative_error"] for e in errors), default=None
        )
        if any(e["relative_error"] > max_validation_error for e in errors):
            raise ValueError(
                f"SP={degree}: held-out latency error exceeds {max_validation_error:.0%}; use a wider calibration range or --grid full. Validation points were not used for fitting."
            )
    if not math.isfinite(max_prefill_seconds) or max_prefill_seconds <= 0:
        raise ValueError("Prefill time limit must be positive")
    return {
        "schema_version": 1,
        "algorithm": "loongserve",
        "models": models,
        "nodes": raw["nodes"],
        "metadata": raw["metadata"],
        "fit": diagnostics,
        "fit_method": "ordinary-least-squares",
        "max_prefill_seconds": max_prefill_seconds,
        "max_validation_error": max_validation_error,
        "samples_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-prefill-seconds",
        type=float,
        default=0.5,
        help="Admission threshold; upstream default is 500 ms; calibrate for your hardware",
    )
    # Tetris Implementation: short-prompt workloads (e.g. LMSYS) leave per-request
    # overhead at high SP that the a + b*sum(n) + c*sum(n^2) model cannot express.
    parser.add_argument(
        "--max-validation-error",
        type=float,
        default=0.25,
        help="Largest accepted held-out relative latency error (default: 0.25)",
    )
    args = parser.parse_args()
    profile = fit(
        json.loads(Path(args.samples).read_text()),
        args.max_prefill_seconds,
        args.max_validation_error,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile, indent=2) + "\n")
    temporary.replace(target)


if __name__ == "__main__":
    main()
