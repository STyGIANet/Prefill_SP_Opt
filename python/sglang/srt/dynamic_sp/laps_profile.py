# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Fit the LAPS compute/memory boundary from independent phase measurements."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sglang.srt.dynamic_sp.laps import LAPSConfig


def fit(samples):
    if len(samples) < 3:
        raise ValueError("Need at least three independent phase measurements")
    l = np.asarray([s["new_tokens"] for s in samples], dtype=float)
    h = np.asarray([s["history_tokens"] for s in samples], dtype=float)
    y = np.asarray(
        [[s["compute_seconds"], s["memory_seconds"]] for s in samples], dtype=float
    )
    if (
        not np.isfinite(l).all()
        or not np.isfinite(h).all()
        or not np.isfinite(y).all()
        or (l <= 0).any()
        or (h < 0).any()
        or (y <= 0).any()
    ):
        raise ValueError(
            "Lengths and phase measurements must be finite; new lengths and durations positive"
        )
    costs, errors = [], []
    for x, target in (
        (np.stack((l * (l + 2 * h), l), axis=1), y[:, 0]),
        (np.stack((l, h), axis=1), y[:, 1]),
    ):
        scale = np.linalg.norm(x, axis=0)
        if (scale == 0).any() or np.linalg.matrix_rank(x / scale) != 2:
            raise ValueError(
                "Boundary fit requires varied input lengths and nonzero historical contexts"
            )
        coef, _ = nnls(x / scale, target)
        coef /= scale
        costs.extend(coef.tolist())
        errors.append(float(np.sqrt(np.mean((x @ coef - target) ** 2))))
    LAPSConfig(boundary_coefficients=tuple(costs))
    return costs, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True)
    p.add_argument("--config", help="Existing LAPS configuration to update")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    raw = json.loads(Path(args.samples).read_text())
    costs, errors = fit(raw["samples"])
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    config["boundary_coefficients"] = costs
    output = Path(args.output)
    output.write_text(json.dumps(config, indent=2) + "\n")
    output.with_suffix(".fit.json").write_text(
        json.dumps(
            {
                "samples": args.samples,
                "metadata": raw.get("metadata", {}),
                "compute_rmse_seconds": errors[0],
                "memory_rmse_seconds": errors[1],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
