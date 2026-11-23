# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Build a frozen SP roofline profile from production-ring measurements."""

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

from .spsrpt import RooflineLatency


def geometry(model_path):
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    if cfg.get("model_type") != "llama" or cfg.get("hidden_act", "silu") != "silu":
        raise ValueError("The roofline dense-stage model currently requires dense Llama with SiLU")
    return {
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "num_layers": cfg["num_hidden_layers"],
        "query_heads": cfg["num_attention_heads"],
        "kv_heads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "head_dim": cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        "vocab_size": cfg["vocab_size"],
        "element_bytes": 2,
    }


def source_compatible(raw, shape):
    meta = raw.get("metadata", {})
    return (
        meta.get("execution") == "production-overlap"
        and meta.get("dtype") == "bfloat16"
        and all(meta.get(k) == shape[k] for k in ("query_heads", "kv_heads", "head_dim"))
        and bool(raw.get("samples"))
        and bool(raw.get("correctness"))
        and all(rank["passed"] for check in raw["correctness"] for rank in check["ranks"])
    )


def build_profile(raw, model_path, source_hash):
    import numpy as np
    from scipy.optimize import least_squares, nnls

    shape = geometry(model_path)
    if not source_compatible(raw, shape):
        raise ValueError("Roofline samples do not match the model's attention geometry, BF16, or production ring")
    hardware = raw["hardware"]
    names = raw["metadata"]["gpu_names"]
    if not hardware or len(hardware) != len(names):
        raise ValueError("Missing per-rank hardware measurements")
    degrees = sorted({int(x["degree"]) for x in raw["samples"]})
    if 1 not in degrees or min(degrees) < 1 or max(degrees) > len(hardware):
        raise ValueError("Require SP1 and valid measured SP degrees")
    checked = {int(x["degree"]) for x in raw["correctness"] if x["ranks"]}
    if not set(degrees).issubset(checked):
        raise ValueError("Missing attention reference checks for measured SP degrees")
    profile = {
        "schema": "sp-roofline-v1",
        "source": {"sha256": source_hash, "scope": raw["metadata"]["scope"]},
        "geometry": shape,
        "hardware": hardware,
        "attention": {}, "network": {}, "fit_diagnostics": {},
        "nodes": [0] * len(hardware),
        "dense_launch_seconds": 0.0,
        "step_launch_seconds": 0.0,
        "placement_calibration": "Measured prefix groups 0..s-1; degree-level transport reused for other subsets",
        "dense_calibration": "Measured 4096-square GEMM and copy ceilings; dense shape efficiency and launch costs unmeasured",
        "metadata": {
            "model_path": os.path.realpath(model_path), "dtype": "bfloat16",
            "world_size": len(hardware), "gpu_names": names,
            "execution_mode": "async", "attention_backend": "ring",
            "kv_transport": "targeted", "shared_attention": "auto", "migration_overlap": True,
        },
    }
    for degree in degrees:
        samples = [x for x in raw["samples"] if x["degree"] == degree]
        if len(samples) < 4:
            raise ValueError(f"SP={degree}: insufficient attention measurements")
        for x in samples:
            if (x["history"] < 0 or x["chunk"] <= 0 or
                    not math.isfinite(x["seconds"]) or x["seconds"] <= 0):
                raise ValueError("Invalid roofline sample")
        if degree > 1:
            # Normalize bytes to GB for a well-conditioned alpha + bytes/beta fit.
            design = np.array([[1, x["network_send_bytes"] / (degree - 1) / 1e9]
                               for x in samples])
            target = np.array([x["transport_seconds"] / (degree - 1) for x in samples])
            if not np.all(np.isfinite(target)) or np.any(target <= 0):
                raise ValueError("Invalid transport measurement")
            coef, _ = nnls(design, target)
            if coef[1] <= 0:
                raise ValueError(f"SP={degree}: transport bandwidth is not identifiable")
            net = {"startup_seconds": float(coef[0]), "bytes_per_second": float(1e9 / coef[1])}
        else:
            net = {"startup_seconds": 0.0, "bytes_per_second": 1e30}
        profile["network"][str(degree)] = net
        params = {"compute_efficiency": 0.5, "memory_efficiency": 0.5, "launch_seconds": 1e-5}
        profile["attention"][str(degree)] = params
        model = RooflineLatency(profile, degree)
        observed = np.array([x["seconds"] for x in samples])

        def residual(values):
            params.update(compute_efficiency=float(values[0]), memory_efficiency=float(values[1]),
                          launch_seconds=float(values[2] * 1e-6))
            predicted = np.array([model.attention_seconds(x["history"], x["chunk"]) for x in samples])
            return (predicted - observed) / observed

        fitted = least_squares(residual, [0.5, 0.5, 10.0],
                               bounds=([1e-4, 1e-4, 0], [1, 1, 1e5]),
                               max_nfev=500, x_scale="jac")
        if not fitted.success:
            raise ValueError(f"SP={degree}: roofline fit failed: {fitted.message}")
        errors = residual(fitted.x)
        # Freeze the selected parameters after the optimizer's finite differences.
        profile["attention"][str(degree)] = dict(params)
        profile["fit_diagnostics"][str(degree)] = {
            "samples": len(samples), "method": "bounded relative-error stage-roofline fit",
            "rmse_seconds": float(np.sqrt(np.mean((errors * observed) ** 2))),
            "max_relative_error": float(np.max(np.abs(errors))),
            "history_range": [min(x["history"] for x in samples), max(x["history"] for x in samples)],
            "length_range": [min(x["chunk"] for x in samples), max(x["chunk"] for x in samples)],
            "jacobian_rank": int(np.linalg.matrix_rank(fitted.jac)),
        }
    return profile


def prepare_profile(model_path, work_dir, output, samples=None, recalibrate=False):
    work_dir, output = Path(work_dir), Path(output)
    shape = geometry(model_path)
    sources = [Path(samples)] if samples else sorted(
        work_dir.glob("roofline-*/samples.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    selected = None
    for path in sources:
        raw = json.loads(path.read_text())
        if source_compatible(raw, shape):
            selected = path, raw
            break
    if selected is None:
        raise ValueError("No compatible production-ring roofline measurements. Supply --roofline-samples or collect them with run_sp_roofline.sh")
    path, raw = selected
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps({
        "schema": "sp-roofline-v1", "source": source_hash, "geometry": shape,
        "model_path": os.path.realpath(model_path),
    }, sort_keys=True).encode()).hexdigest()
    cache = work_dir / "spsrpt-profiles" / f"{signature}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if recalibrate or not cache.exists():
        profile = build_profile(raw, model_path, source_hash)
        profile["source"]["path"] = str(path.resolve())
        temporary = cache.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(profile, indent=2) + "\n")
        temporary.replace(cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    if cache.resolve() != output.resolve():
        shutil.copyfile(cache, output)
    print(f"[spsrpt] Frozen roofline profile: {output}; source={path}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--recalibrate", action="store_true")
    args = parser.parse_args()
    prepare_profile(args.model_path, args.work_dir, args.output, args.samples, args.recalibrate)


if __name__ == "__main__":
    main()
