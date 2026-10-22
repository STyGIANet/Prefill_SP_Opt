# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Select a compatible frozen calibration without launching GPU measurements."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("loongserve", "tetris"), required=True)
    parser.add_argument("--execution-mode", choices=("async", "sync"), default="async")
    parser.add_argument("--allow-profile-backend-mismatch", action="store_true")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--kv-transport", required=True)
    parser.add_argument("--shared-attention", required=True)
    parser.add_argument(
        "--migration-overlap", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    expected = dict(
        model_path=os.path.realpath(args.model_path),
        dtype=args.dtype,
        world_size=8,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(8)],
        execution_mode=args.execution_mode,
        attention_backend=args.attention_backend,
        kv_transport=args.kv_transport,
        shared_attention=args.shared_attention,
        migration_overlap=args.migration_overlap,
    )
    algorithm = args.algorithm
    if args.allow_profile_backend_mismatch:
        expected.pop("attention_backend")
        expected.pop("kv_transport")
    candidates = list((args.work_dir / f"{algorithm}-profiles").glob("*.json"))
    candidates += list(
        args.work_dir.glob(f"dynamic-sp-8gpu.*/{algorithm}/profile.json")
    )
    candidates += list(args.work_dir.glob(f"benchmark-*/*/*/{algorithm}/profile.json"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if path.resolve() == args.output.resolve():
            continue
        try:
            raw = json.loads(path.read_text())
            metadata = raw["metadata"]
            if algorithm == "loongserve":
                if raw.get("algorithm") != algorithm or raw.get("schema_version") != 1:
                    continue
                degrees = {str(i) for i in range(1, 9)}
            else:
                if (
                    raw.get("schema_version") != 2
                    or raw.get("threshold_status") != "simulated"
                    or not raw.get("thresholds")
                ):
                    continue
                degrees = {"1", "2", "4", "8"}
            if any(metadata.get(k) != v for k, v in expected.items()):
                continue
            if raw.get("nodes") != [0] * 8 or set(raw.get("models", {})) != degrees:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        payload = json.dumps(raw, indent=2) + "\n"
        cache = args.work_dir / f"{algorithm}-profiles"
        cache.mkdir(parents=True, exist_ok=True)
        # Content-addressed files preserve every frozen calibration.
        destination = cache / (hashlib.sha256(payload.encode()).hexdigest() + ".json")
        if not destination.exists():
            destination.write_text(payload)
            os.utime(destination, (path.stat().st_atime, path.stat().st_mtime))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"[dynamic-sp] Reusing {algorithm} profile: {path}", flush=True)
        return
    parser.exit(
        2,
        f"No compatible {algorithm} profile found. Use --recalibrate-{algorithm} to measure a new profile, or --{algorithm}-profile PATH.\n",
    )


if __name__ == "__main__":
    main()
