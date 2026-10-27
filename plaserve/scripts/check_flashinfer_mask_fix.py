"""Test saturating causal-mask arithmetic using isolated FlashInfer headers/JIT.

Does not edit the installed package. Pass --unpatched for the same isolated
compilation path with original headers. GPU validation is still required.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys


def fix_header(source):
    replacements = [
        ("kv_len + ceil_div((qo_tile_idx * CTA_TILE_Q), group_size) - qo_len",
         "sub_if_greater_or_zero(kv_len + ceil_div((qo_tile_idx * CTA_TILE_Q), group_size), qo_len)"),
        ("kv_len + ceil_div((bx * CTA_TILE_Q), group_size) - qo_len",
         "sub_if_greater_or_zero(kv_len + ceil_div((bx * CTA_TILE_Q), group_size), qo_len)"),
        ("kv_len - qo_len + ceil_div(((qo_tile_idx + 1) * CTA_TILE_Q), group_size)",
         "sub_if_greater_or_zero(kv_len + ceil_div(((qo_tile_idx + 1) * CTA_TILE_Q), group_size), qo_len)"),
        ("kv_len - qo_len + ceil_div(((bx + 1) * CTA_TILE_Q), group_size)",
         "sub_if_greater_or_zero(kv_len + ceil_div(((bx + 1) * CTA_TILE_Q), group_size), qo_len)"),
    ]
    counts = []
    for old, new in replacements:
        counts.append(source.count(old))
        source = source.replace(old, new)
    if counts[0] == 0 or counts[2] == 0:
        raise RuntimeError(f"Installed kernel differs from inspected source; no patch applied ({counts})")
    return source, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unpatched", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--case", type=int)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()
    import flashinfer
    from flashinfer.jit import env

    original_include = Path(env.FLASHINFER_INCLUDE_DIR)
    relative = Path("flashinfer/attention/prefill.cuh")
    original = (original_include / relative).read_text()
    patched, counts = fix_header(original)
    selected = original if args.unpatched else patched
    digest = hashlib.sha256(selected.encode()).hexdigest()
    mode = "original" if args.unpatched else "saturating"
    root = Path("kernels") / ("flashinfer-" + mode + "-" + digest[:12])
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    include = root / "include"
    if not include.exists():
        shutil.copytree(original_include, include)
    (include / relative).write_text(selected)
    (root / "kernel.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), patched.splitlines(True),
        fromfile="original/prefill.cuh", tofile="saturating/prefill.cuh")))
    # Redirect runtime build inputs and outputs, including AOT selection, before
    # importing the checker or creating attention wrappers.
    env.FLASHINFER_INCLUDE_DIR = include
    env.FLASHINFER_WORKSPACE_DIR = root
    env.FLASHINFER_JIT_DIR = root / "cached_ops"
    env.FLASHINFER_GEN_SRC_DIR = root / "generated"
    env.FLASHINFER_AOT_DIR = root / "empty_aot"
    for path in (env.FLASHINFER_JIT_DIR, env.FLASHINFER_GEN_SRC_DIR, env.FLASHINFER_AOT_DIR):
        path.mkdir(exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "4")
    manifest = {"flashinfer": flashinfer.__version__, "mode": mode,
                "original_header": str(original_include / relative),
                "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "selected_sha256": digest, "replacement_counts": counts,
                "include": str(include), "jit": str(env.FLASHINFER_JIT_DIR),
                "model": args.model, "case": args.case, "status": "running"}
    manifest_path = root / ("all-manifest.json" if args.case is None else f"case-{args.case}-manifest.json")
    print(f"Kernel experiment: {mode}; replacement counts={counts}; artifacts={root}", flush=True)
    checker = Path(__file__).with_name("check_blackwell_attention.py")
    sys.argv = [str(checker), "--model", args.model, "--tp", str(args.tp)]
    if args.case is not None:
        sys.argv += ["--case", str(args.case)]
    try:
        runpy.run_path(str(checker), run_name="__main__")
        # Verify that generated compilation commands reference the copied headers.
        builds = list(env.FLASHINFER_JIT_DIR.glob("batch_prefill*/build.ninja"))
        if not builds or not all(str(include) in p.read_text() for p in builds):
            raise RuntimeError("Cannot verify attention compilation used isolated headers")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
