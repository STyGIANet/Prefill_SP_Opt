# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Isolated FlashInfer FA2 headers for right-aligned rectangular queries."""

import fcntl
import hashlib
import json
import re
import shutil
from pathlib import Path

# Reviewed FA2 tile bounds, including the multiline multi-item bound.
_REVIEWED_HEADERS = {
    "ad8be30492fb04131edc25b539ae77e911177af8647c7e2007cf172a4a69d029": 9,  # 0.7.2
    "30b27d0b237ee155e25990b11893d5a5b026c86e7f02cf476d6ab5c621d22f72": 9,  # 0.7.4
}
_configured = False


def configure():
    """Preserve causal masking when padded Q is longer than real KV.

    FA2 uses unsigned tile bounds. Saturate the subtraction before it is used
    as an iteration bound, as in the Blackwell adapter supplied in plaserve.
    Installed packages remain unchanged; compiled objects use a separate cache.
    """
    global _configured
    if _configured:
        return
    import flashinfer
    from flashinfer.jit import env

    original_include = Path(env.FLASHINFER_INCLUDE_DIR)
    relative = Path("flashinfer/attention/prefill.cuh")
    source = (original_include / relative).read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash not in _REVIEWED_HEADERS:
        raise RuntimeError(
            "LAPS FlashInfer header differs from the current-environment adapter; "
            "review the padded causal bounds before using this version"
        )
    # Saturate causal iteration bounds in single, ragged, and paged prefill.
    term = r"ceil_div\([^;\n]*?group_size\)"
    selected, first_count = re.subn(
        rf"kv_len - qo_len\s*\+\s*({term})",
        r"sub_if_greater_or_zero(kv_len + \1, qo_len)",
        source,
    )
    selected, second_count = re.subn(
        rf"kv_len \+ ({term}) - qo_len",
        r"sub_if_greater_or_zero(kv_len + \1, qo_len)",
        selected,
    )
    if first_count + second_count != _REVIEWED_HEADERS[source_hash]:
        raise RuntimeError("LAPS causal-bound adaptation matched an unexpected number of expressions")
    digest = hashlib.sha256(selected.encode()).hexdigest()
    root = Path(__file__).resolve().parents[1] / "work" / "laps-flashinfer" / digest
    root.mkdir(parents=True, exist_ok=True)
    with (root / "headers.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        include = root / "include"
        ready = root / "manifest.json"
        if not ready.exists():
            shutil.copytree(original_include, include, dirs_exist_ok=True)
            (include / relative).write_text(selected)
            ready.write_text(
                json.dumps(
                    {
                        "flashinfer": flashinfer.__version__,
                        "original_sha256": source_hash,
                        "selected_sha256": digest,
                    },
                    indent=2,
                )
                + "\n"
            )
        if hashlib.sha256((include / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("LAPS isolated FlashInfer header has changed")
    env.FLASHINFER_INCLUDE_DIR = include
    env.FLASHINFER_WORKSPACE_DIR = root
    env.FLASHINFER_JIT_DIR = root / "cached_ops"
    env.FLASHINFER_GEN_SRC_DIR = root / "generated"
    env.FLASHINFER_AOT_DIR = root / "empty_aot"
    env.FLASHINFER_AOT_PROVIDERS = ()
    env.FLASHINFER_AOT_DIRS = (env.FLASHINFER_AOT_DIR,)
    for path in (
        env.FLASHINFER_JIT_DIR,
        env.FLASHINFER_GEN_SRC_DIR,
        env.FLASHINFER_AOT_DIR,
    ):
        path.mkdir(exist_ok=True)
    _configured = True
