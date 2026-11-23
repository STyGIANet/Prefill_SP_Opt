# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Compiled exact allocation for weighted remaining-time indices."""
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


class RemainingTimeAllocator:
    def __init__(self):
        source = Path(__file__).parent / "csrc/remaining_time.cpp"
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
        cache = Path(os.environ.get("SGLANG_SP_BUILD_DIR", str(source.parent.parent / "work/build")))
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / f"remaining-time-{fingerprint}.so"
        with (cache / "remaining-time.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=cache) as temporary:
                    output = Path(temporary) / "scheduler.so"
                    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3",
                                    "-shared", "-fPIC", str(source), "-o", str(output)], check=True)
                    output.replace(target)
        self.library = ctypes.CDLL(str(target))
        self.solve = self.library.remaining_time_allocate
        ints = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        doubles = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        self.solve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ints, ints,
                               doubles, ints, ints]
        self.solve.restype = ctypes.c_int

    def __deepcopy__(self, memo):
        return self

    def allocate(self, world_size, slots, candidates, admission):
        offsets = [0]
        flat = []
        for group in candidates:
            flat.extend(group)
            offsets.append(len(flat))
        selected = np.full(len(candidates), -1, dtype=np.int32)
        status = self.solve(len(candidates), world_size, min(slots, world_size),
                            np.asarray(offsets, dtype=np.int32),
                            np.asarray([c["mask"] for c in flat], dtype=np.int32),
                            np.asarray([c["index"] for c in flat], dtype=np.float64),
                            np.asarray(admission, dtype=np.int32), selected)
        if status:
            raise RuntimeError(f"Remaining-time allocation failed: {status}")
        return [None if index < 0 else flat[index] for index in selected]
