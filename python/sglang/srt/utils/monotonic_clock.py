# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Identify processes that can exchange monotonic timestamps unchanged."""

import os
import time
from pathlib import Path


def monotonic_clock_domain():
    # Do not cache: a process can enter another Linux time namespace.
    if (
        time.get_clock_info("perf_counter").implementation
        != "clock_gettime(CLOCK_MONOTONIC)"
    ):
        return None
    try:
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            os.readlink("/proc/self/ns/time"),
        )
    except OSError:
        return None
