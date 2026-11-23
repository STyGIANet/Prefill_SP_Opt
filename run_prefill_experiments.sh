#!/usr/bin/env bash
# Experiment flow:
# 1. Prepare each selected dataset and one Poisson arrival trace per workload.
#    Every scheduler receives the same token IDs and scheduled arrivals.
# 2. Freeze profiles before evaluation:
#    - Tetris: reuse a compatible profile, or calibrate and fit thresholds on
#      independent training data when no profile is available.
#    - LoongServe (if selected): reuse a compatible profile or calibrate once.
#    - SPSRPT: reuse or fit a roofline profile from existing measurements;
#      missing hardware measurements cause an error rather than automatic runs.
#    - LAPS and native DP: no fitted latency profile. The "ours" label uses
#      the dynamic scheduler with the Tetris latency profile.
# 3. For each dataset and repetition, run schedulers in --schedulers order.
#    With --schedulers tetris laps spsrpt, the first repetition runs:
#      Tetris -> LAPS -> SPSRPT.
#    Order rotates between repetitions; profiles remain fixed throughout.
# 4. Each trial starts a fresh runtime, loads the model, initializes execution
#    resources, warms up, clears warmup KV, and then submits measured arrivals.
#    SP warms backend shapes and processed contexts; LAPS uses native warmup
#    and CUDA graph preparation. Warmup is excluded from reported timings.
# 5. Save results after each trial. A failed trial stops the sequence.
# 6. Print throughput, overall/short/long TTFT, and estimated slowdown;
#    write summary.csv and the full per-prompt ttft.csv distribution.
# --reuse-tetris-from copies saved Tetris results and frozen inputs/profiles,
# then runs only SPSRPT and reports both methods together.
# This wrapper selects the Python environment and invokes experiments.py.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    python_bin="$CONDA_PREFIX/bin/python"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
else
    python_bin=python3
fi
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m sglang.srt.dynamic_sp.experiments "$@"
