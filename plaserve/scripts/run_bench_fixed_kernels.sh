#!/usr/bin/env bash
# Usage: conda activate plaserve; bash scripts/run_bench_fixed_kernels.sh OUTPUT [--prepare-only]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
[[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: run_bench_fixed_kernels.sh OUTPUT [--prepare-only]' >&2; exit 2; }
[[ $# -eq 1 || "$2" == --prepare-only ]] || exit 2
[[ ${CONDA_DEFAULT_ENV:-} == plaserve ]] || { echo 'Activate conda environment plaserve first' >&2; exit 2; }
export FIXED_RUN_DIR=$(realpath -m -- "$1")
mkdir -- "$FIXED_RUN_DIR"
export PYTHON=$(command -v python)
export BENCH_SOURCE_ROOT="$ROOT/vendor/LAPS/bench_laps_prefill_throughput"
export SAFE_LAUNCHER="$FIXED_RUN_DIR/launch_owned.py"
export LAPS_FLASHINFER_KERNEL_ROOT=$(realpath "$ROOT/kernels/flashinfer-saturating-4fbcdb804781")
export LAPS_KERNEL_EVENT_DIR="$FIXED_RUN_DIR/kernel-events"
export PYTHONPATH="$ROOT/scripts/laps_kernel_hook${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2,3,4,5
export NUM_PROMPTS=${NUM_PROMPTS:-500}
export CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS:-1 4 8 16}
export MAX_JOBS=${MAX_JOBS:-4}
unset LAPS_REPLAY_CONTROL_DIR LAPS_LOGITS_DIR LAPS_DIAGNOSTIC_CASE_FILE LAPS_PD_COVERAGE_DIR
python scripts/prepare_bench_fixed.py "$BENCH_SOURCE_ROOT/scripts/bench_h100_4gpu.sh" "$FIXED_RUN_DIR"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['LAPS_FLASHINFER_KERNEL_ROOT'])
m = json.loads((root / 'all-manifest.json').read_text())
assert m['status'] == 'passed' and m['mode'] == 'saturating'
assert hashlib.sha256((root / 'include/flashinfer/attention/prefill.cuh').read_bytes()).hexdigest() == m['selected_sha256']
assert int(os.environ['NUM_PROMPTS']) > 0
cc = [int(c) for c in os.environ['CONCURRENCY_LEVELS'].split()]
assert cc and min(cc) > 0 and len(set(cc)) == len(cc)
p = Path(os.environ['FIXED_RUN_DIR'])
(p / 'run-config.json').write_text(json.dumps({k: os.environ.get(k) for k in
    ('NUM_PROMPTS', 'CONCURRENCY_LEVELS', 'BACKEND', 'LAPS_FLASHINFER_KERNEL_ROOT', 'CUDA_VISIBLE_DEVICES')}, indent=2))
runner = Path('vendor/LAPS/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py')
assert 'LAPS_VERIFY ' not in runner.read_text(), 'Remove replay instrumentation before timing'
PY
if [[ ${2:-} == --prepare-only ]]; then
    echo "Prepared only; no servers or GPU work started: $FIXED_RUN_DIR"
    exit 0
fi
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$FIXED_RUN_DIR/gpu-before.csv"
# Refuse occupied service/bootstrap ports before launching any owned process.
python - <<'PY'
import socket
sockets = []
try:
    for port in (30300, 30301, 30302, 9300, 9301):
        s = socket.socket()
        sockets.append(s)
        s.bind(('127.0.0.1', port))
finally:
    for s in sockets:
        s.close()
PY
set +e
"$PYTHON" "$SAFE_LAUNCHER" "$ROOT/scripts/monitor_bench_gpus.py" "$FIXED_RUN_DIR/gpu-samples.jsonl" &
monitor_pid=$!
stop_monitor() {
    kill -TERM -- "-$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
driver_pid=
cleanup_wrapper() {
    if [[ -n "$driver_pid" ]]; then
        kill -TERM -- "-$driver_pid" 2>/dev/null || true
        wait "$driver_pid" 2>/dev/null || true
    fi
    stop_monitor
}
trap cleanup_wrapper EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON" "$SAFE_LAUNCHER" --exec bash "$FIXED_RUN_DIR/upstream-guarded.sh" 7b > "$FIXED_RUN_DIR/driver.log" 2>&1 &
driver_pid=$!
wait "$driver_pid"
run_status=$?
driver_pid=
stop_monitor
trap - EXIT
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$FIXED_RUN_DIR/gpu-after.csv"
python scripts/validate_bench_results.py "$FIXED_RUN_DIR" \
    --logs "$FIXED_RUN_DIR/raw" --kernel-events "$LAPS_KERNEL_EVENT_DIR" \
    --kernel-root "$LAPS_FLASHINFER_KERNEL_ROOT" --requested "$NUM_PROMPTS" \
    --concurrency $CONCURRENCY_LEVELS --upstream-status "$run_status"
exit $?
