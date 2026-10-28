"""Create a fail-closed, reviewable runtime copy of the upstream shell driver."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from vendor_provenance import snapshot


def prepare(source, output):
    text = source.read_text()
    original = text
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f'Upstream driver changed; expected exactly one {old!r}')
        text = text.replace(old, new)
    for value in ('PREFILL_GPUS="2,3"', 'DECODE_GPUS="4,5"', 'TP_SIZE=2'):
        if text.count(value) != 2:
            raise ValueError(f'Unexpected upstream GPU/TP mapping: {value}')
    for value in ('PREFILL_PORT=30300', 'DECODE_PORT=30301', 'ROUTER_PORT=30302'):
        if text.count(value) != 1:
            raise ValueError(f'Unexpected upstream ports: {value}')
    replace('set -uo pipefail', 'set -euo pipefail\nOWNED_PIDS=()')
    replace('DATASET="$(cd "$(dirname "$0")/../.." && pwd)/data/lmsys_chat_10k.jsonl"',
            'DATASET="$BENCH_SOURCE_ROOT/../data/lmsys_chat_10k.jsonl"')
    replace('SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"', 'SCRIPT_DIR="$BENCH_SOURCE_ROOT"')
    replace('RESULTS_DIR="${SCRIPT_DIR}/results_${MODEL_SHORT}_h100_4gpu_${TIMESTAMP}"',
            'RESULTS_DIR="$FIXED_RUN_DIR"')
    start, end = text.index('cleanup() {'), text.index('\nwait_ready() {')
    text = text[:start] + '''cleanup() {
    local pid
    for pid in "${OWNED_PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${OWNED_PIDS[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    OWNED_PIDS=()
}
''' + text[end:]
    for role, gpu in [('decode', '$DECODE_GPUS'), ('prefill', '$PREFILL_GPUS')]:
        replace(f'CUDA_VISIBLE_DEVICES={gpu} $PYTHON -m sglang.launch_server',
                f'LAPS_KERNEL_EVENT_DIR="$FIXED_RUN_DIR/kernel-events/${{label}}/{role}" '
                f'CUDA_VISIBLE_DEVICES={gpu} $PYTHON "$SAFE_LAUNCHER" -m sglang.launch_server')
    replace('$PYTHON -m sglang_router.launch_router', '$PYTHON "$SAFE_LAUNCHER" -m sglang_router.launch_router')
    for role in ('decode', 'prefill', 'router'):
        old = f'> "${{RAW_DIR}}/${{label}}_{role}.log" 2>&1 &'
        replace(old, old + '\n    OWNED_PIDS+=("$!")')
    replace('> /dev/null 2>&1 || true',
            '--output "${RAW_DIR}/${label}_warmup.json" \\\n        > "${RAW_DIR}/${label}_warmup.txt" 2>&1')
    replace('trap cleanup EXIT', 'trap cleanup EXIT\ntrap \'exit 130\' INT\ntrap \'exit 143\' TERM')
    if 'pkill' in text or 'LAPS_VERIFY ' in text:
        raise ValueError('Unsafe cleanup or diagnostics remain')
    target = output / 'upstream-guarded.sh'
    target.write_text(text)
    subprocess.run(['bash', '-n', str(target)], check=True)
    (output / 'launch_owned.py').write_text(
        'import os, sys\nos.setsid()\n'
        'if sys.argv[1] == "--exec":\n    os.execvp(sys.argv[2], sys.argv[2:])\n'
        'os.execv(sys.executable, [sys.executable] + sys.argv[1:])\n')
    metadata = {'upstream_script': str(source.resolve()),
                'upstream_sha256': hashlib.sha256(original.encode()).hexdigest(),
                'guarded_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'vendor_head': subprocess.check_output(['git', '-C', str(source.parents[2]), 'rev-parse', 'HEAD'], text=True).strip(),
                'prefill_gpus': [2, 3], 'decode_gpus': [4, 5],
                'ports': [30300, 30301, 30302, 9300, 9301]}
    metadata.update(snapshot(source.parents[2], output))
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    prepare(a.source, a.output)
