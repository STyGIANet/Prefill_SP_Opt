#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${VIRTUAL_ENV:-${CONDA_PREFIX:-$root/.venv}}/bin/python"
plot_python="${PLOT_PYTHON:-$python_bin}"
if [[ "${1:-}" == "--plot-only" ]]; then
  [[ $# == 2 ]] || { echo "Usage: $0 --plot-only OUTPUT_DIRECTORY" >&2; exit 2; }
  exec "${PLOT_PYTHON:-python3}" "$root/python/sglang/srt/dynamic_sp/plot_roofline.py" "$2/samples.json"
fi
output="$root/python/sglang/srt/dynamic_sp/work/roofline-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$output"
export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi topo -m > "$output/topology.txt"
printf '[roofline] Output: %s\n' "$output"
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m sglang.srt.dynamic_sp.roofline_probe --output "$output/samples.json" "$@" 2>&1 | tee "$output/run.log"
if "$plot_python" -c 'import matplotlib, numpy' 2>/dev/null; then
  "$plot_python" "$root/python/sglang/srt/dynamic_sp/plot_roofline.py" "$output/samples.json"
else
  printf '[roofline] Measurements saved. Plot with an existing Python containing matplotlib and numpy:\nPLOT_PYTHON=/path/to/python %q --plot-only %q\n' "$0" "$output"
fi
