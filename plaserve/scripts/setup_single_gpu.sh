#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.12}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Install Python 3.12 with venv, or set PYTHON_BIN to Python 3.10-3.13.\n' >&2
  exit 1
fi
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 13), "Use Python 3.10-3.13 for the pinned GPU environment"'
nvidia-smi
"$python_bin" -m venv "$project_root/.venv-gpu"
"$project_root/.venv-gpu/bin/python" -m pip install --upgrade pip
"$project_root/.venv-gpu/bin/python" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
"$project_root/.venv-gpu/bin/python" -m pip install -r "$project_root/requirements-gpu.txt"
"$project_root/.venv-gpu/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); print("compute capability", torch.cuda.get_device_capability(0)); print("torch", torch.__version__, "CUDA", torch.version.cuda)'
printf 'Setup complete. From the PLAserve root run: .venv-gpu/bin/python -m plaserve gpu --require-graphs\n'
