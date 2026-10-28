#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$project_root/dist/PLAserve.tar.gz}"
mkdir -p -- "$(dirname -- "$output")"
tar --exclude='__pycache__' --exclude='*.pyc' \
  --transform='s,^,PLAserve/,' -czf "$output" -C "$project_root" \
  README.md pyproject.toml requirements-gpu.txt .gitignore \
  plaserve scripts cluster configs docs examples tests
printf 'Created %s (extracts into PLAserve/)\n' "$output"
