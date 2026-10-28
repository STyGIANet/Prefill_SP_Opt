#!/usr/bin/env bash
set -uo pipefail
failed=0
for tool in python3 git conda nvcc nvidia-smi; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf 'FOUND %s: %s\n' "$tool" "$(command -v "$tool")"
  else
    printf 'MISSING %s\n' "$tool"
    failed=1
  fi
done
if command -v python3 >/dev/null 2>&1; then
  python3 --version
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || failed=1
fi
exit "$failed"
