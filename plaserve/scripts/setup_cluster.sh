#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-h200}"
case "$profile" in
  h200|h100-4) default_revision=main ;;
  a100-2) default_revision=a100-eval ;;
  *) printf 'Profiles: h200, h100-4, a100-2\n' >&2; exit 2 ;;
esac
revision="${2:-$default_revision}"
env_name="${PLASERVE_ENV:-plaserve}"
command -v conda >/dev/null || { printf 'Load your cluster conda module first.\n' >&2; exit 1; }
command -v git >/dev/null
command -v nvcc >/dev/null || { printf 'Load your cluster CUDA toolkit module first.\n' >&2; exit 1; }
nvidia-smi
if [[ ! -d "$project_root/vendor/LAPS" ]]; then
  bash "$project_root/scripts/fetch_upstream.sh" "$revision"
else
  expected_commit="$(git -C "$project_root/vendor/LAPS" rev-parse --verify --end-of-options "${revision}^{commit}" 2>/dev/null || git -C "$project_root/vendor/LAPS" rev-parse --verify --end-of-options "origin/${revision}^{commit}")"
  actual_commit="$(git -C "$project_root/vendor/LAPS" rev-parse HEAD)"
  if [[ "$expected_commit" != "$actual_commit" ]]; then
    printf 'Existing upstream revision differs from requested revision. Use a separate PLAserve checkout for another profile.\n' >&2
    exit 1
  fi
fi
(
  cd "$project_root/vendor/LAPS"
  ENV_NAME="$env_name" bash install.sh
)
if [[ "$profile" != h200 ]]; then
  conda run -n "$env_name" python -m pip install nixl
fi
printf 'Activate with: conda activate %s\n' "$env_name"
printf 'From the PLAserve root run: python -m plaserve cluster --profile %s\n' "$profile"
