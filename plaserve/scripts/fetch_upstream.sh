#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
revision="${1:-main}"
if [[ "$revision" == -* ]]; then
  printf 'Revision must not begin with a dash.\n' >&2
  exit 2
fi
destination="$project_root/vendor/LAPS"
if [[ -e "$destination" ]]; then
  printf 'Destination already exists: %s. Leaving it unchanged.\n' "$destination" >&2
  exit 1
fi
mkdir -p "$project_root/vendor"
staging_dir="$(mktemp -d "$project_root/vendor/.LAPS-fetch.XXXXXX")"
trap 'rm -rf -- "$staging_dir"' EXIT
git clone --no-checkout https://github.com/Jianshu-She/LAPS.git "$staging_dir"
if ! commit="$(git -C "$staging_dir" rev-parse --verify --end-of-options "${revision}^{commit}" 2>/dev/null)"; then
  commit="$(git -C "$staging_dir" rev-parse --verify --end-of-options "origin/${revision}^{commit}")"
fi
git -C "$staging_dir" checkout --detach "$commit"
mv -- "$staging_dir" "$destination"
git -C "$destination" rev-parse HEAD > "$project_root/vendor/LAPS_REVISION.txt"
printf 'Fetched upstream into %s; revision recorded in vendor/LAPS_REVISION.txt\n' "$destination"
