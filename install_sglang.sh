#!/usr/bin/env bash

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_dir="${VIRTUAL_ENV:-${CONDA_PREFIX:-${repo_root}/.venv}}"
base_python="${PYTHON:-python3}"
extras=()
installer_args=()
dry_run=0

usage() {
    cat <<'EOF'
Usage: ./install_sglang.sh [options] [-- installer-options...]

Options:
  --venv PATH      Select an existing environment or create one at PATH.
  --python PATH   Base Python for a new environment (default: python3).
  --extra NAME    Include a pyproject optional dependency group (repeatable).
  --dry-run       Print commands without creating or modifying an environment.
  -h, --help      Show this help.

Examples:
  ./install_sglang.sh
  ./install_sglang.sh --venv /path/to/venv
  ./install_sglang.sh --venv .venv --python python3.12 --extra ray

EOF
}

while (($#)); do
    case "$1" in
        --venv|--python|--extra)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --venv) env_dir="$2" ;;
                --python) base_python="$2" ;;
                --extra)
                    if [[ ! "$2" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
                        printf 'Invalid extra name: %s\n' "$2" >&2
                        exit 2
                    fi
                    extras+=("$2")
                    ;;
            esac
            shift 2
            ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; installer_args=("$@"); break ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

run() {
    if ((dry_run)); then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ ! -f "${repo_root}/python/pyproject.toml" ]]; then
    printf 'Missing python/pyproject.toml in %s\n' "$repo_root" >&2
    exit 1
fi

target_python="${env_dir}/bin/python"
if [[ ! -x "$target_python" ]]; then
    if [[ -d "$env_dir" ]] && [[ -n "$(ls -A -- "$env_dir")" ]]; then
        printf 'Refusing to overwrite a nonempty directory without bin/python: %s\n' "$env_dir" >&2
        exit 1
    fi
    if command -v uv >/dev/null 2>&1; then
        run uv venv --python "$base_python" "$env_dir"
    else
        run "$base_python" -m venv "$env_dir"
    fi
fi

if ((!dry_run)); then
    "$target_python" - <<'PY'
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("This checkout requires Python 3.11 or newer")
if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / "conda-meta").is_dir():
    raise SystemExit("Select a virtual or Conda environment, not system Python")
PY
fi

if [[ "${SGLANG_BUILD_RUST_EXTS:-all}" != "none" ]]; then
    if ((dry_run)); then
        printf 'Will select and validate the Rust toolchain from %s\n' "${repo_root}/rust/rust-toolchain.toml"
    else
        required_rust="$("$target_python" - "${repo_root}/rust/rust-toolchain.toml" <<'PY'
import sys
import tomllib
from pathlib import Path
print(tomllib.loads(Path(sys.argv[1]).read_text())["toolchain"]["channel"])
PY
)"
        if command -v rustup >/dev/null 2>&1; then
            selected_rust="${RUSTUP_TOOLCHAIN:-$required_rust}"
            if ! rustup run "$selected_rust" rustc --version >/dev/null 2>&1; then
                rustup toolchain install "$selected_rust" --profile minimal
            fi
            export RUSTUP_TOOLCHAIN="$selected_rust"
            rust_bin="$(dirname -- "$(rustup which --toolchain "$selected_rust" cargo)")"
            export PATH="${rust_bin}:${PATH}"
        fi
        if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
            printf 'Rust %s or newer is required. Install it with rustup before retrying.\n' "$required_rust" >&2
            exit 1
        fi
        rust_version="$(rustc --version)"
        "$target_python" - "$required_rust" "$rust_version" <<'PY'
import re
import sys

def version(text):
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        raise SystemExit(f"Cannot determine a Rust version from {text!r}")
    return tuple(int(part or 0) for part in match.groups())

if version(sys.argv[2]) < version(sys.argv[1]):
    raise SystemExit(f"{sys.argv[2]} is too old; this checkout requires Rust {sys.argv[1]} or newer")
print(f"Building Rust extensions with {sys.argv[2]}")
PY
    fi
fi

package="${repo_root}/python"
if ((${#extras[@]})); then
    extra_list="$(IFS=,; printf '%s' "${extras[*]}")"
    package+="[${extra_list}]"
fi

if command -v uv >/dev/null 2>&1; then
    run uv pip install --python "$target_python" --prerelease=allow --editable "$package" "${installer_args[@]}"
else
    if ((!dry_run)) && ! "$target_python" -m pip --version >/dev/null 2>&1; then
        "$target_python" -m ensurepip --upgrade
    fi
    run "$target_python" -m pip install --pre --editable "$package" "${installer_args[@]}"
fi

if ((!dry_run)); then
    "$target_python" - "$repo_root" <<'PY'
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve() / "python" / "sglang" / "__init__.py"
spec = importlib.util.find_spec("sglang")
if spec is None or Path(spec.origin).resolve() != expected:
    raise SystemExit(f"sglang resolves outside this checkout: {spec}")
direct = json.loads(importlib.metadata.distribution("sglang").read_text("direct_url.json") or "{}")
if not direct.get("dir_info", {}).get("editable"):
    raise SystemExit("sglang installation is not marked editable")
print(f"Verified editable SGLang source: {expected.parent}")
PY
    printf 'Python environment: %s\n' "$target_python"
fi
