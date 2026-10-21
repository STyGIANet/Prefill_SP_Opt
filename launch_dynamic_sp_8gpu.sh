#!/usr/bin/env bash

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python_bin="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    python_bin="${CONDA_PREFIX}/bin/python"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
    python_bin="${repo_root}/.venv/bin/python"
else
    python_bin="python3"
fi
model="NousResearch/Meta-Llama-3.1-8B-Instruct"
dataset="emozilla/pg19"
dataset_config="default"
split="train"
num_prompts=32
input_tokens=8192
chunk_tokens=1024
max_local_tokens=65536
dtype="bfloat16"
attention_backend="gathered"
scheduler="greedy"
workload_dir=""
max_running_requests=32
max_batch_tokens=32768
offered_load=8192
arrival_seed=0
arrival_pattern="poisson"
output_dir=""
prepare_only=0
verbose=0
verify_flag="--verify"
prepare_args=()
runner_args=()

# SP runs prefill only; native DP returns one token with no subsequent decode steps.
# Model weights must fit on each worker.
usage() {
    cat <<'EOF'
Usage: ./launch_dynamic_sp_8gpu.sh [options] [-- runner-options...]

Options:
  --python PATH          Python in an installed SGLang environment
  --model ID_OR_PATH     Native Llama checkpoint (default: Llama 3.1 8B)
  --model-revision REV   Model revision resolved to a commit before downloading
  --dataset ID           Hugging Face dataset ID
  --dataset-config NAME Dataset configuration
  --dataset-revision REV Dataset revision resolved to a commit
  --split NAME           Dataset split (default: train)
  --text-column NAME     Dataset text column (default: text)
  --num-prompts N        Number of independent prompts (default: 32)
  --input-tokens N       Tokens per prompt (default: 8192)
  --chunk-tokens N       Prefill tokens per step (default: 1024)
  --max-local-tokens N   Committed KV capacity per worker (default: 65536)
  --attention-backend NAME  gathered (default) or streaming
  --dtype TYPE          float32, bfloat16 (default), or float16
  --offered-load R      Expected input tokens/second (default: 8192)
  --arrival-seed N      Reproducible arrival RNG seed (default: 0)
  --arrival-pattern P   poisson (default) or burst for saturation tests
  --scheduler NAME      greedy (default), sglang-dp, or module:factory
  --max-running-requests N  Resident requests: global SP, per-worker DP (default: 32)
  --max-batch-tokens N   SP new-token budget per step (default: 32768)
  --seed N              Seed for optional trace export (default: 0)
  --output-dir PATH     Fresh directory for workload, manifest, logs, and results
  --workload-dir PATH   Reuse prepared workload and manifest for a matched comparison
  --prepare-only        Download and prepare data without launching workers
  --no-verify           Disable the SP unsharded reference comparison
  --verbose             Also show model-loading and library diagnostics
  -h, --help            Show this help

Install first: ./install_sglang.sh
Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./launch_dynamic_sp_8gpu.sh
EOF
}

while (($#)); do
    case "$1" in
        --workload-dir|--python|--model|--dataset|--dataset-config|--split|--num-prompts|--input-tokens|--chunk-tokens|--max-local-tokens|--output-dir|--model-revision|--dataset-revision|--text-column|--seed|--dtype|--attention-backend|--scheduler|--max-running-requests|--max-batch-tokens|--offered-load|--arrival-seed|--arrival-pattern)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2; exit 2
            fi
            case "$1" in
                --workload-dir) workload_dir="$2" ;;
                --python) python_bin="$2" ;;
                --model) model="$2" ;;
                --dataset) dataset="$2" ;;
                --dataset-config) dataset_config="$2" ;;
                --split) split="$2" ;;
                --num-prompts) num_prompts="$2" ;;
                --input-tokens) input_tokens="$2" ;;
                --chunk-tokens) chunk_tokens="$2" ;;
                --max-local-tokens) max_local_tokens="$2" ;;
                --dtype) dtype="$2" ;;
                --attention-backend) attention_backend="$2" ;;
                --scheduler) scheduler="$2" ;;
                --offered-load) offered_load="$2" ;;
                --arrival-seed) arrival_seed="$2" ;;
                --arrival-pattern) arrival_pattern="$2" ;;
                --max-running-requests) max_running_requests="$2" ;;
                --max-batch-tokens) max_batch_tokens="$2" ;;
                --output-dir) output_dir="$2" ;;
                *) prepare_args+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --prepare-only) prepare_only=1; shift ;;
        --no-verify) verify_flag="--no-verify"; shift ;;
        --verbose) verbose=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; runner_args=("$@"); break ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$attention_backend" in
    gathered|streaming) ;;
    *) printf 'Unsupported attention backend: %s\n' "$attention_backend" >&2; exit 2 ;;
esac

case "$dtype" in
    float32|bfloat16|float16) ;;
    *) printf 'Unsupported dtype: %s\n' "$dtype" >&2; exit 2 ;;
esac

for value in "$num_prompts" "$input_tokens" "$chunk_tokens" "$max_local_tokens" "$max_running_requests" "$max_batch_tokens"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Prompt and token counts must be positive integers: %s\n' "$value" >&2
        exit 2
    fi
done
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
if ((!prepare_only)); then
    "$python_bin" - <<'PY'
import torch
if torch.cuda.device_count() < 8:
    raise SystemExit(f"Eight visible CUDA GPUs are required; found {torch.cuda.device_count()}")
print(f"[dynamic-sp] Found {torch.cuda.device_count()} visible GPUs; using eight workers")
PY
fi
if [[ -z "$output_dir" ]]; then
    mkdir -p "${repo_root}/python/sglang/srt/dynamic_sp/work"
    output_dir="$(mktemp -d "${repo_root}/python/sglang/srt/dynamic_sp/work/dynamic-sp-8gpu.XXXXXXXX")"
fi
mkdir -p "$output_dir"
show_logs() {
    if ((verbose)); then
        cat
    else
        sed -u -n '/^\[dynamic-sp\]/p'
    fi
}
if [[ -n "$workload_dir" ]]; then
    "$python_bin" - "$workload_dir" "$output_dir" <<'PYCODE'
import pathlib, shutil, sys
source, target = map(pathlib.Path, sys.argv[1:])
for name in ("workload.json", "manifest.json"):
    if (target / name).exists():
        raise SystemExit("Choose a fresh output directory")
    shutil.copyfile(source / name, target / name)
PYCODE
    if [[ $? -ne 0 ]]; then exit 1; fi
else
printf '[dynamic-sp] Preparing model and data; logs: %s\n' "$output_dir"
if ! (set -o pipefail; "$python_bin" -m sglang.srt.dynamic_sp.prepare_workload \
    --model "$model" --dataset "$dataset" --dataset-config "$dataset_config" \
    --split "$split" --num-prompts "$num_prompts" --input-tokens "$input_tokens" \
    --chunk-tokens "$chunk_tokens" --offered-load "$offered_load" --arrival-seed "$arrival_seed" \
    --arrival-pattern "$arrival_pattern" --output-dir "$output_dir" "${prepare_args[@]}" \
    2>&1 | tee "${output_dir}/prepare.log" | show_logs); then
    printf 'Workload preparation failed. Diagnostic log: %s/prepare.log\n' "$output_dir" >&2
    tail -n 80 "${output_dir}/prepare.log" >&2
    exit 1
fi
fi
if ((prepare_only)); then
    exit 0
fi
model_path="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "${output_dir}/manifest.json")"
printf '[dynamic-sp] Loading model; full diagnostics: %s/run.log\n' "$output_dir"
if [[ "$scheduler" == "sglang-dp" ]]; then
    if ! (set -o pipefail; "$python_bin" -m sglang.srt.dynamic_sp.dp_baseline \
        --model-path "$model_path" --dtype "$dtype" --dp-size 8 \
        --workload "${output_dir}/workload.json" --output "${output_dir}/results.json" \
        --chunk-size "$chunk_tokens" --max-local-tokens "$max_local_tokens" \
        --max-running-requests "$max_running_requests" "${runner_args[@]}" \
        2>&1 | tee "${output_dir}/run.log" | show_logs); then
        tail -n 80 "${output_dir}/run.log" >&2
        exit 1
    fi
    exit 0
fi
if ! (set -o pipefail; "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=8 \
    -m sglang.srt.dynamic_sp --model-path "$model_path" --dtype "$dtype" --attention-backend "$attention_backend" \
    --workload "${output_dir}/workload.json" --chunk-size "$chunk_tokens" \
    --scheduler "$scheduler" --max-running-requests "$max_running_requests" --max-batch-tokens "$max_batch_tokens" \
    --max-local-tokens "$max_local_tokens" \
    --output "${output_dir}/results.json" "$verify_flag" "${runner_args[@]}" \
    2>&1 | tee "${output_dir}/run.log" | show_logs); then
    printf 'Dynamic SP run failed. Diagnostic log: %s/run.log\n' "$output_dir" >&2
    tail -n 80 "${output_dir}/run.log" >&2
    exit 1
fi
