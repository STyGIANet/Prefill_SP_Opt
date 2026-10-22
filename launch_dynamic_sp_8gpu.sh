#!/usr/bin/env bash
set -euo pipefail
original_args=("$@")

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
# --model gradientai/Llama-3-8B-Instruct-Gradient-1048k
# --model gradientai/Llama-3-8B-Instruct-262k
# --model NousResearch/Meta-Llama-3.1-8B-Instruct
# --eval-dataset emozilla/pg19 --dataset-config default --text-column text --eval-split test
# --eval-dataset ccdv/arxiv-summarization --dataset-config document --text-column article --eval-split test
# --eval-dataset ccdv/govreport-summarization --dataset-config document --text-column report --eval-split test
model="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
dataset="emozilla/pg19"
length_mode="natural"
dataset_config="default"
split="train"
num_prompts=all
input_tokens=all
chunk_tokens=1024
max_local_tokens=""
dtype="bfloat16"
attention_backend="ring"
migration_overlap="--migration-overlap"
scheduler="all"
matrix=0
warmup_requests=8
warmup_tokens=8192
execution_mode="async"
workload_dir=""
max_running_requests=32
max_batch_tokens=32768
offered_load=8192
arrival_seed=0
arrival_pattern="poisson"
output_dir=""
prepare_only=0
verbose=0
verify_flag="--no-verify"
prepare_args=()
runner_args=()
loongserve_calibration_grid="lean"
loongserve_calibration_concurrency="parallel"
loongserve_profile=""
recalibrate_loongserve=0
recalibrate_tetris=0
loongserve_samples=""
loongserve_max_prefill_seconds=""
loongserve_calibration_lengths=""
tetris_profile=""
tetris_samples=""
fit_dataset=""
fit_dataset_config="default"
fit_split="train"
fit_num_prompts=256
fit_input_tokens=all
fit_length_mode="natural"
fit_args=()
tetris_loads="4096 8192 16384 25000 32768"
tetris_objective="mean"
tetris_simulation_requests=1000
kv_transport="targeted"
shared_attention="auto"
tetris_fit_method="nonnegative"

# SP runs prefill only; native DP returns one token with no subsequent decode steps.
# Model weights must fit on each worker.
usage() {
    cat <<'EOF'
Usage: ./launch_dynamic_sp_8gpu.sh [options] [-- runner-options...]

Options:
  --python PATH          Python in an installed SGLang environment
  --model ID_OR_PATH     Native Llama checkpoint (default: Gradient Llama 3 8B 1048K)
  --model-revision REV   Model revision resolved to a commit before downloading
  --eval-dataset ID      Evaluation dataset ID (--dataset is an alias)
  --dataset-config NAME Dataset configuration
  --dataset-revision REV Dataset revision resolved to a commit
  --eval-split NAME      Evaluation partition, e.g. test (--split alias; default: train)
  --text-column NAME     Dataset text column (default: text)
  --num-prompts N        Evaluation prompts: all or a positive count (default: all)
  --input-tokens N       all (default) preserves full records; integer sets an explicit cap
  --length-mode MODE     natural (default) or fixed evaluation lengths
  --chunk-tokens N       Prefill tokens per step (default: 1024)
  --max-local-tokens N   KV capacity per worker (default: SP 131072; DP 600000)
  --attention-backend NAME  ring (default), gathered, or streaming
  --no-migration-overlap    Disable layer-wise KV migration overlap
  --tetris-fit-method NAME  nonnegative (default) or least-squares
  --dtype TYPE          float32, bfloat16 (default), or float16
  --offered-load R      Expected input tokens/second (default: 8192)
  --arrival-seed N      Reproducible arrival RNG seed (default: 0)
  --arrival-pattern P   poisson (default) or burst for saturation tests
  --execution-mode MODE async (default) or sync SP execution
  --scheduler NAME      all SP schedulers (default), greedy, tetris, loongserve, or module:factory
  --recalibrate-loongserve  Explicitly run new GPU calibration (default: reuse compatible profile)
  --loongserve-profile PATH  Reuse a frozen LoongServe batch latency profile
  --loongserve-samples PATH  Fit existing full-batch GPU measurements
  --loongserve-max-prefill-seconds S  Admission limit (profile value; new profiles: 0.5s)
  --loongserve-calibration-concurrency parallel|serial  Disjoint profiling groups (default: parallel)
  --loongserve-calibration-grid lean|full  Sparse default or exhaustive profiling
  --loongserve-calibration-lengths "N ..."  Absolute lengths override (default: lengths scale with SP degree)
  --recalibrate-tetris  Explicitly recalibrate and select thresholds (default: reuse compatible profile)
  --tetris-profile PATH  Reuse a frozen profile; skips all Tetris fitting
  --fit-dataset ID       Independent dataset for Tetris threshold selection
  --fit-split NAME       Fitting partition (default: train)
  --fit-dataset-config NAME  Fitting dataset configuration (default: default)
  --fit-dataset-revision REV  Fitting dataset revision (default: main)
  --fit-text-column NAME Fitting text column (default: text)
  --fit-num-prompts N    Fitting records (default: 256)
  --fit-input-tokens N   Fitting token cap (default: all, no truncation)
  --fit-length-mode MODE natural (default) or fixed fitting lengths
  --tetris-samples PATH  Reuse hardware calibration samples when fitting
  --tetris-loads "R ..." Input-token/s loads (default: "4096 8192 16384 25000 32768")
  --tetris-objective NAME mean (default), p50, or p95 per-prompt TTFT
  --tetris-simulation-requests N  Simulated requests per load (default: 1000)
  --kv-transport NAME    targeted (default) or all_gather
  --shared-attention NAME auto (default) or sdpa
  --max-running-requests N  Resident requests: global SP, per-worker DP (default: 32)
  --max-batch-tokens N   SP new-token budget per step (default: 32768)
  --seed N              Seed for optional trace export (default: 0)
  --output-dir PATH     Fresh directory for workload, manifest, logs, and results
  --workload-dir PATH   Reuse prepared workload and manifest for a matched comparison
  --prepare-only        Prepare evaluation data only, without GPU work or fitting
  --datasets "P ..."     Dataset presets: pg19 arxiv govreport (default: all three)
  --attention-backends "B ..."  Matrix backends (default: ring gathered streaming)
  --kv-transports "T ..." Matrix transports (default: targeted all_gather)
  --repeats N           Measured trials per combination (default: 1)
  --progress-interval S Progress refresh seconds (default: 10; 0 disables)
  --warmup-tokens N     Total sequential warmup token budget (default: 8192)
  --warmup-requests N   Same-process warmup prompts per trial (default: 8)
  --verify              Enable reference checking; affects measured performance
  --no-verify           Disable reference checking (default)
  --verbose             Also show model-loading and library diagnostics
  -h, --help            Show this help

Install first: ./install_sglang.sh
Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./launch_dynamic_sp_8gpu.sh
Tetris: ./launch_dynamic_sp_8gpu.sh --scheduler tetris \
  --fit-dataset emozilla/pg19 --fit-split train \
  --eval-dataset emozilla/pg19 --eval-split test --no-verify

For a single Tetris run, provide --fit-dataset or --tetris-profile. The matrix
fits once on PG19/train by default and freezes that profile across all trials. Fitting never
reads evaluation lengths. --dataset-config, --dataset-revision, --text-column,
--num-prompts, --input-tokens, and arrival options apply only to evaluation.
Fitting artifacts are saved under the new run's tetris/ directory.
EOF
}

while (($#)); do
    case "$1" in
        --eval-dataset) set -- --dataset "${@:2}" ;;
        --eval-split) set -- --split "${@:2}" ;;
        --eval-dataset-config) set -- --dataset-config "${@:2}" ;;
        --eval-dataset-revision) set -- --dataset-revision "${@:2}" ;;
        --eval-text-column) set -- --text-column "${@:2}" ;;
        --eval-num-prompts) set -- --num-prompts "${@:2}" ;;
        --eval-input-tokens) set -- --input-tokens "${@:2}" ;;
        --eval-length-mode) set -- --length-mode "${@:2}" ;;
    esac
    case "$1" in
        --loongserve-calibration-concurrency|--loongserve-calibration-grid|--loongserve-profile|--loongserve-samples|--loongserve-max-prefill-seconds|--loongserve-calibration-lengths|--progress-interval|--datasets|--attention-backends|--kv-transports|--repeats|--warmup-tokens|--warmup-requests|--tetris-fit-method|--execution-mode|--workload-dir|--python|--model|--dataset|--dataset-config|--split|--num-prompts|--input-tokens|--chunk-tokens|--max-local-tokens|--output-dir|--model-revision|--dataset-revision|--text-column|--seed|--dtype|--attention-backend|--scheduler|--max-running-requests|--max-batch-tokens|--offered-load|--arrival-seed|--arrival-pattern|--length-mode|--tetris-profile|--fit-dataset|--fit-split|--fit-dataset-config|--fit-dataset-revision|--fit-text-column|--fit-num-prompts|--fit-input-tokens|--fit-length-mode|--tetris-samples|--tetris-loads|--tetris-objective|--tetris-simulation-requests|--kv-transport|--shared-attention)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing value for %s\n' "$1" >&2; exit 2
            fi
            case "$1" in
                --loongserve-calibration-concurrency) loongserve_calibration_concurrency="$2" ;;
                --loongserve-calibration-grid) loongserve_calibration_grid="$2" ;;
                --loongserve-profile) loongserve_profile="$2" ;;
                --loongserve-samples) loongserve_samples="$2" ;;
                --loongserve-max-prefill-seconds) loongserve_max_prefill_seconds="$2" ;;
                --loongserve-calibration-lengths) loongserve_calibration_lengths="$2" ;;
                --datasets|--attention-backends|--kv-transports|--repeats) matrix=1 ;;
                --progress-interval) runner_args+=(--progress-interval "$2") ;;
                --warmup-tokens) warmup_tokens="$2" ;;
                --warmup-requests) warmup_requests="$2" ;;
                --tetris-fit-method) tetris_fit_method="$2" ;;
                --execution-mode) execution_mode="$2" ;;
                --workload-dir) workload_dir="$2" ;;
                --length-mode) length_mode="$2" ;;
                --tetris-profile) tetris_profile="$2" ;;
                --fit-dataset) fit_dataset="$2" ;;
                --fit-split) fit_split="$2" ;;
                --fit-dataset-config) fit_dataset_config="$2" ;;
                --fit-dataset-revision) fit_args+=(--dataset-revision "$2") ;;
                --fit-text-column) fit_args+=(--text-column "$2") ;;
                --fit-num-prompts) fit_num_prompts="$2" ;;
                --fit-input-tokens) fit_input_tokens="$2" ;;
                --fit-length-mode) fit_length_mode="$2" ;;
                --tetris-samples) tetris_samples="$2" ;;
                --tetris-loads) tetris_loads="$2" ;;
                --tetris-objective) tetris_objective="$2" ;;
                --tetris-simulation-requests) tetris_simulation_requests="$2" ;;
                --kv-transport) kv_transport="$2" ;;
                --shared-attention) shared_attention="$2" ;;
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
        --allow-profile-backend-mismatch) runner_args+=("$1"); shift ;;
        --recalibrate-tetris) recalibrate_tetris=1; shift ;;
        --recalibrate-loongserve) recalibrate_loongserve=1; shift ;;
        --prepare-only) prepare_only=1; shift ;;
        --migration-overlap|--no-migration-overlap) migration_overlap="$1"; shift ;;
        --no-verify) verify_flag="--no-verify"; shift ;;
        --verify) verify_flag="--verify"; shift ;;
        --verbose) verbose=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; runner_args+=("$@"); break ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$max_local_tokens" ]]; then
    if [[ "$scheduler" == sglang-dp ]]; then
        max_local_tokens=600000
    else
        max_local_tokens=131072
    fi
fi

export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
if ((!prepare_only)) && { [[ "$scheduler" == all ]] || ((matrix)); }; then
    exec "$python_bin" -m sglang.srt.dynamic_sp.benchmark --launcher "$repo_root/launch_dynamic_sp_8gpu.sh" "${original_args[@]}"
fi
if [[ ! "$warmup_requests" =~ ^[0-9]+$ ]]; then
    printf 'warmup-requests must be nonnegative\n' >&2; exit 2
fi
if [[ "$execution_mode" == sync && "$warmup_requests" != 0 ]]; then
    printf 'Use --warmup-requests 0 for the legacy sync runner\n' >&2; exit 2
fi
case "$execution_mode" in
    async|sync) ;;
    *) printf "Execution mode must be async or sync\n" >&2; exit 2 ;;
esac

case "$attention_backend" in
    ring|gathered|streaming) ;;
    *) printf 'Unsupported attention backend: %s\n' "$attention_backend" >&2; exit 2 ;;
esac

case "$dtype" in
    float32|bfloat16|float16) ;;
    *) printf 'Unsupported dtype: %s\n' "$dtype" >&2; exit 2 ;;
esac

for value in "$fit_num_prompts" "$tetris_simulation_requests" "$chunk_tokens" "$max_local_tokens" "$max_running_requests" "$max_batch_tokens"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Prompt and token counts must be positive integers: %s\n' "$value" >&2
        exit 2
    fi
done
for value in "$input_tokens" "$fit_input_tokens"; do
    if [[ "$value" != all && ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Token limits must be all or positive integers\n' >&2; exit 2
    fi
done
if [[ "$num_prompts" != all && ! "$num_prompts" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Evaluation prompt count must be all or a positive integer: %s\n' "$num_prompts" >&2
    exit 2
fi
case "$length_mode:$fit_length_mode" in
    fixed:fixed|fixed:natural|natural:fixed|natural:natural) ;;
    *) printf 'Length modes must be fixed or natural\n' >&2; exit 2 ;;
esac
case "$kv_transport:$shared_attention" in
    targeted:auto|targeted:sdpa|all_gather:auto|all_gather:sdpa) ;;
    *) printf 'Invalid KV transport or shared attention choice\n' >&2; exit 2 ;;
esac
case "$tetris_objective" in
    mean|p50|p95) ;;
    *) printf 'Tetris objective must be mean, p50, or p95\n' >&2; exit 2 ;;
esac
if [[ "$scheduler" == tetris ]] && ((!prepare_only)); then
    if [[ -z "$tetris_profile" && -z "$fit_dataset" ]]; then
        printf 'Tetris requires --fit-dataset (independent fitting data) or --tetris-profile (frozen profile).\n' >&2
        exit 2
    fi
    if [[ -n "$tetris_profile" && ( -n "$fit_dataset" || -n "$tetris_samples" ) ]]; then
        printf 'Choose either a frozen --tetris-profile or fitting options, not both.\n' >&2
        exit 2
    fi
fi
for path in "$tetris_profile" "$tetris_samples" "$loongserve_profile" "$loongserve_samples"; do
    if [[ -n "$path" && ! -r "$path" ]]; then
        printf 'Cannot read scheduler profile/samples: %s\n' "$path" >&2; exit 2
    fi
done
if [[ -n "$workload_dir" ]]; then
    for name in workload.json manifest.json; do
        if [[ ! -r "$workload_dir/$name" ]]; then
            printf 'Missing evaluation file: %s/%s\n' "$workload_dir" "$name" >&2; exit 2
        fi
    done
fi
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
    --split "$split" --num-prompts "$num_prompts" --input-tokens "$input_tokens" --length-mode "$length_mode" \
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
# Keep all Tetris fitting inputs separate from the evaluation workload.
run_logged() {
    local log_path="$1"
    shift
    if ! (set -o pipefail; "$@" 2>&1 | tee "$log_path" | show_logs); then
        printf 'Stage failed. Diagnostic log: %s\n' "$log_path" >&2
        tail -n 80 "$log_path" >&2
        exit 1
    fi
}
if [[ "$scheduler" == loongserve ]]; then
    if [[ "$execution_mode" != async ]]; then
        printf 'LoongServe requires async execution.\n' >&2; exit 2
    fi
    if [[ -n "$loongserve_profile" && -n "$loongserve_samples" ]]; then
        printf 'Choose a LoongServe profile or samples, not both.\n' >&2; exit 2
    fi
    loongserve_dir="$output_dir/loongserve"
    mkdir -p "$loongserve_dir"
    if ((recalibrate_loongserve)) && [[ -n "$loongserve_profile" || -n "$loongserve_samples" ]]; then
        printf 'Choose recalibration or an existing profile/samples, not both.\n' >&2; exit 2
    fi
    if ((!recalibrate_loongserve)) && [[ -z "$loongserve_profile" && -z "$loongserve_samples" ]]; then
        loongserve_profile="$loongserve_dir/profile.json"
        "$python_bin" -m sglang.srt.dynamic_sp.profile_cache --algorithm loongserve \
            --work-dir "$repo_root/python/sglang/srt/dynamic_sp/work" --output "$loongserve_profile" \
            --model-path "$model_path" --dtype "$dtype" --attention-backend "$attention_backend" \
            --kv-transport "$kv_transport" --shared-attention "$shared_attention" "$migration_overlap"
    fi
    if [[ -z "$loongserve_profile" ]]; then
        if [[ -z "$loongserve_samples" ]]; then
            loongserve_samples="$loongserve_dir/samples.json"
            calibration_length_args=()
            if [[ -n "$loongserve_calibration_lengths" ]]; then
                read -r -a calibration_lengths <<< "$loongserve_calibration_lengths"
                calibration_length_args=(--lengths "${calibration_lengths[@]}")
            fi
            printf '[dynamic-sp] Calibrating LoongServe SIB; log: %s/calibration.log\n' "$loongserve_dir"
            run_logged "$loongserve_dir/calibration.log" \
                "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=8 \
                -m sglang.srt.dynamic_sp.loongserve_calibrate \
                --model-path "$model_path" --dtype "$dtype" --attention-backend "$attention_backend" \
                --kv-transport "$kv_transport" --shared-attention "$shared_attention" "$migration_overlap" \
                --max-local-tokens "$max_local_tokens" --grid "$loongserve_calibration_grid" --concurrency "$loongserve_calibration_concurrency" "${calibration_length_args[@]}" --output "$loongserve_samples"
        fi
        loongserve_profile="$loongserve_dir/profile.json"
        run_logged "$loongserve_dir/fit.log" \
            "$python_bin" -m sglang.srt.dynamic_sp.loongserve_profile \
            --samples "$loongserve_samples" --max-prefill-seconds "${loongserve_max_prefill_seconds:-0.5}" \
            --output "$loongserve_profile"
    fi
    runner_args+=(--loongserve-profile "$loongserve_profile")
    if [[ -n "$loongserve_max_prefill_seconds" ]]; then
        runner_args+=(--loongserve-max-prefill-seconds "$loongserve_max_prefill_seconds")
    fi
fi

if [[ "$scheduler" == tetris ]]; then
    tetris_dir="$output_dir/tetris"
    mkdir -p "$tetris_dir"
    if [[ -n "$tetris_profile" && -n "$tetris_samples" ]] || { ((recalibrate_tetris)) && [[ -n "$tetris_profile" || -n "$tetris_samples" ]]; }; then
        printf 'Choose Tetris recalibration, a profile, or samples, not multiple sources.\n' >&2; exit 2
    fi
    if ((!recalibrate_tetris)) && [[ -z "$tetris_profile" && -z "$tetris_samples" ]]; then
        tetris_profile="$tetris_dir/profile.json"
        cache_args=()
        for arg in "${runner_args[@]}"; do
            if [[ "$arg" == --allow-profile-backend-mismatch ]]; then cache_args+=("$arg"); fi
        done
        "$python_bin" -m sglang.srt.dynamic_sp.profile_cache --algorithm tetris \
            --work-dir "$repo_root/python/sglang/srt/dynamic_sp/work" --output "$tetris_profile" \
            --model-path "$model_path" --dtype "$dtype" --attention-backend "$attention_backend" \
            --execution-mode "$execution_mode" --kv-transport "$kv_transport" \
            --shared-attention "$shared_attention" "$migration_overlap" "${cache_args[@]}"
    fi
    if [[ -z "$tetris_profile" ]]; then
        printf '[dynamic-sp] Preparing independent Tetris fitting dataset: %s/%s\n' "$fit_dataset" "$fit_split"
        run_logged "$tetris_dir/prepare-fit.log" \
            "$python_bin" -m sglang.srt.dynamic_sp.prepare_workload \
            --model "$model_path" --dataset "$fit_dataset" --dataset-config "$fit_dataset_config" \
            --split "$fit_split" --num-prompts "$fit_num_prompts" \
            --input-tokens "$fit_input_tokens" --length-mode "$fit_length_mode" \
            --output-dir "$tetris_dir/fit-workload" "${fit_args[@]}"
        # Reject reused source records before expensive GPU calibration.
        "$python_bin" -m sglang.srt.dynamic_sp.tetris_profile check-data \
            --training-manifest "$tetris_dir/fit-workload/manifest.json" \
            --evaluation-manifest "$output_dir/manifest.json"
        if [[ -z "$tetris_samples" ]]; then
            tetris_samples="$tetris_dir/samples.json"
            printf '[dynamic-sp] Calibrating Tetris latency; log: %s/calibration.log\n' "$tetris_dir"
            run_logged "$tetris_dir/calibration.log" \
                "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=8 \
                -m sglang.srt.dynamic_sp.tetris_calibrate \
                --model-path "$model_path" --dtype "$dtype" --attention-backend "$attention_backend" --execution-mode "$execution_mode" \
                --kv-transport "$kv_transport" --shared-attention "$shared_attention" "$migration_overlap" \
                --max-local-tokens "$max_local_tokens" --output "$tetris_samples"
        fi
        printf '[dynamic-sp] Fitting hardware latency models\n'
        run_logged "$tetris_dir/fit.log" \
            "$python_bin" -m sglang.srt.dynamic_sp.tetris_profile fit \
            --samples "$tetris_samples" --method "$tetris_fit_method" --window-seconds 30 --min-chunk-tokens 1 \
            --output "$tetris_dir/latency.json"
        printf '[dynamic-sp] Selecting Tetris thresholds from fitting data only\n'
        read -r -a load_values <<< "$tetris_loads"
        tetris_profile="$tetris_dir/profile.json"
        run_logged "$tetris_dir/search.log" \
            "$python_bin" -m sglang.srt.dynamic_sp.tetris_profile search \
            --profile "$tetris_dir/latency.json" \
            --workload "$tetris_dir/fit-workload/workload.json" \
            --training-manifest "$tetris_dir/fit-workload/manifest.json" \
            --loads "${load_values[@]}" --requests "$tetris_simulation_requests" \
            --seed 0 --objective "$tetris_objective" --output "$tetris_profile"
    else
        printf '[dynamic-sp] Reusing frozen Tetris profile: %s\n' "$tetris_profile"
    fi
    "$python_bin" -m sglang.srt.dynamic_sp.tetris_profile check-data \
        --profile "$tetris_profile" --evaluation-manifest "$output_dir/manifest.json"
    runner_args+=(--tetris-profile "$tetris_profile")
fi
printf '[dynamic-sp] Loading model; full diagnostics: %s/run.log\n' "$output_dir"
if [[ "$scheduler" == "sglang-dp" ]]; then
    if ! (set -o pipefail; "$python_bin" -m sglang.srt.dynamic_sp.dp_baseline \
        --model-path "$model_path" --dtype "$dtype" --dp-size 8 \
        --workload "${output_dir}/workload.json" --output "${output_dir}/results.json" \
        --chunk-size "$chunk_tokens" --max-local-tokens "$max_local_tokens" \
        --max-running-requests "$max_running_requests" --progress-throughput-target "$offered_load" "${runner_args[@]}" \
        2>&1 | tee "${output_dir}/run.log" | show_logs); then
        tail -n 80 "${output_dir}/run.log" >&2
        exit 1
    fi
    exit 0
fi
if ! (set -o pipefail; "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=8 \
    -m sglang.srt.dynamic_sp --model-path "$model_path" --dtype "$dtype" --attention-backend "$attention_backend" --execution-mode "$execution_mode" \
    --workload "${output_dir}/workload.json" --chunk-size "$chunk_tokens" \
    --scheduler "$scheduler" --max-running-requests "$max_running_requests" --max-batch-tokens "$max_batch_tokens" \
    --max-local-tokens "$max_local_tokens" --kv-transport "$kv_transport" --shared-attention "$shared_attention" "$migration_overlap" \
    --output "${output_dir}/results.json" --warmup-requests "$warmup_requests" --warmup-tokens "$warmup_tokens" --progress-throughput-target "$offered_load" "$verify_flag" "${runner_args[@]}" \
    2>&1 | tee "${output_dir}/run.log" | show_logs); then
    printf 'Dynamic SP run failed. Diagnostic log: %s/run.log\n' "$output_dir" >&2
    tail -n 80 "${output_dir}/run.log" >&2
    exit 1
fi
