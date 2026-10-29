#!/usr/bin/env bash
# Run maxsp, tetris, LAPS, and dynamic prefill-only on GPUs 0..GPUS-1, as
# launch_dynamic_sp_8gpu.sh does by default, then print the tables and the
# slowdown figure. Runs that already have results are kept unless RERUN=1;
# the first failure or any other user's GPU job stops it.
#   GPUS        prefill GPUs (default 4; 4 uses pd_profiles/*_p4, 8 the full profiles)
#   LOADS       offered input tokens/s (default "4096 8192 16384")
#   DATASET     wildchat (default), lmsys, pg19 (8K-token books), pg19_natural (whole
#               books under 40K tokens), or longbench_v2 (contexts capped at 32K);
#               a missing workload is prepared first
#   NUM_PROMPTS prompts per workload (default: the dataset preset's)
#   MODEL       default qwen3_8b
#   SCHEDULERS  default "maxsp tetris laps dynamic"
#   SUFFIX      appended to every output name, e.g. _v2 (default none)
#   LENGTH_BINS prompt-length bin edges of the TTFT table (default "1024 4096 8192 16384")
#   ROOT        results root (default work/dataset_sweep_prefill[_<GPUS>gpu]);
#               workloads with another prompt count than requested are prepared here
# Arguments after the script name go to every run_dynamic_sp.py call.
set -euo pipefail
cd ~/Prefill_SP_Opt && source .venv/bin/activate

LOADS=${LOADS:-4096 8192 16384}
DATASET=${DATASET:-wildchat}
MODEL=${MODEL:-qwen3_8b}
SCHEDULERS=${SCHEDULERS:-maxsp tetris laps dynamic}
SUFFIX=${SUFFIX:-}
GPUS=${GPUS:-4}
W=python/sglang/srt/dynamic_sp/work
if [ "$GPUS" = 8 ]; then
  ROOT=${ROOT:-$W/dataset_sweep_prefill} PROFILES=$W/tetris_fix_profiles
else
  ROOT=${ROOT:-$W/dataset_sweep_prefill_${GPUS}gpu} PROFILES=$W/pd_profiles
fi
FIGURE=$W/figures/ttft_slowdown_${ROOT#$W/dataset_sweep_}

for load in $LOADS; do
  for scheduler in $SCHEDULERS; do
    name=$scheduler$SUFFIX
    if [ -z "${RERUN:-}" ] && [ -f "$ROOT/$DATASET/$MODEL/l${load}_s0/$name/results.json" ]; then
      echo "[prefill-only] skip $name l$load: results exist"
      continue
    fi
    python python/sglang/srt/dynamic_sp/run_dynamic_sp.py --prefill-only --gpus "$GPUS" \
      --root "$ROOT" --dataset "$DATASET" ${NUM_PROMPTS:+--num-prompts "$NUM_PROMPTS"} \
      --model "$MODEL" --load "$load" --scheduler "$scheduler" --name "$name" "$@"
  done
done

read -r -a loads <<< "$LOADS"
schedulers=() labels=()
for scheduler in $SCHEDULERS; do
  schedulers+=("$scheduler$SUFFIX")
  case $scheduler in
    maxsp) labels+=(--label "maxsp$SUFFIX=request-level SP") ;;
    laps) labels+=(--label "laps$SUFFIX=LAPS") ;;
    dynamic) labels+=(--label "dynamic$SUFFIX=Dynamic SP") ;;
    *) labels+=(--label "$scheduler$SUFFIX=$scheduler") ;;
  esac
done
# report takes the prefill GPU count as 8 - --decode-gpus.
python benchmark/cdsp/run_experiments.py report --root "$ROOT" --decode-gpus $((8 - GPUS)) \
  --profiles "$PROFILES" --length-bins ${LENGTH_BINS:-1024 4096 8192 16384} \
  --datasets "$DATASET" --models "$MODEL" --loads "${loads[@]}" \
  --schedulers "${schedulers[@]}" "${labels[@]}" \
  --figure "${FIGURE}_$DATASET$SUFFIX"
