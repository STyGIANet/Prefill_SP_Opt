#!/usr/bin/env bash
# Compare tetris, dynamic at an earlier commit, the current dynamic, and LAPS on
# the same prepared workloads, then print the tables and the slowdown figure.
# Runs that already have results are kept; any other user's GPU job stops it.
#   LOADS    offered input tokens/s (default "4096 8192 16384")
#   DATASET  default wildchat; MODEL default qwen3_8b
#   BEFORE   git ref of the earlier dynamic scheduler (default HEAD)
#   SUFFIX   appended to every output name (default _cmp)
# The earlier code is checked out as a git worktree at ~/Prefill_SP_Opt_before.
# LAPS runs prefill-only replicas on all eight GPUs, so it has no TBT.
set -euo pipefail
cd ~/Prefill_SP_Opt && source .venv/bin/activate

LOADS=${LOADS:-4096 8192 16384}
DATASET=${DATASET:-wildchat}
MODEL=${MODEL:-qwen3_8b}
BEFORE=${BEFORE:-HEAD}
SUFFIX=${SUFFIX:-_cmp}
ROOT=python/sglang/srt/dynamic_sp/work/dataset_sweep_pd_tp4
WORKTREE=$HOME/Prefill_SP_Opt_before

if [ -d "$WORKTREE" ]; then
  git -C "$WORKTREE" checkout --quiet --detach "$BEFORE"
else
  git worktree add --detach "$WORKTREE" "$BEFORE"
fi
echo "[compare] dynamic_before$SUFFIX runs $(git -C "$WORKTREE" log -1 --format='%h %s')"

for load in $LOADS; do
  dir=$ROOT/$DATASET/$MODEL/l${load}_s0
  for spec in "tetris:tetris" "dynamic_before:dynamic --code $WORKTREE/python" \
              "dynamic:dynamic" "laps:laps"; do
    name=${spec%%:*}$SUFFIX
    read -r -a options <<< "${spec#*:}"
    if [ -f "$dir/$name/results.json" ]; then
      echo "[compare] skip $name l$load: results exist"
      continue
    fi
    python python/sglang/srt/dynamic_sp/run_dynamic_sp.py --dataset "$DATASET" --model "$MODEL" \
      --load "$load" --name "$name" --scheduler "${options[@]}"
  done
done

read -r -a loads <<< "$LOADS"
python benchmark/cdsp/run_experiments.py report --datasets "$DATASET" --models "$MODEL" \
  --loads "${loads[@]}" \
  --schedulers "tetris$SUFFIX" "dynamic_before$SUFFIX" "dynamic$SUFFIX" "laps$SUFFIX" \
  --label "tetris$SUFFIX=tetris" \
  --label "dynamic_before$SUFFIX=Dynamic SP (before)" \
  --label "dynamic$SUFFIX=Dynamic SP (current)" \
  --label "laps$SUFFIX=LAPS (prefill only)" \
  --figure "python/sglang/srt/dynamic_sp/work/figures/ttft_slowdown$SUFFIX"
