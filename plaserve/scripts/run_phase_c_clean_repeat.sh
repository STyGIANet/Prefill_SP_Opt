#!/usr/bin/env bash
# Usage: bash scripts/run_phase_c_clean_repeat.sh CAMPAIGN_DIR [REPEAT...]
# Retry single repeats (default 5 6 7) until one runs with no other user on GPUs 2-5.
C=$(realpath "${1:?usage: run_phase_c_clean_repeat.sh CAMPAIGN_DIR [REPEAT...]}"); shift
[ $# -eq 0 ] && set -- 5 6 7
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate plaserve
export HF_HOME=$HOME/.cache/huggingface HF_HUB_CACHE=$HOME/.cache/huggingface/hub HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets HF_HUB_OFFLINE=1
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
for n in "$@"; do
  free=0
  while [ $free -lt 3 ]; do
    apps=$(nvidia-smi -i 2,3,4,5 --query-compute-apps=pid --format=csv,noheader)
    if [ -z "$apps" ]; then free=$((free+1)); else free=0; fi
    sleep 20
  done
  echo "$(date +%T) starting repeat $n"
  python -u $ROOT/scripts/run_phase_c_one_repeat.py $C $n > $C/repeat-$n-session.log 2>&1
  tail -1 $C/repeat-$n-session.log
  if grep -q '"validation": "passed", "results": 12, "foreign_users": 0, "unknown_owners": 0, "errors": 0' $C/repeat-$n-session.log; then
    echo "CLEAN repeat $n"; exit 0
  fi
done
echo "No clean repeat after 3 attempts"; exit 1
