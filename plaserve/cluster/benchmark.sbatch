#!/usr/bin/env bash
#SBATCH --job-name=PLAserve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.log

set -euo pipefail
# Submit from the PLAserve root. Override account/partition/GPU resource syntax
# using your site's sbatch options; module setup belongs in your submission shell.
project_root="${PLASERVE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$project_root"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${PLASERVE_ENV:-plaserve}"
fi
read -r -a selected_models <<< "${MODELS:-32b}"
if [[ "${PROFILE:-h200}" == a100-2 ]]; then
  python -m plaserve cluster --profile a100-2 --output "results/${SLURM_JOB_ID:-manual}"
else
  python -m plaserve cluster --profile "${PROFILE:-h200}" \
    --models "${selected_models[@]}" --output "results/${SLURM_JOB_ID:-manual}"
fi
