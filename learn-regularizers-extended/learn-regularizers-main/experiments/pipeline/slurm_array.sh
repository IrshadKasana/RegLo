#!/bin/bash
# SLURM template: one array task per trained entity (24 with the default grid).
#
#   N=$(python3 experiments/pipeline/run_pipeline.py --list | wc -l)
#   jid=$(sbatch --parsable --array=0-$((N-1)) experiments/pipeline/slurm_array.sh 1)   # phase 1
#   jid=$(sbatch --parsable --dependency=afterok:$jid --array=0-$((N-1)) experiments/pipeline/slurm_array.sh 2)
#   jid=$(sbatch --parsable --dependency=afterok:$jid --array=0-$((N-1)) experiments/pipeline/slurm_array.sh 3)
#   sbatch --dependency=afterok:$jid experiments/pipeline/slurm_array.sh report
#
# Phase-2/3 tasks reuse cached optimizee runs, so parallel tasks never repeat finished work.
# Adapt partition / gres / time to your cluster.
#SBATCH --job-name=l2o-pipeline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=artifacts/pipeline/console/%x_%A_%a.out
#SBATCH --error=artifacts/pipeline/console/%x_%A_%a.err
set -euo pipefail

PHASE="${1:?usage: slurm_array.sh <1|2|3|report> [extra run_pipeline args]}"
shift || true
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"
mkdir -p artifacts/pipeline/console
if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

if [ "${PHASE}" = "report" ]; then
  python3 experiments/pipeline/run_pipeline.py --phases report "$@"
  exit 0
fi

mapfile -t RUN_IDS < <(python3 experiments/pipeline/run_pipeline.py --list "$@")
RUN_ID="${RUN_IDS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "phase ${PHASE} | task ${SLURM_ARRAY_TASK_ID:-0} | ${RUN_ID}"
python3 experiments/pipeline/run_pipeline.py --phases "${PHASE}" --run-ids "${RUN_ID}" "$@"
