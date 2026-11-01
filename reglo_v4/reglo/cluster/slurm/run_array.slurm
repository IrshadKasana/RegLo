#!/bin/bash
#SBATCH --job-name=reglo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/%x-%A_%a.out
# ════════════════════════════════════════════════════════════════════════════
#  One manifest line per array task: task i runs line i.
#  Submit through cluster/slurm/submit.sh, which builds the manifest and exports
#
#      MANIFEST          artifacts/<profile>/manifests/submitted/<experiment>[…].txt
#      REGLO_ROOT        the repository
#      REGLO_MAX_HOURS   stop cleanly before the time limit (exit 75, resumable)
#      REGLO_REQUEUE     1: a run stopped at max_hours requeues its own task
#
#  The #SBATCH lines above are only defaults: submit.sh sets partition, resources and
#  time on the sbatch command line (which wins).
# ════════════════════════════════════════════════════════════════════════════

REGLO_ROOT="${REGLO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REGLO_ROOT" || { echo "cannot cd to $REGLO_ROOT"; exit 2; }
source cluster/env.sh

INDEX="${SLURM_ARRAY_TASK_ID:-${INDEX:-0}}"
if [ -z "${MANIFEST:-}" ] || [ ! -f "$MANIFEST" ]; then
  echo "MANIFEST is not set or missing: '${MANIFEST:-}' (submit with cluster/slurm/submit.sh)"
  exit 2
fi

echo "== $(date '+%F %T') | job ${SLURM_JOB_ID:-interactive} (array ${SLURM_ARRAY_JOB_ID:--}, task $INDEX," \
     "restart ${SLURM_RESTART_COUNT:-0}) | $(hostname) | line $INDEX of $MANIFEST"
echo "== python $REGLO_PYTHON | profile $REGLO_PROFILE | artifacts $REGLO_ARTIFACTS | max_hours ${REGLO_MAX_HOURS:-0}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

status=0
"$REGLO_PYTHON" scripts/run_manifest.py "$MANIFEST" "$INDEX" || status=$?
echo "== $(date '+%F %T') | exit $status"

if [ "$status" -eq 75 ]; then  # stopped at max_hours with a checkpoint: continue in a fresh allocation
  TASK="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
  TASK="${TASK:-$SLURM_JOB_ID}"
  if [ "${REGLO_REQUEUE:-1}" = "1" ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 50 ] && scontrol requeue "$TASK"; then
    echo "== requeued $TASK: it continues from its checkpoint"
    exit 0
  fi
  echo "== stopped at max_hours: submit the same command again to continue (finished runs are skipped)"
fi
exit "$status"
