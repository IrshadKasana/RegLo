#!/bin/bash
# Submit the full pipeline of ONE case on Agastya (login node; run from anywhere):
#
#   bash experiments/pipeline/agastya/submit_agastya.sh                 # mlp_mnist, 1 lane, 3 processes, 190 h
#   bash experiments/pipeline/agastya/submit_agastya.sh convnet4_cifar10
#   PREFLIGHT=0 bash experiments/pipeline/agastya/submit_agastya.sh     # restart / continue without the preflight
#
# gpu-ext allows 2 running jobs per user (max_run), walltime up to 192 h, one V100-32GB per node. One lane job
# per case. Re-running is safe: finished units and phases are skipped, meta-training resumes from its resume file.
# Knobs: PROCS=3 GPU_BUDGET_GB=31 WALL_H=190 MAX_GEN=3 EXP_PREFIX=agastya NUM_WORKERS=3 PREFLIGHT=1
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT" && mkdir -p logs/agastya
CASE="${1:-mlp_mnist}"
case "$CASE" in mlp_mnist) N=lrMLP;; convnet4_cifar10) N=lrCNN;; resnet18_imagenet) N=lrR18;;
    *) echo "!! unknown case '$CASE' (mlp_mnist | convnet4_cifar10 | resnet18_imagenet)"; exit 2;; esac
PROCS="${PROCS:-3}"; GPU_BUDGET_GB="${GPU_BUDGET_GB:-31}"; WALL_H="${WALL_H:-190}"; MAX_GEN="${MAX_GEN:-3}"
EXP_PREFIX="${EXP_PREFIX:-agastya}"; PREFLIGHT="${PREFLIGHT:-1}"
[ "$WALL_H" -gt 192 ] && { echo "!! WALL_H=$WALL_H > 192 (gpu-ext limit)"; exit 2; }
if qstat -u "$USER" 2>/dev/null | grep -q " $N "; then
    echo "!! a $N job is already queued/running:"; qstat -u "$USER"; echo "   (FORCE=1 to submit another lane anyway)"
    [ "${FORCE:-0}" = 1 ] || exit 2
fi
stamp=$(date +%Y%m%d_%H%M)
id=$(qsub -N "$N" -l walltime="${WALL_H}:00:00" -o "$ROOT/logs/agastya/pbs_${CASE}_${stamp}.out" \
     -v CASE="$CASE",PROCS="$PROCS",WALL_H="$WALL_H",GPU_BUDGET_GB="$GPU_BUDGET_GB",EXP_PREFIX="$EXP_PREFIX",GEN=1,MAX_GEN="$MAX_GEN",PREFLIGHT="$PREFLIGHT",NUM_WORKERS="${NUM_WORKERS:-3}" \
     "$HERE/lane.pbs") || { echo "!! qsub failed"; exit 1; }
echo "$(date) $CASE gen 1 $id (submit_agastya.sh)" >> logs/agastya/submits.txt
echo "$CASE -> job $id  (procs $PROCS, GPU budget ${GPU_BUDGET_GB} GB, wall ${WALL_H} h, preflight $PREFLIGHT)"
echo "lane log:  logs/agastya/${CASE}_gen1_${id%%.*}.log"
echo "progress:  bash experiments/pipeline/agastya/status.sh $CASE"
echo "unit logs: artifacts/pipeline/${EXP_PREFIX}_$CASE/rudra_queue/logs/"
