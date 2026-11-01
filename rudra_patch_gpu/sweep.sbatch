#!/bin/bash -x
#SBATCH -J v3sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/sweep_%A_%a.out
#SBATCH -e logs/sweep_%A_%a.err
# Do not sbatch this directly; use  bash slurm/submit_rudra.sh [PRESET] [COPIES] [LANES] [WALLTIME].
# One array task = one "lane" = one A100 shared by PROCS worker processes. Lanes split the work by deterministic
# sharding and wait on each other's result files at every phase boundary (no SLURM dependency between lanes), so ALL lanes
# of an array should be able to start together: choose LANES <= the number of GPUs you can get at once.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs
PRESET="${PRESET:-full}"
LANES="${LANES:-${SLURM_ARRAY_TASK_COUNT:-4}}"
LANE="${SLURM_ARRAY_TASK_ID:-0}"
MAX_HOURS="${MAX_HOURS:-23.5}"
PROCS="${PROCS:-4}"                                  # worker processes sharing the 80 GB A100 (GAM trial peaks at ~13-18 GB)
DATASETS="${DATASETS:-svhn cifar10 mnist fmnist}"    # datasets this cluster is responsible for (sources); all four must be on disk
REGS="${REGS:-l1 l2 sam gsam gam}"
WAIT_HOURS="${WAIT_HOURS:-24}"
. "$ROOT/experiments/convnet4_sweep_v3_4datasets/slurm/env.sh" || exit 1
if ! MIN_FREE_GB="${MIN_FREE_GB:-$(( PROCS * 13 ))}" python "$ROOT/experiments/convnet4_sweep_v3_4datasets/slurm/gpu_guard.py"; then
    # The GPU this lane got is not free (someone else / a stale process).  Put THIS array task back in the queue, avoiding the node
    # (keeps the job id, the array slot and the afterany chain, so no duplicate lane can appear).  Up to 8 tries.
    TRY="${SLURM_RESTART_COUNT:-0}"
    BAD="$(hostname -s)"
    if [ "$TRY" -lt 8 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "lane $LANE: GPU not free on $BAD -> requeue #$((TRY+1)) excluding it"
        scontrol requeue "$SLURM_JOB_ID" && scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="${SLURM_EXC_NODES:+$SLURM_EXC_NODES,}$BAD" && exit 0
        echo "requeue/update failed - lane $LANE aborted"; exit 3
    fi
    echo "lane $LANE aborted: GPU not free after $TRY requeues"; exit 3
fi
export PYTHONUNBUFFERED=1 MAX_PROCS_PER_GPU="$PROCS"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-20} / PROCS )); [ "$THREADS" -lt 1 ] && THREADS=1
EXTRA=(); [ -n "${M5_MODE:-}" ] && EXTRA+=(--m5-mode "$M5_MODE")
# work-stealing: workers claim trials one by one, so a lane that fails/starts late/never starts only costs speed (DYNAMIC=0: old fixed split)
[ "${DYNAMIC:-1}" = 1 ] && EXTRA+=(--dynamic --claim-ttl-min "${CLAIM_TTL_MIN:-90}")
echo "lane $LANE/$LANES preset=$PRESET procs=$PROCS threads=$THREADS datasets=[$DATASETS] regs=[$REGS] max-hours=$MAX_HOURS host=$(hostname) $(date)"
nvidia-smi || echo "WARNING: nvidia-smi not found"
cd experiments/convnet4_sweep_v3_4datasets
python run_sweep.py node --lane "$LANE" --lanes "$LANES" --procs "$PROCS" --threads "$THREADS" \
    --max-hours "$MAX_HOURS" --wait-hours "$WAIT_HOURS" --require-gpu --preset "$PRESET" \
    --datasets $DATASETS --regs $REGS --final-seeds 1 2 "${EXTRA[@]}"
rc=$?
echo "Finished $(date) rc=$rc"
echo "NOTE: if the log ends with WALLTIME_EXIT, submit again (or use COPIES>1): finished trials are skipped."
exit $rc
