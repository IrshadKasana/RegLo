#!/bin/bash -x
#SBATCH -J c4sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/sweep_%A_%a.out
#SBATCH -e logs/sweep_%A_%a.err
# Do not sbatch this directly; use  bash slurm/submit_rudra.sh [PRESET] [COPIES] [LANES] [WALLTIME].
# One array task = one "lane" = one A100 shared by several worker processes. Lanes split the work by
# deterministic sharding and wait on each other's result files at the two real dependencies
# (M1 -> M2, axes -> local), so no SLURM dependency between lanes is needed.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs
PRESET="${PRESET:-full}"
LANES="${LANES:-${SLURM_ARRAY_TASK_COUNT:-4}}"
LANE="${SLURM_ARRAY_TASK_ID:-0}"
MAX_HOURS="${MAX_HOURS:-23.5}"
PROCS="${PROCS:-5}"                   # capped automatically by GPU memory (5 on an 80 GB A100)
MODULE_ENV="${MODULE_ENV:-mldl/Pytorch}"
module purge
module load "$MODULE_ENV"
if [ -n "${CONDA_ENV:-}" ]; then      # optional own env: MODULE_ENV=mldl/Miniconda CONDA_ENV=myenv bash slurm/submit_rudra.sh
    eval "$(conda shell.bash hook)" && conda activate "$CONDA_ENV"
fi
export PYTHONUNBUFFERED=1
THREADS=$(( ${SLURM_CPUS_PER_TASK:-20} / PROCS )); [ "$THREADS" -lt 1 ] && THREADS=1
echo "lane $LANE/$LANES preset=$PRESET procs=$PROCS threads=$THREADS max-hours=$MAX_HOURS host=$(hostname) $(date)"
nvidia-smi || echo "WARNING: nvidia-smi not found"
cd experiments/convnet4_lambda_sweep_svhn_cifar10
python run_sweep.py node --lane "$LANE" --lanes "$LANES" --procs "$PROCS" --threads "$THREADS" \
    --max-hours "$MAX_HOURS" --wait-hours 12 --require-gpu --preset "$PRESET" \
    --datasets svhn cifar10 --regs l1 l2 sam gsam gam --meta-seeds 0 --radius 1
rc=$?
echo "Finished $(date) rc=$rc"
echo "NOTE: if the log ends with WALLTIME_EXIT, submit again (or use COPIES>1): finished cells are skipped."
exit $rc
