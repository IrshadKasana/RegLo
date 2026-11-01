#!/bin/bash -x
#SBATCH -J c4report
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p cpu
#SBATCH -t 00:30:00
#SBATCH -o logs/report_%j.out
#SBATCH -e logs/report_%j.err
# Aggregates results/<preset>/*.json -> summary.md and best_lambdas.json. Cheap; CPU partition is fine.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
PRESET="${PRESET:-full}"
module purge; module load "${MODULE_ENV:-mldl/Pytorch}"
cd experiments/convnet4_lambda_sweep_svhn_cifar10
python run_sweep.py report --preset "$PRESET" --datasets svhn cifar10 --regs l1 l2 sam gsam gam --meta-seeds 0 --radius 1
