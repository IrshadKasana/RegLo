#!/bin/bash -x
#SBATCH -J r18lo
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o logs/lo_probe.%j.out
#SBATCH -e logs/lo_probe.%j.err
# Submit from the project root:  sbatch experiments/resnet18_imagenet_sweep/slurm/lo_probe.sbatch
# The real M3 phase (6 meta_lr values) on one GPU, ~8 min each.  Same trial names as the production sweep, so these results are
# REUSED by it (finished trials are skipped).  Purpose: does the proxy-trained learned optimizer train ResNet-18 at all?
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs
. "$ROOT/experiments/resnet18_imagenet_sweep/slurm/env.sh" || exit 1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "host $(hostname) $(date)"
cd experiments/resnet18_imagenet_sweep
for MLR in 0.0001 0.0003 0.001 0.003 0.01 0.03; do
    echo "== M3 meta_lr=$MLR =="
    python run_sweep.py job-lo --preset full --dataset imagenet --reg none --ls 0 --lr 0 --mlr "$MLR" --seed 0
done
echo LO_PROBE_DONE
