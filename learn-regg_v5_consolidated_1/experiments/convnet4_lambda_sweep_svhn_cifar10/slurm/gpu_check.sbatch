#!/bin/bash -x
#SBATCH -J c4gpucheck
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o logs/gpucheck.%j.out
#SBATCH -e logs/gpucheck.%j.err
# Submit from the project root:  sbatch slurm/gpu_check.sbatch     (use  -p gpu  ... see README_RUDRA.md)
# Runs: smoke versions of M1 / M3 / M4 on the GPU (--require-gpu), then full-width timings + peak memory.
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs
MODULE_ENV="${MODULE_ENV:-mldl/Pytorch}"
module purge; module load "$MODULE_ENV"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
cd experiments/convnet4_lambda_sweep_svhn_cifar10
echo "== smoke workers (GPU required) =="
python run_sweep.py worker --phase m1 --preset smoke --datasets cifar10 svhn --require-gpu && \
python run_sweep.py job-lo --preset smoke --dataset svhn --reg gam --ls 0.01 --lr 0.1 && \
python run_sweep.py job-lo --preset smoke --dataset cifar10 --reg gsam --ls 0 --lr 0.1 && echo SMOKE_OK
echo "== full-width timing =="
python slurm/gpucheck.py
rm -rf results/smoke /tmp/gpucheck_lo.pt
