#!/bin/bash -x
#SBATCH -J v3gpucheck
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -o logs/gpucheck.%j.out
#SBATCH -e logs/gpucheck.%j.err
# Submit from the project root:  sbatch experiments/convnet4_sweep_v3_4datasets/slurm/gpu_check.sbatch
# 1) smoke wiring on the GPU (all four datasets are loaded: each is an OOD target of the others)
# 2) one full-width GAM meta-training trial -> seconds + peak GPU memory => how many worker processes fit on the A100-80GB
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs
. "$ROOT/experiments/convnet4_sweep_v3_4datasets/slurm/env.sh" || exit 1
MIN_FREE_GB=60 python "$ROOT/experiments/convnet4_sweep_v3_4datasets/slurm/gpu_guard.py" || exit 3
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; echo "host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
cd experiments/convnet4_sweep_v3_4datasets
echo "== smoke workers (GPU required) =="
python run_sweep.py worker --phase m1 --preset smoke --datasets mnist cifar10 --require-gpu && \
python run_sweep.py job-m2 --preset smoke --dataset fmnist --reg gam --adamw-lr 0.001 --lam 0.01 && \
python run_sweep.py job-lo --preset smoke --dataset mnist --reg gam --ls 0.01 --lr 0.1 --mlr 0.001 --rho 0.04 && echo SMOKE_OK
echo "== full-width timing / memory =="
python slurm/gpucheck.py && echo GPUCHECK_OK
rm -rf results/smoke
