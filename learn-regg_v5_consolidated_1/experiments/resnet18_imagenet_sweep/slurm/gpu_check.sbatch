#!/bin/bash -x
#SBATCH -J r18check
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00
#SBATCH -o logs/gpucheck.%j.out
#SBATCH -e logs/gpucheck.%j.err
# Submit from the project root:  sbatch experiments/resnet18_imagenet_sweep/slurm/gpu_check.sbatch
# 1) smoke wiring on the GPU (ImageNet-32 subset + the four OOD target datasets)
# 2) full-width ResNet-18 trials (M1, M2-GAM, M5-GAM) -> seconds + peak GPU memory => how many worker processes fit on the A100-80GB
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs
. "$ROOT/experiments/resnet18_imagenet_sweep/slurm/env.sh" || exit 1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; echo "host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
cd experiments/resnet18_imagenet_sweep
echo "== smoke workers (GPU required) =="
SWEEP_IMAGENET_LIMIT=20000 python run_sweep.py worker --phase m1 --preset smoke --require-gpu && \
SWEEP_IMAGENET_LIMIT=20000 python run_sweep.py job-m2 --preset smoke --dataset imagenet --reg gam --adamw-lr 0.001 --lam 0.01 && \
SWEEP_IMAGENET_LIMIT=20000 python run_sweep.py job-lo --preset smoke --dataset imagenet --reg gam --ls 0.01 --lr 0.1 --mlr 0.001 --rho 0.04 && echo SMOKE_OK
echo "== full-width timing / memory =="
python slurm/gpucheck.py && echo GPUCHECK_OK
rm -rf results/smoke
