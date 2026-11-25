#!/bin/bash
#SBATCH -J rgCheck
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o logs/rudra/gpucheck_%j.out
#SBATCH -e logs/rudra/gpucheck_%j.err
# Step 2 of README_RUDRA.md (before the full submission). From the repository root:
#     mkdir -p logs/rudra && sbatch experiments/pipeline/rudra/gpu_check.sbatch
# 1) end-to-end smoke of all 24 units of each case on the REAL datasets with the tiny --quick preset
#    (checks data paths, every method x regularizer x B, phases 1-3 and the report)       -> SMOKE_OK_<case>
# 2) speed + peak GPU memory of the heaviest units with the production settings          -> GPUCHECK_OK
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs/rudra
. "$ROOT/experiments/pipeline/rudra/env.sh" || exit 1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "== unit tests"; python tests/test_pipeline.py 2>&1 | tail -10
for c in mlp_mnist convnet4_cifar10 resnet18_imagenet; do
    echo "== smoke $c"
    args=$(python experiments/pipeline/rudra/rudra_jobs.py args "$c")
    eval "python experiments/pipeline/run_pipeline.py $args --quick --set experiment.name=rudra_smoke_$c" \
        > "logs/rudra/smoke_${c}_${SLURM_JOB_ID:-x}.log" 2>&1 \
        && echo "SMOKE_OK_$c" || { echo "!! SMOKE FAILED $c: tail of logs/rudra/smoke_${c}_${SLURM_JOB_ID:-x}.log"; tail -30 "logs/rudra/smoke_${c}_${SLURM_JOB_ID:-x}.log"; }
done
echo "== timing / memory (production settings)"
python experiments/pipeline/rudra/gpu_check.py
rm -rf artifacts/pipeline/gpucheck_*
echo "finished $(date)"
