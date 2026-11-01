#!/bin/bash 
#SBATCH --job-name=adamgsam_ts      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=2-23:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=artifacts/dmo/console/outs/0105_analysis_adam_gsam_test_all.out
#SBATCH --error=artifacts/dmo/console/errors/0105_analysis_adam_gsam_test_all.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

cd "${ROOT_DIR}"
python3 experiments/dmo/evaluate_optimizee.py --dataset=mnist --optimizee_model=mlp --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=mnist --optimizee_model=mlp_Relu --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=mnist --optimizee_model=conv --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=fmnist --optimizee_model=mlp --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=fmnist --optimizee_model=mlp_Relu --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=fmnist --optimizee_model=conv --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=cifar10 --optimizee_model=mlp --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=cifar10 --optimizee_model=mlp_Relu --base_optimizer=adam --if_normal_optim --regularizer=gsam
python3 experiments/dmo/evaluate_optimizee.py --dataset=cifar10 --optimizee_model=conv --base_optimizer=adam --if_normal_optim --regularizer=gsam
