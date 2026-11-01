#!/bin/bash 
#SBATCH --job-name=rC10convSDMO_CL_AV      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=2      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/opz_convC10_trts_sdmo_cur_learn.out
#SBATCH --error=console/errors/opz_convC10_trts_sdmo_cur_learn.err


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

cd "${SCRIPT_DIR}"
python3 optimizee_train.py  --optimizee_epochs=10000 --n_training=5 --if_avreg --dataset=cifar10 --optimizee_model=conv
