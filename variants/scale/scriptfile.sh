#!/bin/bash 
#SBATCH --job-name=SDMO_CL_AV      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=2      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/sdmo_cur_learn_AV.out
#SBATCH --error=console/errors/sdmo_cur_learn_AV.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

cd "${SCRIPT_DIR}"
python3 train.py --if_cl --if_avreg
