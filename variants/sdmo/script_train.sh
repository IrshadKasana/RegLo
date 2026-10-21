#!/bin/bash 
#SBATCH --job-name=SDMO      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/2_train_sdmo_no_CL.out
#SBATCH --error=console/errors/2_train_sdmo_no_CL.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

cd "${SCRIPT_DIR}"
python3 train.py
