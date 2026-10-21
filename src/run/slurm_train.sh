#!/bin/bash 
#SBATCH --job-name=T_gamDMO      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=2-23:59:59          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=artifacts/dmo/console/outs/testp_DMO_gam_adam_290424.out
#SBATCH --error=artifacts/dmo/console/errors/testp_DMO_gam_adam_290424.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

cd "${ROOT_DIR}"
python3 experiments/dmo/train.py --regularizer=gam
