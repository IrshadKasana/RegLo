#!/bin/bash
# Full hierarchical pipeline on one machine:
#   phase 1 (meta-training) -> phase 2 (optimizer evaluation) -> phase 3 (optimizee evaluation) -> 3D tables
# Extra arguments are forwarded, e.g.:
#   bash experiments/pipeline/run_full.sh --set experiment.name=run_a --device cuda:0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

python3 experiments/pipeline/run_pipeline.py --phases 1 "$@"
python3 experiments/pipeline/run_pipeline.py --phases 2 "$@"
python3 experiments/pipeline/run_pipeline.py --phases 3 report "$@"
