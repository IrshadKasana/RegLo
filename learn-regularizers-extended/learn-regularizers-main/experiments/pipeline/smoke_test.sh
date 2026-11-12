#!/bin/bash
# Tiny end-to-end run of all 6 methods x 5 regularizers x both B on learnable synthetic data
# (no downloads, CPU-friendly). Checks wiring, not performance.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
python3 experiments/pipeline/run_pipeline.py --quick \
  --set experiment.synthetic_data=true --set experiment.name=smoke \
  --set meta_training.smooth_every=5 "$@"
echo "tables: artifacts/pipeline/smoke/tables/summary.md"
