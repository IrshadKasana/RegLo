#!/bin/bash
# Tiny end-to-end run of all 6 methods x 5 regularizers x both B on the REAL datasets
# (SVHN, MNIST, FMNIST, CIFAR-10, CIFAR-100, ImageNet-32). Uses the --quick preset, so it checks
# wiring and data paths, not performance. Results: artifacts/pipeline/smoke_real/tables/summary.md
# Extra arguments are forwarded, e.g. --device cuda:0
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PY="${PYTHON:-python3}"
"${PY}" experiments/pipeline/run_pipeline.py --quick \
  --set experiment.name=smoke_real --set experiment.download=false \
  --set meta_training.smooth_every=5 "$@"
echo "tables: artifacts/pipeline/smoke_real/tables/summary.md"
