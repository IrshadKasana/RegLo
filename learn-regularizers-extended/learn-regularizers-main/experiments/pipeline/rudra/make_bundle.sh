#!/bin/bash
# Build the two upload archives for Rudra (run on your machine, from anywhere):
#   <out>/learn-regularizers-rudra_code.tgz            code + configs (no datasets / artifacts / .git), ~3 MB
#   <out>/learn-regularizers-rudra_small_datasets.tgz  MNIST, FashionMNIST, CIFAR-10, CIFAR-100, SVHN (~0.7 GB)
# ImageNet-32 (3.9 GB) is NOT packed: link the copy already converted on Rudra (README_RUDRA.md step 0),
# or pack it too with WITH_IMAGENET=1.
#   bash experiments/pipeline/rudra/make_bundle.sh [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="${1:-$(dirname "$ROOT")}"
cd "$ROOT"
tar --exclude=./datasets --exclude=./artifacts --exclude=./.git --exclude=./.venv --exclude='__pycache__' \
    --exclude='*.pyc' --exclude=./logs --exclude=./archive -czf "$OUT/learn-regularizers-rudra_code.tgz" .
D=(datasets/MNIST/raw datasets/FashionMNIST/raw datasets/cifar-10-batches-py datasets/cifar-100-python datasets/SVHN)
[ "${WITH_IMAGENET:-0}" = 1 ] && D+=(datasets/imagenet32)
tar -czf "$OUT/learn-regularizers-rudra_small_datasets.tgz" "${D[@]}"
ls -lh "$OUT"/learn-regularizers-rudra_*.tgz
