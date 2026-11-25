#!/bin/bash
# Login-node setup check (light):   bash experiments/pipeline/agastya/setup_agastya.sh
# Links the datasets that already exist on Agastya (from ~/learn-regg_v5_consolidated_1/datasets) into ./datasets,
# then checks the python env, every dataset file the three cases need, and lists the run grid of mlp_mnist.
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../../.." && pwd)"; cd "$ROOT" || exit 1
SRC="${SRC_DATASETS:-$HOME/learn-regg_v5_consolidated_1/datasets}"
mkdir -p datasets/SVHN
for d in MNIST FashionMNIST cifar-10-batches-py; do [ -e "datasets/$d" ] || ln -s "$SRC/$d" "datasets/$d"; done
for f in train_32x32.mat test_32x32.mat; do [ -e "datasets/SVHN/$f" ] || ln -s "$SRC/$f" "datasets/SVHN/$f"; done
. "$HERE/env.sh" || exit 1
ok=1
python - <<'PY' || ok=0
import importlib, sys
for m in ("torch", "torchvision", "numpy", "scipy", "yaml", "matplotlib"):
    mod = importlib.import_module(m); print(f"OK   {m} {getattr(mod, '__version__', '')}")
import torch
from torch.utils.flop_counter import FlopCounterMode  # noqa: F401
from torch.func import functional_call  # noqa: F401
print("torch CUDA build:", torch.version.cuda, "archs:", torch.cuda.get_arch_list())
assert "sm_70" in torch.cuda.get_arch_list(), "this torch build has no V100 (sm_70) kernels"
PY
for f in MNIST/raw/train-images-idx3-ubyte MNIST/raw/t10k-images-idx3-ubyte \
         FashionMNIST/raw/train-images-idx3-ubyte FashionMNIST/raw/t10k-images-idx3-ubyte \
         cifar-10-batches-py/data_batch_1 cifar-10-batches-py/test_batch cifar-100-python/train cifar-100-python/test \
         SVHN/train_32x32.mat SVHN/test_32x32.mat imagenet32/x.npy imagenet32/y.npy; do
    if [ -e "datasets/$f" ]; then echo "OK   $f"; else echo "MISSING $f"; ok=0; fi
done
python -c "
import numpy as np; x = np.load('datasets/imagenet32/x.npy', mmap_mode='r'); y = np.load('datasets/imagenet32/y.npy', mmap_mode='r')
print(f'     imagenet32: x {x.shape} {x.dtype}, y {y.shape}')
assert x.shape == (1331167, 3, 32, 32) and len(y) == len(x), 'unexpected imagenet32 layout'" || ok=0
echo "== run grid (mlp_mnist)"; EXP_PREFIX=agastya python experiments/pipeline/rudra/rudra_jobs.py list mlp_mnist | paste - - - | sed 's/^/   /' || ok=0
[ "$ok" = 1 ] && echo "== SETUP_OK. Next: bash experiments/pipeline/agastya/submit_agastya.sh" || echo "== fix the problems above first"
