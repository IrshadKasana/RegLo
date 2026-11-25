#!/bin/bash
# Step 1 of README_RUDRA.md. Run on the Rudra LOGIN node (light: imports, file checks, run-grid listing):
#     bash experiments/pipeline/rudra/check_setup.sh
# Uses the same python as the jobs (env.sh: ~/rudra_env.sh, MLDL/Pytorch-gpu, or MODULE_ENV / CONDA_ENV / VENV / KEEP_ENV=1).
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT" || exit 1
. "$HERE/env.sh" || exit 1
ok=1
python - <<'PY' || ok=0
import importlib, sys
bad = []
for m in ("torch", "torchvision", "numpy", "scipy", "yaml", "matplotlib"):
    try:
        mod = importlib.import_module(m); print(f"OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"MISS {m}: {e}"); bad.append(m)
import torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if v < (2, 1):
    print(f"MISS torch>=2.1 needed (torch.func, FlopCounterMode); found {torch.__version__}"); bad.append("torch")
from torch.utils.flop_counter import FlopCounterMode  # noqa: F401
from torch.func import functional_call  # noqa: F401
print("torch CUDA build:", torch.version.cuda)
if bad:
    print("-> pip install --user " + " ".join("PyYAML" if b == "yaml" else b for b in bad if b != "torch")); sys.exit(1)
PY
DR="${DATA_ROOT:-$ROOT/datasets}"
echo "== datasets under $DR  (set DATA_ROOT=... to use another folder; the jobs get it too)"
for f in MNIST/raw/train-images-idx3-ubyte MNIST/raw/t10k-images-idx3-ubyte \
         FashionMNIST/raw/train-images-idx3-ubyte FashionMNIST/raw/t10k-images-idx3-ubyte \
         cifar-10-batches-py/data_batch_1 cifar-10-batches-py/test_batch cifar-100-python/train cifar-100-python/test \
         SVHN/train_32x32.mat SVHN/test_32x32.mat imagenet32/x.npy imagenet32/y.npy; do
    if [ -e "$DR/$f" ]; then echo "OK   $f"; else echo "MISSING $f"; ok=0; fi
done
if [ -e "$DR/imagenet32/y.npy" ]; then
    python -c "
import numpy as np; y = np.load('$DR/imagenet32/y.npy', mmap_mode='r'); x = np.load('$DR/imagenet32/x.npy', mmap_mode='r')
print(f'     imagenet32: x {x.shape} {x.dtype}, y {y.shape} (expected (1331167, 3, 32, 32) uint8 = 1,281,167 train + 50,000 val)')
assert x.shape == (1331167, 3, 32, 32) and len(y) == len(x), 'unexpected imagenet32 layout'" || ok=0
fi
echo "== run grid per case (scheduling order, GPU-GB estimate used to pack processes on one A100)"
for c in mlp_mnist convnet4_cifar10 resnet18_imagenet; do
    echo "-- $c"; python experiments/pipeline/rudra/rudra_jobs.py list "$c" | paste - - - | sed 's/^/   /' || ok=0
done
[ "$ok" = 1 ] && echo "== setup OK. Next: mkdir -p logs/rudra && sbatch experiments/pipeline/rudra/gpu_check.sbatch" \
              || echo "== fix the problems above first"
