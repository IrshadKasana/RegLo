#!/bin/bash
# One-time environment + data check for PARAM Rudra.  Run on the LOGIN node (nothing heavy):  bash slurm/setup_env.sh
# Needs: torch (CUDA build), torchvision (MNIST / FashionMNIST / CIFAR-10 / SVHN readers), numpy, scipy (SVHN .mat files).
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
. "$HERE/env.sh" || exit 1       # finds a python with torch (see env.sh for MODULE_ENV / CONDA_ENV / VENV)
echo "== python: $(which python)  ($(python --version 2>&1))"
python - <<'PY'
import importlib
for m in ("torch", "torchvision", "numpy", "scipy"):
    try:
        mod = importlib.import_module(m); print(f"OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"MISS {m}: {e}")
import torch
print("torch CUDA build:", torch.version.cuda, "| cuda visible on the login node (False is normal):", torch.cuda.is_available())
PY
python -c "import scipy" 2>/dev/null || { echo "== scipy missing -> pip install --user scipy"; pip install --user scipy; }
python -c "import torchvision" 2>/dev/null || echo "!! torchvision missing: pip install --user torchvision (match the torch version) or use another module"
echo "== datasets under $ROOT/datasets"
ok=1
for f in imagenet32/x.npy imagenet32/y.npy cifar-10-batches-py/data_batch_1 train_32x32.mat test_32x32.mat MNIST/raw/train-images-idx3-ubyte MNIST/raw/t10k-images-idx3-ubyte \
         FashionMNIST/raw/train-images-idx3-ubyte FashionMNIST/raw/t10k-images-idx3-ubyte; do
    if [ -f "$ROOT/datasets/$f" ]; then echo "OK   $f"; else echo "MISSING $f"; ok=0; fi
done
[ -f "$ROOT/MD5SUMS_datasets.txt" ] && (cd "$ROOT" && md5sum -c --quiet MD5SUMS_datasets.txt && echo "OK   all dataset checksums match") || echo "(no checksum file or a checksum FAILED - see above)"
[ "$ok" = 1 ] && echo "== done. Next:  mkdir -p logs && sbatch experiments/resnet18_imagenet_sweep/slurm/gpu_check.sbatch" || echo "== fix the missing files first"
