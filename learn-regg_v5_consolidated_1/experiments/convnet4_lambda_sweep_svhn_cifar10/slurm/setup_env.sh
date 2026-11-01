#!/bin/bash
# One-time environment check for PARAM Rudra.  Run on the LOGIN node:   bash slurm/setup_env.sh
# (nothing heavy runs here: only module loads, `python -c import`, and an optional `pip install --user`).
#
# The sweep needs: torch (CUDA build), numpy, scipy (reads the SVHN .mat files).
# Rudra ships ready-made environments (manual §"Loading the Conda Base Module"): `module load mldl/Pytorch`
# = PyTorch 2.2.1, py3.10, CUDA 11.8 (GPU).  We use that and only add scipy if it is missing.
MODULE_ENV="${MODULE_ENV:-mldl/Pytorch}"
module purge 2>/dev/null
module load "$MODULE_ENV" || { echo "module '$MODULE_ENV' not found. Try: module avail 2>&1 | grep -i -E 'torch|conda'"; exit 1; }
echo "== python: $(which python)  ($(python --version 2>&1))"
python - <<'PY'
import importlib
for m in ("torch", "numpy", "scipy"):
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"MISS {m}: {e}")
import torch
print("torch CUDA build:", torch.version.cuda, "| cuda visible here (login node has no GPU, False is normal):", torch.cuda.is_available())
PY
if ! python -c "import scipy" 2>/dev/null; then
    echo "== scipy missing -> installing to ~/.local (pip --user)"
    pip install --user scipy
fi
echo "== done. Next:  sbatch slurm/gpu_check.sbatch"
