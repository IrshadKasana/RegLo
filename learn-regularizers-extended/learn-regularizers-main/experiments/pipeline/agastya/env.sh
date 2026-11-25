# Sourced by every Agastya script in experiments/pipeline/agastya/ (do not run directly).
# Python env: conda env lreg_pipe (python 3.10, torch 2.7.1+cu118, built by build_env.sh). The cu118 wheels bundle
# their CUDA runtime and run on the V100 nodes' driver 455 (CUDA 11.x minor-version compatibility), so no module load.
#   LREG_ENV=/path/to/env   use another env
LREG_ENV="${LREG_ENV:-$HOME/.conda/envs/lreg_pipe}"
if [ ! -x "$LREG_ENV/bin/python" ]; then
    echo "!! $LREG_ENV/bin/python not found: run  bash experiments/pipeline/agastya/build_env.sh  on the login node"
    return 1 2>/dev/null || exit 1
fi
export PATH="$LREG_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$LREG_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1            # ignore ~/.local packages (they belong to other envs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation with several processes on one V100
echo "== env: python=$(command -v python) torch=$(python -c 'import torch;print(torch.__version__)' 2>&1)"
