# Sourced by every Rudra script:  . "$(dirname "$0")/env.sh"   (do not run directly)
# Finds a python that can import torch.  Controls (environment variables, all optional):
#   MODULE_ENV="m1 m2 ..."   space-separated modules to load, in order            e.g. MODULE_ENV="anaconda3 cuda/11.8"
#   CONDA_ENV=name|/path     conda env to activate after the modules              (needs conda on PATH or in a module)
#   VENV=/path/to/venv       python venv to activate instead
#   KEEP_ENV=1               load nothing (use the python already on PATH)
# With nothing set it tries the usual Rudra setup (~/rudra_env.sh, MLDL/Pytorch-gpu, MLDL/Pytorch), then falls back to whatever python is on PATH.
_have_torch() { python -c "import torch" >/dev/null 2>&1; }
if [ "${KEEP_ENV:-0}" != 1 ]; then
    if [ -n "${MODULE_ENV:-}" ]; then
        for _m in $MODULE_ENV; do module load "$_m" || echo "!! module '$_m' failed to load"; done
    elif [ -z "${CONDA_ENV:-}${VENV:-}" ]; then
        # 1) the file the old Rudra project used (it loads MLDL/Pytorch-gpu = conda env Pytorch-gpu, py3.10)
        if [ -f "$HOME/rudra_env.sh" ]; then . "$HOME/rudra_env.sh" >/dev/null 2>&1; MODULE_ENV="~/rudra_env.sh"; fi
        # 2) module names on Rudra (case-sensitive: MLDL/..., not mldl/...)
        if ! _have_torch; then
            for _m in MLDL/Pytorch-gpu MLDL/Pytorch; do
                if module load "$_m" 2>/dev/null && _have_torch; then MODULE_ENV="$_m"; break; fi
            done
        fi
    fi
fi
if [ -n "${CONDA_ENV:-}" ]; then
    command -v conda >/dev/null 2>&1 || { [ -n "${CONDA_SH:-}" ] && . "$CONDA_SH"; }
    eval "$(conda shell.bash hook)" 2>/dev/null; conda activate "$CONDA_ENV" || echo "!! conda activate $CONDA_ENV failed"
fi
[ -n "${VENV:-}" ] && . "$VENV/bin/activate"
if ! _have_torch; then
    echo "!! no python with torch found.  python=$(command -v python)"
    echo "   Find one:  module avail 2>&1 | grep -i -E 'torch|conda|python'   or   conda env list"
    echo "   then:      MODULE_ENV=\"<module(s)>\" [CONDA_ENV=<env>] sbatch/bash ...   (or VENV=/path, or KEEP_ENV=1)"
    return 1 2>/dev/null || exit 1
fi
echo "== env: python=$(command -v python) torch=$(python -c 'import torch;print(torch.__version__)') modules=[${MODULE_ENV:-none}] conda=[${CONDA_ENV:-none}]"
export MODULE_ENV CONDA_ENV VENV

# --- GPU fencing -----------------------------------------------------------------------------------------------
# Rudra does NOT set CUDA_VISIBLE_DEVICES: a 1-GPU job sees both A100s and defaults to cuda:0, where another user's job (assigned
# the other GPU but also defaulting to cuda:0) may already hold ~74 GB.  Use the index SLURM actually assigned to THIS job.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    _idx=$(scontrol show job -d "$SLURM_JOB_ID" 2>/dev/null | grep -o 'GRES=[^ ]*' | grep -o 'IDX:[0-9,-]*' | head -1 | cut -d: -f2)
    if [ -n "$_idx" ]; then
        _idx=$(echo "$_idx" | awk -F, '{o="";for(i=1;i<=NF;i++){if($i~/-/){split($i,r,"-");for(j=r[1];j<=r[2];j++)o=o (o==""?"":",") j}else o=o (o==""?"":",") $i};print o}')
        export CUDA_VISIBLE_DEVICES="$_idx"; echo "== GPU: SLURM assigned index [$_idx] -> CUDA_VISIBLE_DEVICES=$_idx"
    else   # fallback: the GPU with the most free memory
        _idx=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -nr | head -1 | cut -d, -f1)
        if [ -n "$_idx" ]; then export CUDA_VISIBLE_DEVICES="$_idx"; echo "== GPU: no SLURM index found; picked the emptiest GPU [$_idx]"; fi
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null | sed 's/^/   /'
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "== GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (set by the system)"; fi
