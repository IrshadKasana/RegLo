#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  Environment for every cluster job — sourced by the job scripts on the compute
#  node and by the submit scripts on the login node (PBS and SLURM alike).
#  Edit the marked lines once for your account.
#
#  One-time setup on the LOGIN node (it has internet; compute nodes usually do not):
#
#    conda create -y -n reglo python=3.10 && conda activate reglo
#    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
#    pip install -e ".[dev,analysis]"
#    python scripts/prepare_data.py                # downloads + meta split, before any job
#    pytest -q                                     # optional, a few minutes on CPU
#
#  The cu118 wheels bundle their own CUDA runtime, so no CUDA module is needed —
#  only an NVIDIA driver that supports CUDA 11.8 (every job's log prints the driver).
#  A100/H100 nodes: the cu121 wheels work as well (--index-url …/whl/cu121).
# ════════════════════════════════════════════════════════════════════════════

# Repository root (the submit scripts export it; fall back to this file's parent folder).
REGLO_ROOT="${REGLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REGLO_ROOT

# ── Python ──────────────────────────────────────────────────────────────────
# The interpreter of the environment above, by absolute path: no `conda activate`
# inside batch jobs. Set REGLO_PYTHON yourself to override the search.   ← edit if needed
# (the submit scripts refuse to submit if this interpreter cannot import torch and omegaconf.)
if [ -z "${REGLO_PYTHON:-}" ]; then
  if [ "${CONDA_DEFAULT_ENV:-}" = "reglo" ] && [ -x "${CONDA_PREFIX:-}/bin/python" ]; then
    REGLO_PYTHON="$CONDA_PREFIX/bin/python"
  fi
fi
if [ -z "${REGLO_PYTHON:-}" ]; then
  for candidate in \
      "$HOME/.conda/envs/reglo/bin/python" \
      "$HOME/miniconda3/envs/reglo/bin/python" \
      "$HOME/anaconda3/envs/reglo/bin/python" \
      "$HOME/miniforge3/envs/reglo/bin/python" \
      "$HOME/mambaforge/envs/reglo/bin/python" \
      "$REGLO_ROOT/.venv/bin/python"; do
    if [ -x "$candidate" ]; then REGLO_PYTHON="$candidate"; break; fi
  done
fi
if [ -z "${REGLO_PYTHON:-}" ]; then  # last resort: whatever python3 is on PATH here, by absolute path
  REGLO_PYTHON="$(command -v python3 || command -v python || echo python3)"
fi
export REGLO_PYTHON

# ── Modules ─────────────────────────────────────────────────────────────────
# Not needed with the PyTorch wheels above. Uncomment if your site insists.  ← edit if needed
# module load compilers/cuda/11.8

# ── Where data and results live ─────────────────────────────────────────────
# Defaults: datasets/ and artifacts/ inside the repository. Point them at a scratch
# file system if your home quota is small: while an M3/M4 run of ResNet-18 is in
# progress, its resume checkpoint holds the LSTM state of every weight (~3.6 GB; it
# is deleted when the run finishes).
export REGLO_DATA="${REGLO_DATA:-$REGLO_ROOT/datasets}"
export REGLO_ARTIFACTS="${REGLO_ARTIFACTS:-$REGLO_ROOT/artifacts}"
export REGLO_PROFILE="${REGLO_PROFILE:-production}"   # the cluster runs the real study

# ── Runtime ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"   # matches the 8 CPU cores the submit scripts request
export PYTHONUNBUFFERED=1                        # log lines reach the job's output file as they happen
export MPLBACKEND=Agg                            # figures without a display (make_report.py)

# max_hours for a walltime "HH:MM:SS" or "D-HH:MM:SS": stop 15 minutes (or 3 %) before the limit,
# save a checkpoint and exit with code 75 instead of being killed mid-epoch. (The runs also stop
# early when their next epoch or trajectory would not fit.)
reglo_max_hours() {
  local spec="$1" days=0 h=0 m=0 s=0
  if [[ "$spec" == *-* ]]; then days="${spec%%-*}"; spec="${spec#*-}"; fi
  IFS=: read -r h m s <<< "$spec"
  "$REGLO_PYTHON" - "$days" "${h:-0}" "${m:-0}" "${s:-0}" <<'PY'
import sys
d, h, m, s = (float(x or 0) for x in sys.argv[1:5])
total = d * 24 + h + m / 60 + s / 3600
print(f"{max(total - max(0.25, 0.03 * total), 0.05):.2f}")
PY
}
