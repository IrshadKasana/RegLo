#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/rudra/%x_%j.out
#SBATCH -e logs/rudra/%x_%j.err
# One lane = one A100 running a pool of pipeline processes for ONE case (see rudra_jobs.py).
# Do not sbatch this directly: use  bash experiments/pipeline/rudra/submit_rudra.sh
# Needs (exported by submit_rudra.sh): CASE PROCS WALL [GPU_BUDGET_GB MIN_FREE_GB EXCLUDE GEN MAX_GEN CPUS ...]
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT" || exit 1
mkdir -p logs/rudra
: "${CASE:?CASE not set}" "${PROCS:?PROCS not set}" "${WALL:?WALL not set}"
GEN="${GEN:-1}"; MAX_GEN="${MAX_GEN:-8}"
H=${WALL%%:*}; H=$((10#$H)); MAX_HOURS=$(awk -v h="$H" 'BEGIN{printf "%.2f", h-0.5}')   # stop 30 min early
echo "lane case=$CASE gen=$GEN/$MAX_GEN procs=$PROCS wall=$WALL max_hours=$MAX_HOURS job=${SLURM_JOB_ID:-} host=$(hostname) $(date)"
. "$ROOT/experiments/pipeline/rudra/env.sh" || exit 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / (PROCS * 2) )); [ "$OMP_NUM_THREADS" -lt 1 ] && OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || echo "WARNING: nvidia-smi failed"

resubmit() {   # same lane again (same case / sizes / exclusions); reason in $1
    if [ "$GEN" -ge "$MAX_GEN" ]; then echo "!! MAX_GEN=$MAX_GEN lanes reached for $CASE - not resubmitting. Re-run submit_rudra.sh to continue."; return; fi
    local x=(); [ -n "${EXCLUDE:-}" ] && x=(--exclude="$EXCLUDE")
    local id
    id=$(GEN=$((GEN+1)) sbatch --parsable ${x[@]+"${x[@]}"} -J "$SLURM_JOB_NAME" --time="$WALL" \
         --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" "$ROOT/experiments/pipeline/rudra/lane.sbatch")
    if [ -n "$id" ]; then echo "== resubmitted ($1): job $id (gen $((GEN+1)))"; echo "$(date) $CASE gen $((GEN+1)) job $id ($1)" >> logs/rudra/resubmits.txt
    else echo "!! resubmission failed ($1): run   bash experiments/pipeline/rudra/submit_rudra.sh $CASE   on the login node"; fi
}

# GPU guard: Rudra does not isolate GPUs well (stale processes seen on ragpu004 / ragpu006). If the GPU we got is
# not (nearly) empty, requeue this lane on another node instead of crashing every process with OOM.
if ! python - <<PY
import socket, sys, torch
need = float("${MIN_FREE_GB:-60}")
if not torch.cuda.is_available():
    print("GPU GUARD: no CUDA device visible"); sys.exit(3)
free, total = (x / 2**30 for x in torch.cuda.mem_get_info(0))
print(f"GPU GUARD: {socket.gethostname()} {torch.cuda.get_device_name(0)} free {free:.1f}/{total:.1f} GB (need {need:.0f})")
sys.exit(0 if free >= need else 3)
PY
then
    BAD="$(hostname -s)"; export EXCLUDE="${EXCLUDE:+$EXCLUDE,}$BAD"
    echo "GPU on $BAD is not free -> resubmitting with EXCLUDE=$EXCLUDE"
    resubmit "gpu guard"; exit 3
fi

python experiments/pipeline/rudra/rudra_jobs.py pool "$CASE" --procs "$PROCS" --max-hours "$MAX_HOURS" \
    --gpu-budget-gb "${GPU_BUDGET_GB:-72}"
rc=$?
echo "pool rc=$rc $(date)"
case $rc in
    0)  echo "== $CASE complete; tables in artifacts/pipeline/${EXP_PREFIX:-rudra}_$CASE/tables/" ;;
    10) resubmit "walltime, work left" ;;
    11) echo "== nothing left for this lane; other lanes of $CASE are finishing the rest" ;;
    12) echo "!! some units failed ${MAX_FAILS:-2} times: see artifacts/pipeline/${EXP_PREFIX:-rudra}_$CASE/rudra_queue/logs/" ;;
    *)  echo "!! pool crashed (rc=$rc)"; resubmit "pool rc=$rc" ;;
esac
exit 0
