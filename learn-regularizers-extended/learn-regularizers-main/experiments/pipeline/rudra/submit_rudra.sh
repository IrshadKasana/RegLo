#!/bin/bash
# Submit the full pipeline on PARAM Rudra for the three cases (run on the LOGIN node, from anywhere):
#
#   bash experiments/pipeline/rudra/submit_rudra.sh                      # all three cases, default lanes
#   bash experiments/pipeline/rudra/submit_rudra.sh mlp_mnist            # one case (also: to restart a case)
#   LANES=2 bash experiments/pipeline/rudra/submit_rudra.sh resnet18_imagenet
#
# Default layout = 5 GPU jobs, the gpu-QOS limit of this account (MaxJobsPU 5, MaxWall 2 days):
#   mlp_mnist          1 lane  x 6 processes  (CPU-bound small nets; all 24 units on one A100)
#   convnet4_cifar10   1 lane  x 6 processes
#   resnet18_imagenet  3 lanes x <=3 processes (GPU-memory-bound: meta-training ResNet-18 width-16 proxy)
# Lanes stop 30 min before the walltime and resubmit themselves while work is left (up to MAX_GEN generations),
# so nothing has to be chained or held in the queue. Re-running this script is safe: finished work is skipped.
#
# Environment knobs (all optional):
#   WALL=48:00:00  CPUS=32  EXCLUDE=ragpu004  MAX_GEN=8  NUM_WORKERS=3  DATA_ROOT=/path/to/datasets
#   R18_META_WIDTH=16 (ResNet-18 proxy width for meta-training)  MEM_SCALE=1.0 (scale the GPU-GB estimates)
#   GPU_BUDGET_GB=72  MIN_FREE_GB=60  EXP_PREFIX=rudra  EXTRA_SETS="key=value ..." (extra --set for all units)
#   LANES / PROCS: override the per-case defaults (only when one case is given)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT" && mkdir -p logs/rudra
WALL="${WALL:-48:00:00}"
H=${WALL%%:*}; H=$((10#$H))
if [ "$H" -gt 48 ] && [ "${FORCE_WALL:-0}" != 1 ]; then
    echo "!! WALL=$WALL > 48:00:00: Rudra's gpu QOS (this account) would keep the job PENDING forever. FORCE_WALL=1 to override."; exit 2; fi
CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && CASES=(mlp_mnist convnet4_cifar10 resnet18_imagenet)

declare -A DEF_LANES=([mlp_mnist]=1 [convnet4_cifar10]=1 [resnet18_imagenet]=3)
declare -A DEF_PROCS=([mlp_mnist]=6 [convnet4_cifar10]=6 [resnet18_imagenet]=3)
declare -A SHORT=([mlp_mnist]=rgMLP [convnet4_cifar10]=rgCNN [resnet18_imagenet]=rgR18)

total=0
for c in "${CASES[@]}"; do
    [ -n "${DEF_LANES[$c]:-}" ] || { echo "!! unknown case '$c' (mlp_mnist | convnet4_cifar10 | resnet18_imagenet)"; exit 2; }
    n=${DEF_LANES[$c]}; [ ${#CASES[@]} -eq 1 ] && n="${LANES:-$n}"; total=$((total + n))
done
running=$(squeue -h -u "$USER" -p gpu 2>/dev/null | wc -l)
if [ $((total + running)) -gt 5 ] && [ "${FORCE_LANES:-0}" != 1 ]; then
    echo "!! $total new + $running existing gpu jobs > 5 (QOS MaxJobsPU). Extra jobs only wait in the queue."
    echo "   Submit fewer (e.g. one case, LANES=1) or FORCE_LANES=1 to submit anyway."; exit 2
fi

export WALL EXCLUDE="${EXCLUDE-ragpu004}" MAX_GEN="${MAX_GEN:-8}" NUM_WORKERS="${NUM_WORKERS:-3}" \
       R18_META_WIDTH="${R18_META_WIDTH:-16}" MEM_SCALE="${MEM_SCALE:-1.0}" GPU_BUDGET_GB="${GPU_BUDGET_GB:-72}" \
       MIN_FREE_GB="${MIN_FREE_GB:-60}" EXP_PREFIX="${EXP_PREFIX:-rudra}" EXTRA_SETS="${EXTRA_SETS:-}" GEN=1
[ -n "${DATA_ROOT:-}" ] && export DATA_ROOT
CPUS="${CPUS:-32}"
XA=(); [ -n "$EXCLUDE" ] && XA=(--exclude="$EXCLUDE")
for c in "${CASES[@]}"; do
    n=${DEF_LANES[$c]}; p=${DEF_PROCS[$c]}
    if [ ${#CASES[@]} -eq 1 ]; then n="${LANES:-$n}"; p="${PROCS:-$p}"; fi
    for i in $(seq 1 "$n"); do
        id=$(CASE="$c" PROCS="$p" sbatch --parsable ${XA[@]+"${XA[@]}"} -J "${SHORT[$c]}" --time="$WALL" \
             --cpus-per-task="$CPUS" "$HERE/lane.sbatch")
        if [ -n "$id" ]; then echo "$c lane $i/$n -> job $id  (procs $p, wall $WALL)"; echo "$(date) $c lane $i job $id" >> logs/rudra/resubmits.txt
        else echo "!! sbatch failed for $c lane $i"; fi
    done
done
echo
echo "watch:    squeue --me        progress: python experiments/pipeline/rudra/rudra_jobs.py status"
echo "lane log: logs/rudra/<rgMLP|rgCNN|rgR18>_<jobid>.out    unit logs: artifacts/pipeline/${EXP_PREFIX}_<case>/rudra_queue/logs/"
