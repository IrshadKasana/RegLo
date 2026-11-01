#!/bin/bash
# Submit the five-method, four-dataset sweep on PARAM Rudra (SLURM).
#   [DATASETS="mnist fmnist"] [PROCS=4] [M5_MODE=full|block] bash slurm/submit_rudra.sh [PRESET=full] [COPIES=1] [LANES=4] [WALLTIME=96:00:00]
# PRESET  : smoke | full               COPIES : chain length per lane (afterany) - chaining beyond the 4-day limit
# LANES   : concurrent GPU jobs (array tasks, 1 A100-80GB each); manual: <=10 simultaneous jobs
# WALLTIME: manual: max 4 days (96:00:00) on the default QoS; each lane stops cleanly 30 min before.
# DATASETS: which datasets THIS cluster computes (default: all four). Use disjoint subsets on two clusters, then merge
#           the result folders with slurm/merge_results.sh.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT" && mkdir -p logs
PRESET="${1:-full}"; COPIES="${2:-1}"; LANES="${3:-4}"; WALL="${4:-96:00:00}"
H=${WALL%%:*}; H=$((10#$H))
MAXH=$(awk -v h="$H" 'BEGIN{printf "%.1f", h-0.5}')
SB="$HERE/sweep.sbatch"
export PRESET LANES MAX_HOURS="$MAXH"          # sbatch exports the environment to the job by default
export DATASETS="${DATASETS:-svhn cifar10 mnist fmnist}" PROCS="${PROCS:-4}" REGS="${REGS:-l1 l2 sam gsam gam}"
echo "preset=$PRESET lanes=$LANES copies=$COPIES walltime=$WALL datasets=[$DATASETS] procs/GPU=$PROCS m5_mode=${M5_MODE:-config-default}"
dep=""
for i in $(seq 1 "$COPIES"); do
    if [ -n "$dep" ]; then id=$(sbatch --parsable --array=0-$((LANES-1)) --time="$WALL" --dependency=afterany:"$dep" "$SB")
    else id=$(sbatch --parsable --array=0-$((LANES-1)) --time="$WALL" "$SB"); fi
    echo "copy $i -> array job $id"; dep="$id"
done
rid=$(sbatch --parsable --dependency=afterany:"$dep" "$HERE/report.sbatch")
echo "report job $rid (after all lanes).  watch: squeue --me ;  results: experiments/convnet4_sweep_v3_4datasets/results/$PRESET/"
