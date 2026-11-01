#!/bin/bash
# Submit the ConvNet-4 lambda sweep on PARAM Rudra (SLURM).
#   bash slurm/submit_rudra.sh [PRESET=full] [COPIES=1] [LANES=4] [WALLTIME=24:00:00]
# PRESET : smoke | fast | full          COPIES: chain length per lane (afterany) - day-chaining
# LANES  : concurrent GPU jobs (array tasks); manual: <=10 simultaneous jobs, 2 A100 per GPU node
# WALLTIME: manual: max 4 days (96:00:00) under the default QoS; the lane stops cleanly 30 min before.
# Run from anywhere; the script cds to the project root first (sbatch --chdir semantics = SLURM_SUBMIT_DIR).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT" && mkdir -p logs
PRESET="${1:-full}"; COPIES="${2:-1}"; LANES="${3:-4}"; WALL="${4:-24:00:00}"
H=${WALL%%:*}; H=$((10#$H))
MAXH=$(awk -v h="$H" 'BEGIN{printf "%.1f", h-0.5}')
SB="$HERE/sweep.sbatch"
export PRESET LANES MAX_HOURS="$MAXH"          # sbatch exports the environment to the job by default
dep=""
for i in $(seq 1 "$COPIES"); do
    if [ -n "$dep" ]; then id=$(sbatch --parsable --array=0-$((LANES-1)) --time="$WALL" --dependency=afterany:"$dep" "$SB")
    else id=$(sbatch --parsable --array=0-$((LANES-1)) --time="$WALL" "$SB"); fi
    echo "copy $i -> array job $id"; dep="$id"
done
rid=$(sbatch --parsable --dependency=afterany:"$dep" "$HERE/report.sbatch")
echo "report job $rid (after all lanes).  watch: squeue --me ;  results: experiments/convnet4_lambda_sweep_svhn_cifar10/results/$PRESET/summary.md"
