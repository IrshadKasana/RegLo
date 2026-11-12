#!/bin/bash
# Submit the five-method ResNet-18 / ImageNet-32 sweep on PARAM Rudra (SLURM).
#   [PROCS=3] [M5_MODE=full|block] bash slurm/submit_rudra.sh [PRESET=full] [COPIES=2] [LANES=4] [WALLTIME=48:00:00]
# PRESET  : smoke | full               COPIES : chain length per lane (afterany) - chaining beyond the 4-day limit
# LANES   : concurrent GPU jobs (array tasks, 1 A100-80GB each); manual: <=10 simultaneous jobs
# WALLTIME: gpu QOS of this account caps at 2 days (48:00:00) - NOT the 4 days of the manual; each lane stops cleanly 30 min before.
# DATASETS: source datasets (only "imagenet"; the OOD targets are always loaded).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT" && mkdir -p logs
PRESET="${1:-full}"; COPIES="${2:-2}"; LANES="${3:-4}"; WALL="${4:-48:00:00}"
H=${WALL%%:*}; H=$((10#$H))
# Rudra's gpu QOS (this account): MaxWall 2-00:00:00 and MaxJobsPU 5.  A longer request is accepted but stays PENDING forever
# (reason QOSMaxWallDurationPerJobLimit).  So: walltime <= 48 h, lanes <= 4 (+1 report job).
if [ "$H" -gt 48 ] && [ "${FORCE_WALL:-0}" != 1 ]; then echo "!! walltime $WALL > 48:00:00: the gpu QOS would leave it pending forever. Use 48:00:00 (FORCE_WALL=1 to override)."; exit 2; fi
if [ "$LANES" -gt 5 ] && [ "${FORCE_LANES:-0}" != 1 ]; then echo "!! LANES=$LANES > 5: QOS MaxJobsPU is 5 running jobs. Use <= 4 (FORCE_LANES=1 to override)."; exit 2; fi
MAXH=$(awk -v h="$H" 'BEGIN{printf "%.1f", h-0.5}')
SB="$HERE/sweep.sbatch"
export PRESET LANES MAX_HOURS="$MAXH"          # sbatch exports the environment to the job by default
EXCL="${EXCLUDE-ragpu004}"            # nodes to avoid (ragpu004: a stale 74 GB process sat on its GPU, Oct 2026). EXCLUDE="" disables.
XA=(); [ -n "$EXCL" ] && XA=(--exclude="$EXCL")
export DATASETS="${DATASETS:-imagenet}" PROCS="${PROCS:-3}" REGS="${REGS:-l1 l2 sam gsam gam}"
echo "preset=$PRESET lanes=$LANES copies=$COPIES walltime=$WALL datasets=[$DATASETS] procs/GPU=$PROCS exclude=[${EXCL:-none}] m5_mode=${M5_MODE:-config-default}"
dep=""
for i in $(seq 1 "$COPIES"); do
    if [ -n "$dep" ]; then id=$(sbatch --parsable ${XA[@]+"${XA[@]}"} --array=0-$((LANES-1)) --time="$WALL" --dependency=afterany:"$dep" "$SB")
    else id=$(sbatch --parsable ${XA[@]+"${XA[@]}"} --array=0-$((LANES-1)) --time="$WALL" "$SB"); fi
    if [ -z "$id" ]; then echo "!! sbatch failed for copy $i - chain stops here (copies 1..$((i-1)) are queued; re-run this command later, finished trials are skipped)"; break; fi
    echo "copy $i -> array job $id"; dep="$id"
done
rid=$(sbatch --parsable --dependency=afterany:"$dep" "$HERE/report.sbatch")
echo "report job $rid (after all lanes).  watch: squeue --me ;  results: experiments/resnet18_imagenet_sweep/results/$PRESET/"
