#!/bin/bash
# Submit the five-method, four-dataset sweep on Agastya.  Usage:  bash submit_agastya.sh [COPIES]
# gpu-ext allows 2 running jobs per user -> 2 lane job(s), one node/V100 each.
# COPIES = chain length per lane (each copy depends afterany on the previous). A lane stops
# cleanly before the walltime (WALLTIME_EXIT); the next copy resumes, finished cells are skipped,
# and surplus copies exit within seconds, so over-long chains are harmless.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"     # repo root: PBS_O_WORKDIR must be the root (scripts cd into experiments/...)
cd "$ROOT" && mkdir -p logs
COPIES="${1:-1}"
chain () {            # $1 = script -> echoes id of the last link
    local dep="" id last=""
    for i in $(seq 1 "$COPIES"); do
        if [ -n "$dep" ]; then id=$(qsub -W depend=afterany:"$dep" "$HERE/$1"); else id=$(qsub "$HERE/$1"); fi
        dep="$id"; last="$id"
    done
    echo "$last"
}
ids=""
for f in "$HERE"/lane*.pbs; do f=$(basename "$f"); id=$(chain "$f"); ids="${ids:+$ids:}$id"; echo "$f -> last job $id"; done
qsub -W depend=afterany:"$ids" "$HERE/report.pbs"
echo "queued. watch: qstat -u $USER ;  results: results/full/summary.md"
