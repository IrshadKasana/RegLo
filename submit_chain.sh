#!/bin/bash
# submit_chain.sh — PBS Pro submission helper for Agastya (24h walltime).
#
#   bash submit_chain.sh                     # train -> eval -> bench -> report
#   bash submit_chain.sh sweep               # 108 exp1 sweep cells, parallel
#   bash submit_chain.sh sweep 3             # same, each cell self-chained x3 days
#   bash submit_chain.sh chain N f1.pbs ...  # submit each script N times, each
#                                            # copy depends afterany on the
#                                            # previous -> multi-day training/
#                                            # eval resumes AUTOMATICALLY
#   bash submit_chain.sh phases N            # phase flow, but every train and
#                                            # eval job is self-chained N times
#
# WHY self-chaining works: train.py writes <model>_resume.pt after every
# epoch-block and exits cleanly (WALLTIME_EXIT) before the 24h kill;
# evaluate_optimizee dumps a per-instance partial JSON. Re-running the SAME
# script resumes exactly where the previous day stopped, and a completed run
# detects its final artifact and exits in seconds — so an over-long chain is
# harmless. Set chain length N = ceil(expected_days).
set -u
cd "$(dirname "$0")"

self_chain () {                      # $1 = script, $2 = copies, $3 = extra dep
    local dep="${3:-}" id last=""
    for i in $(seq 1 "$2"); do
        if [ -n "$dep" ]; then
            id=$(qsub -W depend=afterany:"$dep" "$1")
        else
            id=$(qsub "$1")
        fi
        dep="$id"; last="$id"
    done
    echo "$last"
}

submit_dir () {                      # $1 = dir, $2 = dep list, $3 = copies
    local ids="" copies="${3:-1}"
    for f in "$1"/*.pbs; do
        [ -e "$f" ] || continue
        id=$(self_chain "$f" "$copies" "${2:-}")
        ids="${ids:+$ids:}$id"
    done
    echo "$ids"
}

case "${1:-}" in
  sweep)
    copies="${2:-1}"; n=0
    for f in pbs_scripts/sweep/*.pbs; do
        [ -e "$f" ] || continue
        self_chain "$f" "$copies" "" >/dev/null; n=$((n+1))
    done
    echo "submitted $n sweep cells (parallel, x$copies day-chain each)."
    echo "when they finish:  python pick_best_lambdas.py --apply"
    exit 0 ;;
  chain)
    copies="${2:?usage: submit_chain.sh chain N script.pbs [...]}" ; shift 2
    for f in "$@"; do
        last=$(self_chain "$f" "$copies" "")
        echo "chained x$copies: $f  (last job $last)"
    done
    exit 0 ;;
  phases|"")
    copies="${2:-1}" ;;
  *)
    echo "unknown mode '$1' (see header)"; exit 1 ;;
esac

echo "== train (x$copies day-chain each) =="
TR=$(submit_dir pbs_scripts/experiments/train "" "$copies")
echo "   $(echo "$TR" | tr ':' '\n' | grep -c . ) chains"
echo "== eval  (after train, x$copies each) =="
EV=$(submit_dir pbs_scripts/experiments/eval "$TR" "$copies")
echo "   $(echo "$EV" | tr ':' '\n' | grep -c . ) chains"
echo "== bench (after eval) =="
BE=$(submit_dir pbs_scripts/experiments/bench "$EV" 1)
echo "== report (after bench) =="
submit_dir pbs_scripts/experiments/report "$BE" 1 >/dev/null
echo
echo "All phases queued with dependencies. Watch:  qstat -u \$USER"
echo "Figures land in: artifacts/dmo/results/plots/paper/<expN>/"
