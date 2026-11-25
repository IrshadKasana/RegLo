#!/bin/bash
# Progress of the Agastya run(s):   bash experiments/pipeline/agastya/status.sh [case ...]
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/../../.." || exit 1
. "$HERE/env.sh" >/dev/null || exit 1
qstat -u "$USER" 2>/dev/null | tail -n +4
EXP_PREFIX="${EXP_PREFIX:-agastya}" python experiments/pipeline/rudra/rudra_jobs.py status "${@:-mlp_mnist}"
L=$(ls -t logs/agastya/*_gen*_*.log 2>/dev/null | head -1); [ -n "$L" ] && { echo "== tail of $L"; tail -15 "$L"; }
