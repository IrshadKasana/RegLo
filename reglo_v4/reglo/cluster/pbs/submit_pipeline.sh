#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  The whole study, stage by stage, as chained PBS Pro jobs (Agastya):
#
#    Stage 1   for every round of configs/rounds.yaml (A: ConvNet-4 / SVHN, B: MLP / FMNIST):
#              m3_lstm (1 task) and m4_lstm_reg × {l1,l2,sam,gam,gsam} (array of 5) on the
#              round's source pair, and M2 = AdamW + r on that pair (starts at once)
#    Stage 2   optimizer evaluation  ┐ every pair: M1 and M2 start at once (AdamW needs no LSTM);
#    Stage 3   optimizee evaluation  ┘ M3/M4 of round R wait for round R's Stage-1 jobs
#    Stage 4   same-GPU compute profiles, after Stage 1 (the audits need finished runs: see the end)
#
#    bash cluster/pbs/submit_pipeline.sh                    # the core study (production profile)
#    bash cluster/pbs/submit_pipeline.sh --rounds A         # round A only (B later: --rounds B)
#    bash cluster/pbs/submit_pipeline.sh --extended         # … plus WRN-28, ViT-Lite, CIFAR-100
#    bash cluster/pbs/submit_pipeline.sh --pilot            # … plus the M1-vs-M3 pilot
#    bash cluster/pbs/submit_pipeline.sh --profile smoke    # the smoke test as cluster jobs (minutes)
#    bash cluster/pbs/submit_pipeline.sh --dry-run          # print the submit commands only
#    bash cluster/pbs/submit_pipeline.sh --walltime 96:00:00
#
#  --rounds B after round A: M1 runs that already finished return at once.
#  Other options are passed to every submit.sh call. If a Stage-1 job fails or stops at
#  max_hours (exit 75), PBS removes the arrays waiting for it; resubmit that Stage-1 job
#  (it resumes) and then those arrays, e.g.
#      bash cluster/pbs/submit.sh stage2_optimizer_evaluation --only method=m4_lstm_reg --only round=A --after <new id>
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGLO_ROOT="$(cd "$HERE/../.." && pwd)"
export REGLO_ROOT
# shellcheck source=../env.sh
source "$REGLO_ROOT/cluster/env.sh"
submit() { bash "$HERE/submit.sh" "$@"; }
rounds_of() {  # the rounds an experiment file lists
  "$REGLO_PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/src")
from reglo.config import load_experiment
print(" ".join(map(str, load_experiment(sys.argv[2]).get("rounds") or [])))' "$REGLO_ROOT" "$1"
}

EVALUATION=(stage2_optimizer_evaluation stage3_optimizee_evaluation)
PILOT=0
ROUNDS=""
PASS=()
PROFILE="${REGLO_PROFILE:-production}"
while [ $# -gt 0 ]; do
  case "$1" in
    --extended) EVALUATION+=(extended_evaluation); shift ;;
    --pilot)    PILOT=1; shift ;;
    --rounds)   ROUNDS="$2"; shift 2 ;;
    --profile)  PROFILE="$2"; export REGLO_PROFILE="$2"; PASS+=("$1" "$2"); shift 2 ;;
    *)          PASS+=("$1"); shift ;;
  esac
done
set -- ${PASS[@]+"${PASS[@]}"}
[ -n "$ROUNDS" ] || ROUNDS="$(rounds_of stage1_meta_training)"
wanted() { case " $ROUNDS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Stage 1, round by round
declare -A M3 M4
ALL=""
for R in $ROUNDS; do
  M3[$R]=$(submit stage1_meta_training --kind meta_train --only meta_training=m3_lstm --only round="$R" "$@")
  M4[$R]=$(submit stage1_meta_training --kind meta_train --only meta_training=m4_lstm_reg --only round="$R" "$@")
  submit stage1_meta_training --kind train --only round="$R" "$@" >/dev/null   # M2 = AdamW + r on the source
  ALL="${ALL:+$ALL:}${M3[$R]}:${M4[$R]}"
done

# Stages 2 and 3 (and the optional grids): M1 and M2 once, M3/M4 per round
if [ "$PILOT" -eq 1 ] && wanted A; then
  submit pilot --only method=m1_adamw "$@" >/dev/null
  submit pilot --only method=m3_lstm --after "${M3[A]}" "$@" >/dev/null
fi
for experiment in "${EVALUATION[@]}"; do
  submit "$experiment" --only method=m1_adamw "$@" >/dev/null
  submit "$experiment" --only method=m2_adamw_reg "$@" >/dev/null   # AdamW + r (source pairs: Stage 1's)
  for R in $(rounds_of "$experiment"); do
    wanted "$R" || continue
    submit "$experiment" --only method=m3_lstm --only round="$R" --after "${M3[$R]}" "$@" >/dev/null
    submit "$experiment" --only method=m4_lstm_reg --only round="$R" --after "${M4[$R]}" "$@" >/dev/null
  done
done

# Stage 4: timing now; audits once the runs exist
submit stage4_analysis --kind compute --after "$ALL" "$@" >/dev/null

cat >&2 <<MSG

All submitted (profile $PROFILE, rounds $ROUNDS). Watch with:   watch -n 60 qstat -u \$USER -t
Check what has finished:     python scripts/check_study.py --profile $PROFILE
When Stages 2–3 have finished:
  bash cluster/pbs/submit.sh stage4_analysis --kind diagnostics --profile $PROFILE
and then the 3D tables and figures:
  python scripts/make_report.py --profile $PROFILE
MSG
