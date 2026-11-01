#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  Submit one experiment grid as ONE PBS Pro array job (one array element per run).
#
#    bash cluster/pbs/submit.sh stage1_meta_training
#    bash cluster/pbs/submit.sh stage2_optimizer_evaluation --only method=m3_lstm --after 12345.agastya
#    bash cluster/pbs/submit.sh stage3_optimizee_evaluation --only optimizee=resnet18 --walltime 72:00:00
#    bash cluster/pbs/submit.sh stage4_analysis --kind diagnostics        # once Stages 2–3 finished
#    bash cluster/pbs/submit.sh stage2_optimizer_evaluation --profile smoke   # the smoke test, as jobs
#    bash cluster/pbs/submit.sh stage2_optimizer_evaluation --dry-run     # show the qsub command only
#
#  qsub options (defaults in brackets, or set REGLO_QUEUE / REGLO_SELECT / REGLO_WALLTIME):
#    --queue NAME          [gpu-ext]
#    --select SPEC         [1:ncpus=8:ngpus=1:mem=24gb]
#    --walltime HH:MM:SS   [48:00:00]   runs stop cleanly ~3 % before it (max_hours) and resume
#    --after JOBID         start only after JOBID finished successfully (repeatable)
#    --dry-run             build and validate the manifest, print the qsub command
#  Everything else goes to scripts/make_manifest.py: --profile …, --kind …, --only key=value …
#  (the profile defaults to $REGLO_PROFILE, which cluster/env.sh sets to production).
#
#  The job reads a frozen copy of the manifest (manifests/submitted/), so rebuilding a
#  manifest later never shifts the lines of jobs that are still queued.
#
#  stdout carries only the job id, so scripts can chain jobs:
#    META=$(bash cluster/pbs/submit.sh stage1_meta_training)
#    bash cluster/pbs/submit.sh stage2_optimizer_evaluation --after "$META"
#  Job output: cluster_logs/<name>.o<job>[.<index>]
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

usage() {  # the comment block above
  awk 'NR > 2 && /^#/ { sub(/^# ?/, ""); if ($0 !~ /^═/) print; next } NR > 2 { exit }' "$0" >&2
  exit "${1:-2}"
}

[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage 0 ;; esac
EXPERIMENT="$1"; shift

QUEUE="${REGLO_QUEUE:-gpu-ext}"
SELECT="${REGLO_SELECT:-1:ncpus=8:ngpus=1:mem=24gb}"
WALLTIME="${REGLO_WALLTIME:-48:00:00}"
AFTER=""
DRY_RUN=0
MAKE_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --queue)    QUEUE="$2"; shift 2 ;;
    --select)   SELECT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --after)    AFTER="${AFTER:+$AFTER:}$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --profile)  export REGLO_PROFILE="$2"; MAKE_ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help)  usage 0 ;;
    *)          MAKE_ARGS+=("$1"); shift ;;
  esac
done

REGLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REGLO_ROOT
# shellcheck source=../env.sh
source "$REGLO_ROOT/cluster/env.sh"

if ! "$REGLO_PYTHON" -c "import torch, omegaconf" >/dev/null 2>&1; then
  echo "error: $REGLO_PYTHON cannot import torch and omegaconf; set REGLO_PYTHON to your environment's" \
       "python (see cluster/env.sh)" >&2
  exit 2
fi

# Build and validate the run list on the login node (YAML composition only, no GPU).
MANIFEST="$("$REGLO_PYTHON" "$REGLO_ROOT/scripts/make_manifest.py" "$EXPERIMENT" \
            ${MAKE_ARGS[@]+"${MAKE_ARGS[@]}"} --print-path)"
COUNT="$(grep -c -v -E '^[[:space:]]*(#|$)' "$MANIFEST" || true)"
if [ "${COUNT:-0}" -lt 1 ]; then
  echo "error: $MANIFEST has no runs" >&2
  exit 2
fi
if [ "$DRY_RUN" -eq 0 ]; then  # freeze what this job will read
  FROZEN="$(dirname "$MANIFEST")/submitted/$(basename "${MANIFEST%.txt}").$(date +%Y%m%d-%H%M%S)-$$.txt"
  mkdir -p "$(dirname "$FROZEN")"
  cp "$MANIFEST" "$FROZEN"
  MANIFEST="$FROZEN"
fi

MAX_HOURS="$(reglo_max_hours "$WALLTIME")"
NAME="rg_$(printf '%s' "$EXPERIMENT" | tr -c 'A-Za-z0-9_' '_' | cut -c1-12)"
VARS="MANIFEST=$MANIFEST,REGLO_ROOT=$REGLO_ROOT,REGLO_PYTHON=$REGLO_PYTHON,REGLO_MAX_HOURS=$MAX_HOURS"
VARS="$VARS,REGLO_DATA=$REGLO_DATA,REGLO_ARTIFACTS=$REGLO_ARTIFACTS,REGLO_PROFILE=$REGLO_PROFILE"
[ -n "${REGLO_CONFIGS:-}" ] && VARS="$VARS,REGLO_CONFIGS=$REGLO_CONFIGS"
QSUB=(qsub -N "$NAME" -q "$QUEUE" -l "select=$SELECT" -l "walltime=$WALLTIME" -j oe)
if [ "$COUNT" -gt 1 ]; then
  QSUB+=(-J "0-$((COUNT - 1))")          # PBS array: PBS_ARRAY_INDEX = manifest line
else
  VARS="$VARS,INDEX=0"                   # PBS Pro arrays need ≥ 2 elements: plain job
fi
QSUB+=(-v "$VARS")
[ -n "$AFTER" ] && QSUB+=(-W "depend=afterok:$AFTER")
QSUB+=("$REGLO_ROOT/cluster/pbs/run_array.pbs")

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry run] $COUNT run(s) from $MANIFEST (max_hours $MAX_HOURS)" >&2
  printf '%q ' "${QSUB[@]}" >&2; echo >&2
  echo "dryrun_${NAME}"
  exit 0
fi

mkdir -p "$REGLO_ROOT/cluster_logs"
JOB="$(cd "$REGLO_ROOT/cluster_logs" && "${QSUB[@]}")"
echo "submitted $EXPERIMENT ${MAKE_ARGS[*]:-}: $COUNT run(s) as $JOB${AFTER:+ (after $AFTER)} — $MANIFEST" >&2
echo "$JOB"
