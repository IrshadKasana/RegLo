#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  Submit one experiment grid as ONE SLURM array job (one array task per run) —
#  PARAM supercomputers and other SLURM clusters.
#
#    bash cluster/slurm/submit.sh stage1_meta_training
#    bash cluster/slurm/submit.sh stage2_optimizer_evaluation --only method=m3_lstm --after 123456
#    bash cluster/slurm/submit.sh stage3_optimizee_evaluation --only optimizee=resnet18 --time 3-00:00:00
#    bash cluster/slurm/submit.sh stage4_analysis --kind diagnostics      # once Stages 2–3 finished
#    bash cluster/slurm/submit.sh stage2_optimizer_evaluation --profile smoke  # the smoke test, as jobs
#    bash cluster/slurm/submit.sh stage2_optimizer_evaluation --dry-run   # show the sbatch command only
#
#  sbatch options (defaults in brackets, or the environment variable named):
#    --partition NAME      [gpu]           REGLO_PARTITION   (sinfo -s lists your cluster's)
#    --gres SPEC           [gpu:1]         REGLO_GRES        (e.g. gpu:a100:1)
#    --cpus N              [8]             REGLO_CPUS
#    --mem SIZE            [32G]           REGLO_MEM
#    --time D-HH:MM:SS     [2-00:00:00]    REGLO_TIME        runs stop cleanly ~3 % before it (max_hours)
#    --account NAME        []              REGLO_ACCOUNT     (if your site needs -A)
#    --qos NAME            []              REGLO_QOS
#    --max-parallel N      [all]           at most N tasks of the array at once
#    --after JOBID         start only after JOBID finished successfully (repeatable)
#    --no-requeue          a run stopped at max_hours exits 75 instead of requeueing itself
#    --dry-run             build and validate the manifest, print the sbatch command
#  Everything else goes to scripts/make_manifest.py: --profile …, --kind …, --only key=value …
#  (the profile defaults to $REGLO_PROFILE, which cluster/env.sh sets to production).
#
#  A run that reaches max_hours saves a checkpoint and requeues its own array task
#  (scontrol requeue), so long Stage-1 runs continue across walltimes and the jobs
#  waiting for them keep waiting. Where users may not requeue, submit the same command
#  again: finished runs return at once, interrupted ones resume.
#
#  stdout carries only the job id, so scripts can chain jobs:
#    META=$(bash cluster/slurm/submit.sh stage1_meta_training)
#    bash cluster/slurm/submit.sh stage2_optimizer_evaluation --after "$META"
#  Job output: cluster_logs/<name>-<job>_<task>.out
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

usage() {  # the comment block above
  awk 'NR > 2 && /^#/ { sub(/^# ?/, ""); if ($0 !~ /^═/) print; next } NR > 2 { exit }' "$0" >&2
  exit "${1:-2}"
}

[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage 0 ;; esac
EXPERIMENT="$1"; shift

PARTITION="${REGLO_PARTITION:-gpu}"
GRES="${REGLO_GRES:-gpu:1}"
CPUS="${REGLO_CPUS:-8}"
MEM="${REGLO_MEM:-32G}"
TIME="${REGLO_TIME:-2-00:00:00}"
ACCOUNT="${REGLO_ACCOUNT:-}"
QOS="${REGLO_QOS:-}"
MAX_PARALLEL=""
AFTER=""
REQUEUE=1
DRY_RUN=0
MAKE_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --partition)    PARTITION="$2"; shift 2 ;;
    --gres)         GRES="$2"; shift 2 ;;
    --cpus)         CPUS="$2"; shift 2 ;;
    --mem)          MEM="$2"; shift 2 ;;
    --time)         TIME="$2"; shift 2 ;;
    --account)      ACCOUNT="$2"; shift 2 ;;
    --qos)          QOS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --after)        AFTER="${AFTER:+$AFTER:}$2"; shift 2 ;;
    --no-requeue)   REQUEUE=0; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --profile)      export REGLO_PROFILE="$2"; MAKE_ARGS+=("$1" "$2"); shift 2 ;;
    -h|--help)      usage 0 ;;
    *)              MAKE_ARGS+=("$1"); shift ;;
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

MAX_HOURS="$(reglo_max_hours "$TIME")"
NAME="rg_$(printf '%s' "$EXPERIMENT" | tr -c 'A-Za-z0-9_' '_' | cut -c1-12)"
ARRAY="0-$((COUNT - 1))${MAX_PARALLEL:+%$MAX_PARALLEL}"
EXPORTS="ALL,MANIFEST=$MANIFEST,REGLO_ROOT=$REGLO_ROOT,REGLO_PYTHON=$REGLO_PYTHON,REGLO_MAX_HOURS=$MAX_HOURS"
EXPORTS="$EXPORTS,REGLO_DATA=$REGLO_DATA,REGLO_ARTIFACTS=$REGLO_ARTIFACTS,REGLO_PROFILE=$REGLO_PROFILE"
EXPORTS="$EXPORTS,REGLO_REQUEUE=$REQUEUE"
[ -n "${REGLO_CONFIGS:-}" ] && EXPORTS="$EXPORTS,REGLO_CONFIGS=$REGLO_CONFIGS"
SBATCH=(sbatch --parsable -J "$NAME" -p "$PARTITION" --gres="$GRES" --cpus-per-task="$CPUS" --mem="$MEM"
        --time="$TIME" --array="$ARRAY" --export="$EXPORTS"
        --output="$REGLO_ROOT/cluster_logs/%x-%A_%a.out")
[ "$REQUEUE" -eq 1 ] && SBATCH+=(--requeue)
[ -n "$ACCOUNT" ] && SBATCH+=(--account="$ACCOUNT")
[ -n "$QOS" ] && SBATCH+=(--qos="$QOS")
[ -n "$AFTER" ] && SBATCH+=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)
SBATCH+=("$REGLO_ROOT/cluster/slurm/run_array.slurm")

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry run] $COUNT run(s) from $MANIFEST (max_hours $MAX_HOURS)" >&2
  printf '%q ' "${SBATCH[@]}" >&2; echo >&2
  echo "dryrun_${NAME}"
  exit 0
fi

mkdir -p "$REGLO_ROOT/cluster_logs"
JOB="$(cd "$REGLO_ROOT" && "${SBATCH[@]}")"
JOB="${JOB%%;*}"   # --parsable prints "id" or "id;cluster"
echo "submitted $EXPERIMENT ${MAKE_ARGS[*]:-}: $COUNT run(s) as $JOB${AFTER:+ (after $AFTER)} — $MANIFEST" >&2
echo "$JOB"
