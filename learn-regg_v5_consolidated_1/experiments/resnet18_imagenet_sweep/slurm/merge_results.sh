#!/bin/bash
# Union of two sweep result folders (e.g. the Agastya run and the Rudra run) into ONE, so that a single
# report / plot / trial-table can be produced.  Trials are independent files (one JSON per trial), so a merge is a copy.
#   bash slurm/merge_results.sh SRC_RESULTS_DIR DST_RESULTS_DIR            # dry run (counts, conflicts)
#   bash slurm/merge_results.sh SRC_RESULTS_DIR DST_RESULTS_DIR --apply
# SRC/DST are .../results/full folders.  Existing files in DST are never overwritten; a trial present in BOTH
# (same file name) is reported as a conflict - this happens only if both clusters were given the same dataset.
set -u
SRC="${1:?usage: merge_results.sh SRC_RESULTS_DIR DST_RESULTS_DIR [--apply]}"; DST="${2:?DST_RESULTS_DIR}"; APPLY="${3:-}"
mkdir -p "$DST"
for d in m1 m2 lo ckpt weights splits; do
    [ -d "$SRC/$d" ] || continue
    mkdir -p "$DST/$d"
    n_new=$(rsync -rn --ignore-existing --out-format='%n' "$SRC/$d"/ "$DST/$d"/ | grep -vc '/$')
    n_conf=$(rsync -rcn --out-format='%n' "$SRC/$d"/ "$DST/$d"/ | grep -v '/$' | wc -l)
    n_conf=$((n_conf - n_new))
    echo "$d: $n_new new file(s), $n_conf already present but different (kept as in DST)"
    [ "$APPLY" = "--apply" ] && rsync -r --ignore-existing "$SRC/$d"/ "$DST/$d"/
done
[ "$APPLY" = "--apply" ] && echo "== merged into $DST. Now:  python run_sweep.py report --preset full ; python plot_sweep_results.py ; python make_trial_table.py" \
                         || echo "== dry run only. Add --apply."
