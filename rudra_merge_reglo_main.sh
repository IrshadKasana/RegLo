#!/bin/bash
# Run ON RUDRA (login node).  Unpacks a repo archive (.tgz from Agastya, or a GitHub .zip) into ~/Rego/<its top folder name>
# and puts the Rudra bundle you already assembled on top of it.
#   bash rudra_merge_reglo_main.sh [ARCHIVE] [BUNDLE_DIR] [DEST_DIR]
# defaults: ARCHIVE=~/Rego/learn-regg_v5_consolidated_1.tgz  BUNDLE_DIR=~/Rego/convnet4_rudra/learn-regg_v3_rudra  DEST_DIR=~/Rego/<top folder>
# Conservative: refuses to touch an existing DEST_DIR; never overwrites a file of the zip except the v3 sweep folder, where every
# differing file is first saved next to it as <name>.from_zip.
BASE="${BASE:-$HOME/Rego}"
ZIP="${1:-$BASE/learn-regg_v5_consolidated_1.tgz}"; D="${2:-$BASE/convnet4_rudra/learn-regg_v3_rudra}"; R="${3:-}"
V=experiments/convnet4_sweep_v3_4datasets
die() { echo "!! $*"; exit 1; }
[ -f "$ZIP" ] || die "zip not found: $ZIP"
[ -d "$D/datasets" ] && [ -d "$D/$V/slurm" ] && [ -d "$D/src/learning_regularizers" ] || die "bundle $D incomplete (need datasets/, $V/slurm, src/learning_regularizers)"
echo "== 1. unpack $ZIP"
T=$(mktemp -d "$BASE/.unz.XXXXXX") || die "cannot make temp dir in $BASE"
case "$ZIP" in *.zip) unzip -q "$ZIP" -d "$T" ;; *) tar xzf "$ZIP" -C "$T" ;; esac || { rm -rf "$T"; die "unpack failed"; }
[ "$(ls "$T" | wc -l)" = 1 ] || { rm -rf "$T"; die "expected one top folder in the archive, got: $(ls "$T")"; }
[ -n "$R" ] || R="$BASE/$(ls "$T")"
[ -e "$R" ] && { rm -rf "$T"; die "$R already exists - not touching it (delete/rename it yourself if you want to redo)"; }
mv "$T"/* "$R" && rmdir "$T"
[ -d "$R/$V" ] || die "$R/$V missing - is this the right archive?"

echo "== 2. src/ : add what the zip lacks (existing files stay)"
n=0
while IFS= read -r f; do
    if [ ! -e "$R/src/$f" ]; then mkdir -p "$R/src/$(dirname "$f")"; cp -p "$D/src/$f" "$R/src/$f"; n=$((n+1)); fi
done < <(cd "$D/src" && find . -type f -not -name '*.pyc' | sed 's#^\./##')
echo "   added $n files"
if cmp -s "$D/project_paths.py" "$R/project_paths.py"; then echo "   project_paths.py identical"; else echo "   NOTE project_paths.py differs (kept the zip's)"; fi

echo "== 3. datasets: link to the ones already uploaded (no copy)"
if [ -e "$R/datasets" ]; then echo "   $R/datasets exists - left alone"; else ln -s "$D/datasets" "$R/datasets" && echo "   $R/datasets -> $D/datasets"; fi
cp -n "$D/MD5SUMS_datasets.txt" "$R/" 2>/dev/null

echo "== 4. snapshot results from Agastya (partial, taken mid-run): set aside so Rudra starts clean"
if [ -d "$R/$V/results/full" ]; then
    mv "$R/$V/results/full" "$R/$V/results/full_agastya_snapshot" && echo "   results/full -> results/full_agastya_snapshot  (merge later with slurm/merge_results.sh)"
fi

echo "== 5. v3 sweep folder: put the Rudra version of the code + slurm/ over the zip's copy"
c=0; s=0
while IFS= read -r f; do
    dst="$R/$V/$f"
    if [ -e "$dst" ] && ! cmp -s "$D/$V/$f" "$dst"; then cp -p "$dst" "$dst.from_zip"; fi
    if [ ! -e "$dst" ] || ! cmp -s "$D/$V/$f" "$dst"; then mkdir -p "$(dirname "$dst")"; cp -p "$D/$V/$f" "$dst"; c=$((c+1)); else s=$((s+1)); fi
done < <(cd "$D/$V" && find . -type f -not -path './results/*' -not -path '*/__pycache__/*' | sed 's#^\./##')
echo "   copied/updated $c files, $s identical"
(cd "$R/$V" && find . -name '*.from_zip' | sed 's#^#   replaced (original saved): #')
mkdir -p "$R/logs"

echo "== 6. dataset checksums"
(cd "$R" && md5sum -c --quiet MD5SUMS_datasets.txt && echo "   ALL DATASET FILES OK")
echo "== done.  Next:"
echo "   cd $R && bash $V/slurm/setup_env.sh"
