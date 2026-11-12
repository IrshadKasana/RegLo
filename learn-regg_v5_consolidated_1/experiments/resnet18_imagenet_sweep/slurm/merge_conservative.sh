#!/bin/bash
# CONSERVATIVE merge of one directory tree into another: nothing in DST is ever overwritten or deleted.
#   bash slurm/merge_conservative.sh SRC_DIR DST_DIR            # dry run: shows what would happen
#   bash slurm/merge_conservative.sh SRC_DIR DST_DIR --apply    # copies files that DST does not have;
#                                                               # files that DIFFER are copied next to the original as <name>.incoming
# Typical use:  SRC = a freshly unzipped bundle,  DST = an existing project root.  Review every *.incoming file by hand.
set -u
SRC="${1:?usage: merge_conservative.sh SRC_DIR DST_DIR [--apply]}"; DST="${2:?DST_DIR}"; APPLY="${3:-}"
[ -d "$SRC" ] && [ -d "$DST" ] || { echo "both SRC and DST must exist"; exit 1; }
EXCL=(--exclude='__pycache__' --exclude='*.pyc' --exclude='results/' --exclude='logs/')
new=$(rsync -rn --ignore-existing --out-format='%n' "${EXCL[@]}" "$SRC"/ "$DST"/ | grep -v '/$' | sort)
diffs=$(rsync -rcn --out-format='%n' "${EXCL[@]}" "$SRC"/ "$DST"/ | grep -v '/$' | sort | comm -23 - <(echo "$new"))
echo "== files only in SRC (would be copied): $(echo -n "$new" | grep -c .)"; echo "$new" | head -n 40
echo "== files in both but DIFFERENT (would be left alone; SRC copy saved as .incoming): $(echo -n "$diffs" | grep -c .)"; echo "$diffs"
if [ "$APPLY" = "--apply" ]; then
    rsync -r --ignore-existing "${EXCL[@]}" "$SRC"/ "$DST"/
    echo "$diffs" | while read -r f; do [ -n "$f" ] && cp -p "$SRC/$f" "$DST/$f.incoming" && echo "saved $DST/$f.incoming"; done
    echo "== applied. Nothing was overwritten."
else
    echo "== dry run only. Add --apply to copy."
fi
