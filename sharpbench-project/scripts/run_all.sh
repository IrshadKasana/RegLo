#!/usr/bin/env bash
# run_all.sh -- parallel sweep shards on Linux / macOS / WSL.
#   ./scripts/run_all.sh 4 1        # 4 shards, stage 1
#   ./scripts/run_all.sh 4 2        # 4 shards, stage 2
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs figures tables

SHARDS=${1:-4}; STAGE=${2:-1}
CONFIG=${3:-configs/sweep_grid.yaml}; WINNERS=${4:-configs/winners.yaml}

case "$STAGE" in
  1) STAGE_ARGS=(--stage1);            PREFIX=stage1 ;;
  2) [[ -f "$WINNERS" ]] || { echo "$WINNERS not found; run stage 1 + analyze first"; exit 1; }
     STAGE_ARGS=(--stage2 "$WINNERS"); PREFIX=stage2 ;;
  *) STAGE_ARGS=();                    PREFIX=full   ;;
esac

echo "Launching $SHARDS shard(s), stage $STAGE, config $CONFIG"
pids=()
for ((i=0; i<SHARDS; i++)); do
  out="results/${PREFIX}_${i}.jsonl"
  echo "  shard $i -> $out"
  python -m sharpbench sweep --config "$CONFIG" --shard "$i/$SHARDS" \
         --out "$out" "${STAGE_ARGS[@]}" > "logs/${PREFIX}_${i}.log" 2>&1 &
  pids+=($!); sleep 0.8
done

echo "Waiting for ${#pids[@]} shard(s)... (tail -f logs/${PREFIX}_0.log to watch)"
for p in "${pids[@]}"; do wait "$p"; done

echo; echo "Done. Next:"
if [[ "$STAGE" == "1" ]]; then
  echo "  python -m sharpbench analyze --results 'results/stage1_*.jsonl' --emit-winners $WINNERS"
else
  echo "  python -m sharpbench analyze --results 'results/stage2_*.jsonl' --latex tables/main.tex --csv tables/main.csv"
  echo "  python -m sharpbench plot --results 'results/stage1_*.jsonl' 'results/stage2_*.jsonl'"
fi
