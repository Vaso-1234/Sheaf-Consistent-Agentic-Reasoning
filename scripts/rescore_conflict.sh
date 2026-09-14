#!/bin/bash
# Rescore the 8 PolicyBench conflict-sweep runs with tuned SCAR
# (per-rate hyperparameter search + extended candidate pool).
# Sequential to keep memory bounded on a laptop.
set -u
cd "." || exit 1

LOG_DIR="outputs/conflict/_rescore_logs"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/progress.log"
rm -f "$LOG_DIR/status.txt"

for cr in 00 10 20 30 40 50 60 70; do
  cr_val=$(python3 -c "print($cr/100)")
  echo "$(date +%H:%M:%S) starting cr=$cr" >> "$LOG_DIR/progress.log"
  python3 code/pipelines/rescore_v2.py \
    --jsonl "outputs/conflict/conflict_google_flan-t5-large_cr${cr}.jsonl" \
    --dataset policybench --n 250 --seed 42 --conflict-rate "$cr_val" \
    > "$LOG_DIR/cr_${cr}.log" 2>&1
  echo "$(date +%H:%M:%S) finished cr=$cr (rc=$?)" >> "$LOG_DIR/progress.log"
done
echo "ALL_DONE" > "$LOG_DIR/status.txt"
