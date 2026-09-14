#!/bin/bash
# Evaluate SCAR with a single pre-declared default config across the four
# FLAN-T5-Large main corpora, without touching the summaries. Prints hit
# accuracy per corpus so the manuscript can quote a real "default vs tuned"
# delta honestly.
set -u
cd "." || exit 1

LOG_DIR="outputs/main/_default_eval"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/progress.log"

declare -a rows=(
  "hotpotqa|main/main_hotpotqa_flant5large.jsonl|800|0.0"
  "two_wiki|main/main_twowiki_flant5large.jsonl|400|0.0"
  "musique|main/main_musique_flant5large.jsonl|300|0.0"
  "policybench|main/main_policybench_flant5large.jsonl|400|0.0"
)

for row in "${rows[@]}"; do
  ds=$(echo "$row" | cut -d'|' -f1)
  jf=$(echo "$row" | cut -d'|' -f2)
  n=$(echo "$row" | cut -d'|' -f3)
  cr=$(echo "$row" | cut -d'|' -f4)
  echo "$(date +%H:%M:%S) start ${ds}" >> "$LOG_DIR/progress.log"
  python3 code/pipelines/eval_scar_default.py \
    --jsonl "outputs/${jf}" --dataset "$ds" --n "$n" --seed 42 --conflict-rate "$cr" \
    > "$LOG_DIR/${ds}.log" 2>&1
  echo "$(date +%H:%M:%S) done ${ds} (rc=$?)" >> "$LOG_DIR/progress.log"
done
echo "ALL_DONE" > "$LOG_DIR/status.txt"
