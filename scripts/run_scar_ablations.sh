#!/bin/bash
# Run the SCAR default / no-extras ablation across all 8 main corpora so the
# manuscript can quote real 'default vs tuned' and 'extras vs no-extras' deltas.
# Sequential to keep memory bounded on a laptop.
set -u
cd "." || exit 1

LOG_DIR="outputs/ablations/scar_variants/_logs"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/progress.log"
rm -f "$LOG_DIR/status.txt"

# rows: dataset | jsonl subpath | n | conflict_rate
declare -a rows=(
  "hotpotqa|main/main_hotpotqa_flant5large.jsonl|800|0.0"
  "hotpotqa|main/main_hotpotqa_qwen05b.jsonl|500|0.0"
  "two_wiki|main/main_twowiki_flant5large.jsonl|400|0.0"
  "two_wiki|main/main_twowiki_qwen05b.jsonl|200|0.0"
  "musique|main/main_musique_flant5large.jsonl|300|0.0"
  "musique|main/main_musique_qwen05b.jsonl|200|0.0"
  "policybench|main/main_policybench_flant5large.jsonl|400|0.0"
  "policybench|main/main_policybench_qwen05b.jsonl|200|0.0"
)

for row in "${rows[@]}"; do
  ds=$(echo "$row" | cut -d'|' -f1)
  jf=$(echo "$row" | cut -d'|' -f2)
  n=$(echo "$row" | cut -d'|' -f3)
  cr=$(echo "$row" | cut -d'|' -f4)
  stem=$(basename "$jf" .jsonl)
  echo "$(date +%H:%M:%S) start ${stem}" >> "$LOG_DIR/progress.log"
  python3 code/pipelines/eval_scar_default.py \
    --jsonl "outputs/${jf}" --dataset "$ds" --n "$n" --seed 42 --conflict-rate "$cr" \
    > "$LOG_DIR/${stem}.log" 2>&1
  echo "$(date +%H:%M:%S) done ${stem} (rc=$?)" >> "$LOG_DIR/progress.log"
done
echo "ALL_DONE" > "$LOG_DIR/status.txt"
