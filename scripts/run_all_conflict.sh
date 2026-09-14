#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

echo "=== Conflict sweep on PolicyBench (FLAN-T5-Large, 300 per rate) ==="
python3 code/pipelines/run_conflict.py \
  --model google/flan-t5-large \
  --dataset policybench \
  --n 300 --n-samples 5 --log-every 30 --skip-heavy \
  --conflict-rates 0.0 0.2 0.4 0.6

echo "[conflict sweep done]"
