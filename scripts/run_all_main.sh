#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

echo "=== 1/4  FLAN-T5-Large on HotpotQA (n=800) ==="
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset hotpotqa --n 800 --n-samples 5 --log-every 50 --skip-heavy --out-tag main_hotpotqa_flant5large

echo "=== 2/4  FLAN-T5-Large on 2WikiMultiHop (n=400) ==="
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset two_wiki --n 400 --n-samples 5 --log-every 25 --skip-heavy --out-tag main_twowiki_flant5large

echo "=== 3/4  FLAN-T5-Large on MuSiQue (n=300) ==="
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset musique --n 300 --n-samples 5 --log-every 25 --skip-heavy --out-tag main_musique_flant5large

echo "=== 4/4  FLAN-T5-Large on PolicyBench (n=400, cr=0.4) ==="
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset policybench --n 400 --n-samples 5 --log-every 25 --skip-heavy --conflict-rate 0.4 --out-tag main_policybench_flant5large

echo "[all main runs done]"
