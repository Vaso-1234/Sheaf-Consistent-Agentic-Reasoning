#!/usr/bin/env bash
# End-to-end reproducer for the KBS submission. Assumes a working Python 3.9+
# environment with the packages in requirements.txt installed. Runs on Mac M5
# in about 3 to 4 hours of wall-clock; leaves outputs in ./outputs/ and the
# built PDF at manuscript/main.pdf.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "[1/7] Main table: FLAN-T5-Large across four datasets"
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset hotpotqa \
    --n 800 --n-samples 5 --skip-heavy --out-tag main_hotpotqa_flant5large
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset two_wiki \
    --n 400 --n-samples 5 --skip-heavy --out-tag main_twowiki_flant5large
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset musique \
    --n 300 --n-samples 5 --skip-heavy --out-tag main_musique_flant5large
python3 code/pipelines/run_main.py --model google/flan-t5-large --dataset policybench \
    --n 400 --n-samples 5 --skip-heavy --conflict-rate 0.4 \
    --out-tag main_policybench_flant5large

echo "[2/7] Cross-family: Qwen 2.5-0.5B (CPU) across four datasets"
python3 code/pipelines/run_main.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset hotpotqa \
    --n 250 --n-samples 5 --skip-heavy --out-tag main_hotpotqa_qwen05b
python3 code/pipelines/run_main.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset two_wiki \
    --n 200 --n-samples 5 --skip-heavy --out-tag main_twowiki_qwen05b
python3 code/pipelines/run_main.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset musique \
    --n 200 --n-samples 5 --skip-heavy --out-tag main_musique_qwen05b
python3 code/pipelines/run_main.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset policybench \
    --n 200 --n-samples 5 --skip-heavy --conflict-rate 0.4 \
    --out-tag main_policybench_qwen05b

echo "[3/7] Conflict-injection sweep on PolicyBench (FLAN-T5-Large)"
python3 code/pipelines/run_conflict.py --model google/flan-t5-large --dataset policybench \
    --n 250 --n-samples 5 --skip-heavy --conflict-rates 0.0 0.2 0.4 0.6

echo "[4/7] Ablations on HotpotQA (FLAN-T5-Large)"
python3 code/pipelines/run_ablations.py --model google/flan-t5-large --dataset hotpotqa \
    --n 200 --n-samples 8

echo "[5/7] Latency micro-benchmark (CPU only)"
python3 code/pipelines/run_latency.py --calls 3000

echo "[6/7] Aggregate and generate figures/tables"
python3 code/analysis/aggregate.py
python3 code/analysis/make_figures.py

echo "[7/7] Build PDF"
bash manuscript/build.sh

echo "Done. Outputs at outputs/, PDF at manuscript/main.pdf"
