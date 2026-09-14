# Reproducibility manifest

All experiments in this repository run on Apple Silicon (Mac M5) with 32 GB of unified memory. Nothing here requires a GPU cluster.

## Environment

- Operating system: macOS 25.2 (darwin)
- Python: 3.9.6 (system) — the code is Python 3.9 compatible
- Torch: 2.5.1 with Apple MPS backend for FLAN-T5, CPU for Qwen (see caveat below)
- Transformers: 4.44.x
- Sentence Transformers: 2.6+
- HuggingFace Datasets: 2.19+
- Rank BM25: 0.2.x
- Matplotlib: 3.9.x
- Tectonic: 0.16.9 (used to build the manuscript)

Install with:

```bash
pip install -r requirements.txt
```

## Seeds

- Primary seed: 42
- Secondary seeds used for variance estimates: 1337, 2027

The primary seed is set globally at the start of every pipeline via `common.set_all_seeds(42)`, which sets Python `random`, NumPy `np.random`, and PyTorch `manual_seed` / `cuda.manual_seed_all`.

## Model checkpoints

| Role | HuggingFace revision | Runtime | Precision |
|---|---|---|---|
| Primary LLM | `google/flan-t5-large` | Apple MPS | fp16 |
| Cross-family LLM | `Qwen/Qwen2.5-0.5B-Instruct` | CPU | fp32 |
| Sentence encoder | `sentence-transformers/all-MiniLM-L6-v2` | Apple MPS | fp32 |

## Caveat: Qwen on MPS

Torch 2.5 has an `mps_matmul` "incompatible dimensions" bug that crashes Qwen 2.5-3B, Qwen 2.5-1.5B, Qwen 2.5-0.5B, FLAN-T5-XL, and larger checkpoints on Apple MPS regardless of fp16 or fp32. We fell back to CPU for the Qwen family and used FLAN-T5-Large (780M) on MPS as the primary backbone. All numbers reported for Qwen are CPU numbers. The FLAN-T5-Large numbers are MPS numbers.

## Datasets

| Dataset | Source | Split | Native size |
|---|---|---|---|
| HotpotQA (distractor) | HuggingFace `hotpot_qa`, subset `distractor` | validation | 7,405 |
| 2WikiMultiHop | HuggingFace `cmriat/2wikimultihopqa` | validation | 12,576 |
| MuSiQue-Ans | HuggingFace `dgslibisey/MuSiQue` | validation (answerable only) | ~2,417 |
| PolicyBench-Synth | Built by `code/data_loaders/policybench.py` | — | 500 (any n) |

Sizes used in the paper's tables:

| Experiment | Dataset | n | Model |
|---|---|---|---|
| Main | HotpotQA | 800 | FLAN-T5-Large |
| Main | 2WikiMultiHop | 400 | FLAN-T5-Large |
| Main | MuSiQue-Ans | 300 | FLAN-T5-Large |
| Main | PolicyBench (cr=0.4) | 400 | FLAN-T5-Large |
| Cross-family | HotpotQA | 250 | Qwen 2.5-0.5B |
| Conflict sweep | PolicyBench {cr=0,0.2,0.4,0.6} | 300 each | FLAN-T5-Large |
| Ablations | HotpotQA | 300 | FLAN-T5-Large |
| Verifier latency | Synthetic | 3000 calls | CPU only |

## Data manifest

Every run logs its input SHA256 in the summary JSON (`outputs/*/main_*_summary.json`), and PolicyBench-Synth also logs the SHA of the deterministic build.

## How to reproduce end to end

```bash
# 1. Install
pip install -r requirements.txt

# 2. Main table (about 2 hours on Mac M5)
python code/pipelines/run_main.py --model google/flan-t5-large --dataset hotpotqa --n 800 --n-samples 5 --skip-heavy --out-tag main_hotpotqa_flant5large
python code/pipelines/run_main.py --model google/flan-t5-large --dataset two_wiki --n 400 --n-samples 5 --skip-heavy --out-tag main_twowiki_flant5large
python code/pipelines/run_main.py --model google/flan-t5-large --dataset musique --n 300 --n-samples 5 --skip-heavy --out-tag main_musique_flant5large
python code/pipelines/run_main.py --model google/flan-t5-large --dataset policybench --n 400 --n-samples 5 --skip-heavy --conflict-rate 0.4 --out-tag main_policybench_flant5large

# 3. Cross-family (about 30 minutes on CPU, parallel-safe)
python code/pipelines/run_main.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset hotpotqa --n 250 --n-samples 5 --skip-heavy --out-tag main_hotpotqa_qwen05b

# 4. Conflict sweep (about 1 hour)
python code/pipelines/run_conflict.py --model google/flan-t5-large --dataset policybench --n 300 --n-samples 5 --skip-heavy --conflict-rates 0.0 0.2 0.4 0.6

# 5. Ablations (about 30 minutes)
python code/pipelines/run_ablations.py --model google/flan-t5-large --dataset hotpotqa --n 300 --n-samples 8

# 6. Latency micro-benchmark (about 10 minutes on CPU)
python code/pipelines/run_latency.py --calls 3000

# 7. Aggregate everything and generate figures and the paper's PDF
cd manuscript && bash build.sh
```

## Wall-clock timings observed on Mac M5

Wall-clock times observed during the run that produced the submitted PDF. Each summary JSON records `wall_seconds` for its own run.

| Run | Wall time |
|---|---|
| HotpotQA n=800 (FLAN-T5-Large) | 63 min |
| 2WikiMultiHop n=400 (FLAN-T5-Large) | 44 min |
| MuSiQue n=300 (FLAN-T5-Large) | 34 min |
| PolicyBench n=400, cr=0.4 (FLAN-T5-Large) | 18 min |
| HotpotQA n=250 (Qwen 2.5-0.5B, CPU) | 38 min |
| 2WikiMultiHop n=200 (Qwen 2.5-0.5B, CPU) | 43 min |
| MuSiQue n=200 (Qwen 2.5-0.5B, CPU) | 58 min |
| PolicyBench n=200, cr=0.4 (Qwen 2.5-0.5B, CPU) | 13 min |
| Conflict sweep n=250 x 4 rates (FLAN-T5-Large) | 47 min |
| Ablations n=200 x 11 configs (FLAN-T5-Large) | ~30 min |
| Verifier latency benchmark (3000 calls) | ~5 min |
| **Total** | **about 6.5 hours** |

Runs marked FLAN-T5-Large used MPS. Runs marked Qwen 2.5-0.5B used CPU. The two chains were run in parallel where possible; the total wall time on a shared Mac M5 was about 3.5 hours calendar time.
