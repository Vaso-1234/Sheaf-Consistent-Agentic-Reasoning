# SCAR: Sheaf-Consistent Agentic Reasoning

Anonymised code and reproducibility package for the Knowledge-Based Systems
submission *"Sheaf-Consistent Agentic Reasoning: A Knowledge-Component Verifier
for Retrieval-Augmented Decision Support"*.

Author names and affiliations are withheld for double-blind review.

## What this repository contains

- `code/` - data loaders, retrievers, SCAR verifier, baselines, analysis scripts
- `scripts/` - convenience drivers for rescoring and ablations
- `outputs/` - per-run JSONL predictions and aggregate summaries
- `manuscript/` - anonymised LaTeX source, figures, and compiled PDF
- `requirements.txt`, `run_all.sh`, `REPRODUCIBILITY.md`

## Datasets

| Dataset | Link |
|---|---|
| HotpotQA (distractor) | https://huggingface.co/datasets/hotpot_qa |
| 2WikiMultihopQA | https://huggingface.co/datasets/cmriat/2wikimultihopqa |
| MuSiQue | https://huggingface.co/datasets/dgslibisey/MuSiQue |
| PolicyBench-Synth | Synthetic; built by `code/data_loaders/policybench.py` (deterministic given seed / conflict rate) |

Public benchmarks are downloaded automatically on first run via HuggingFace Datasets.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh
python3 code/analysis/aggregate.py
python3 code/analysis/make_figures.py
```

Primary seed: `42`. See `REPRODUCIBILITY.md` for the full environment.

## License

Code released under the MIT License (see `LICENSE`). Public datasets remain under their original licenses.
