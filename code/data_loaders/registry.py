"""Unified dataset entry point used by all pipelines."""

from __future__ import annotations

from typing import Optional

from .hotpotqa import load_hotpotqa
from .musique import load_musique
from .policybench import build_policybench
from .schema import Example
from .two_wiki import load_two_wiki


def load_examples(name: str, n: Optional[int] = None, seed: int = 42,
                  conflict_rate: float = 0.0) -> list[Example]:
    """Return N Example objects from the named dataset.

    Recognised names: hotpotqa, hotpotqa-bridge, hotpotqa-comparison,
    two_wiki, musique, policybench.
    conflict_rate applies only to policybench.
    """
    if name == "hotpotqa":
        return load_hotpotqa(n)
    if name == "hotpotqa-bridge":
        return load_hotpotqa(n, qtype="bridge")
    if name == "hotpotqa-comparison":
        return load_hotpotqa(n, qtype="comparison")
    if name == "two_wiki":
        return load_two_wiki(n)
    if name == "musique":
        return load_musique(n)
    if name == "policybench":
        return build_policybench(n or 500, seed=seed, conflict_rate=conflict_rate)
    raise ValueError(f"unknown dataset: {name}")


DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "hotpotqa-bridge": "HotpotQA-Bridge",
    "hotpotqa-comparison": "HotpotQA-Comp",
    "two_wiki": "2WikiMultiHop",
    "musique": "MuSiQue-Ans",
    "policybench": "PolicyBench-Synth",
}
