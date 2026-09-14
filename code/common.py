"""Shared config: seeds, paths, device pick, deterministic hashing."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = REPO_ROOT / "outputs"
POLICY_DIR = DATA_DIR / "policybench"
WIKIKG_DIR = DATA_DIR / "wikikg"

for d in (DATA_DIR, CACHE_DIR, OUTPUTS_DIR, POLICY_DIR, WIKIKG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 1337, 2027)
PRIMARY_SEED = SEEDS[0]


def set_all_seeds(seed: int = PRIMARY_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = _punct_re.sub(" ", t)
    return _ws_re.sub(" ", t).strip()


def token_set(text: str) -> set:
    return set(normalize_text(text).split())


def em_score(pred: str, gold: str) -> float:
    p, g = normalize_text(pred), normalize_text(gold)
    if not g:
        return 0.0
    if p == g:
        return 1.0
    return 0.0


def f1_score(pred: str, gold: str) -> float:
    p, g = token_set(pred), token_set(gold)
    if not p or not g:
        return float(p == g)
    common = p & g
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def loose_em(pred: str, gold: str) -> float:
    """Contains-style EM used for hard multi-hop cases (pred contains gold or vice versa)."""
    p, g = normalize_text(pred), normalize_text(gold)
    if not g:
        return 0.0
    if p == g or g in p or p in g:
        return 1.0
    return 0.0


def bootstrap_ci(values, n_boot: int = 500, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sha256_of_iter(items: Iterable) -> str:
    h = hashlib.sha256()
    for x in items:
        h.update(json.dumps(x, sort_keys=True, default=str).encode())
    return h.hexdigest()


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def env_or(default: str, key: str) -> str:
    return os.environ.get(key, default)
