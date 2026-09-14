"""Standalone conflict-gate helper used by ablations."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def top_vote_share(cands: Sequence[str]) -> float:
    if not cands:
        return 0.0
    counts = Counter(cands)
    _, top = counts.most_common(1)[0]
    return top / len(cands)


def observed_conflict(cands: Sequence[str]) -> float:
    return 1.0 - top_vote_share(cands)


def should_invoke_sheaf(cands: Sequence[str], gate: float = 0.65) -> bool:
    return top_vote_share(cands) < gate
