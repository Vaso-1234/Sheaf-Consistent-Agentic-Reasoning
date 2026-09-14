"""Lightweight information-extraction subgraph builder.

For each retrieved passage we run regex + heuristic patterns to produce
short (subject, relation, object) triples. This gives a third evidence
source that is genuinely different in form from raw text and BM25 hits,
so a sheaf across the three sources has something to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    provenance: str = ""

    def as_text(self) -> str:
        return f"{self.head} | {self.relation} | {self.tail}"


_ws = re.compile(r"\s+")


def clean(s: str) -> str:
    return _ws.sub(" ", (s or "").strip())


# Patterns: capture a light, high-precision set of triples from sentences.
_PATTERNS = [
    # X is/was a Y  -> (X, is_a, Y)
    (re.compile(r"\b([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9\-\s]{2,60}?)(?=[\.,;:\(])"), "is_a"),
    # X directed Y  -> (X, directed, Y)
    (re.compile(r"\b([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})\s+(directed|produced|wrote|starred|founded|created|invented|composed|painted|discovered|led|owned|managed|acquired)\s+(?:in\s+)?(?:the\s+|a\s+|an\s+)?([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Za-z0-9\-\.']*){0,6})"), None),
    # X of Y  -> (X, of, Y) for short phrases
    (re.compile(r"\b([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})\s+of\s+([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})"), "of"),
    # X was born in Y  -> (X, born_in, Y)
    (re.compile(r"\b([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})\s+was\s+born\s+in\s+([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})"), "born_in"),
    # X died in Y
    (re.compile(r"\b([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})\s+died\s+in\s+([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,4})"), "died_in"),
    # X released in <year>
    (re.compile(r"\b([A-Z][A-Za-z0-9\-\.']*(?:\s+[A-Z][A-Za-z0-9\-\.']*){0,5})\s+(?:was\s+released|premiered|opened|debuted)\s+(?:in\s+)?(\d{4})"), "released_in"),
    # numeric parameter: "for X days/months/years"
    (re.compile(r"\b(?:for|every|within|after|of|at\s+least|no\s+more\s+than|minimum\s+of|maximum\s+of|up\s+to|exactly)\s+(\d+)\s+(days?|hours?|minutes?|weeks?|months?|years?|dollars?|percent|characters?|attempts?|business\s+days?)"), "numeric"),
]


def extract_triples_from_text(text: str, provenance: str = "") -> list[Triple]:
    text = clean(text)
    triples: list[Triple] = []
    seen = set()
    for pat, forced_rel in _PATTERNS:
        for m in pat.finditer(text):
            if forced_rel is None:
                head = clean(m.group(1))
                rel = clean(m.group(2))
                tail = clean(m.group(3))
            elif forced_rel == "numeric":
                head = "policy_parameter"
                rel = clean(m.group(2)).lower()
                tail = clean(m.group(1))
            else:
                head = clean(m.group(1))
                rel = forced_rel
                tail = clean(m.group(2))
            if not head or not tail or head.lower() == tail.lower():
                continue
            key = (head.lower(), rel.lower(), tail.lower())
            if key in seen:
                continue
            seen.add(key)
            triples.append(Triple(head=head, relation=rel, tail=tail, provenance=provenance))
    return triples


def build_subgraph(passages: Sequence, top_k: int = 10, max_triples: int = 40) -> list[Triple]:
    triples: list[Triple] = []
    for p in list(passages)[:top_k]:
        triples.extend(extract_triples_from_text(p.text, provenance=p.title))
        if len(triples) >= max_triples:
            break
    return triples[:max_triples]


def triples_to_text(triples: Sequence[Triple], limit: int = 20) -> str:
    lines = []
    for t in list(triples)[:limit]:
        lines.append(t.as_text())
    return "\n".join(lines) if lines else "(no triples extracted)"
