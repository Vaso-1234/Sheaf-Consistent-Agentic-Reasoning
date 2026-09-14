"""Common example schema used by every dataset loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Passage:
    title: str
    text: str
    is_gold: bool = False


@dataclass
class Triple:
    head: str
    relation: str
    tail: str


@dataclass
class ToolObs:
    tool: str
    payload: str


@dataclass
class Example:
    qid: str
    question: str
    answer: str
    answer_aliases: list = field(default_factory=list)
    passages: list = field(default_factory=list)   # list[Passage]
    triples: list = field(default_factory=list)    # list[Triple]
    tool_obs: list = field(default_factory=list)   # list[ToolObs]
    supporting_titles: list = field(default_factory=list)
    dataset: str = ""
    qtype: str = ""


def example_to_dict(ex: Example) -> dict:
    return {
        "qid": ex.qid,
        "question": ex.question,
        "answer": ex.answer,
        "answer_aliases": ex.answer_aliases,
        "passages": [p.__dict__ for p in ex.passages],
        "triples": [t.__dict__ for t in ex.triples],
        "tool_obs": [o.__dict__ for o in ex.tool_obs],
        "supporting_titles": ex.supporting_titles,
        "dataset": ex.dataset,
        "qtype": ex.qtype,
    }
