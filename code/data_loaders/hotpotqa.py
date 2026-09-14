"""HotpotQA distractor loader (Elsevier-KBS submission).

Uses HuggingFace hotpot_qa/distractor validation split. Each example has 10
paragraphs (2 gold + 8 distractors), each a list of sentences with a title.
We flatten each paragraph into a single Passage.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from typing import Iterable, Optional

from datasets import load_dataset

from .schema import Example, Passage


def load_hotpotqa(n: Optional[int] = None, split: str = "validation", qtype: Optional[str] = None):
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    out: list[Example] = []
    for row in ds:
        if qtype is not None and row.get("type") != qtype:
            continue
        supporting = set(row["supporting_facts"]["title"])
        passages: list[Passage] = []
        titles = row["context"]["title"]
        sents = row["context"]["sentences"]
        for t, sent_list in zip(titles, sents):
            text = " ".join(s.strip() for s in sent_list).strip()
            passages.append(Passage(title=t, text=text, is_gold=(t in supporting)))
        ex = Example(
            qid=row["id"],
            question=row["question"].strip(),
            answer=row["answer"].strip(),
            answer_aliases=[],
            passages=passages,
            supporting_titles=sorted(supporting),
            dataset="hotpotqa",
            qtype=row.get("type") or "",
        )
        out.append(ex)
        if n is not None and len(out) >= n:
            break
    return out


if __name__ == "__main__":
    for e in load_hotpotqa(3):
        print(e.qid, e.qtype, "|", e.question[:70], "|", e.answer)
        print("  passages:", len(e.passages), "gold:", sum(p.is_gold for p in e.passages))
