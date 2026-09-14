"""MuSiQue-Ans loader via dgslibisey/MuSiQue mirror. Only answerable questions."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from typing import Optional

from datasets import load_dataset

from .schema import Example, Passage


def load_musique(n: Optional[int] = None, split: str = "validation"):
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    out: list[Example] = []
    for row in ds:
        if not row.get("answerable", True):
            continue
        passages: list[Passage] = []
        for p in row["paragraphs"]:
            title = p.get("title") or ""
            text = (p.get("paragraph_text") or "").strip()
            is_gold = bool(p.get("is_supporting", False))
            passages.append(Passage(title=title, text=text, is_gold=is_gold))
        supporting_titles = sorted({p.title for p in passages if p.is_gold})
        ex = Example(
            qid=row["id"],
            question=row["question"].strip(),
            answer=str(row["answer"]).strip(),
            answer_aliases=list(row.get("answer_aliases") or []),
            passages=passages,
            supporting_titles=supporting_titles,
            dataset="musique",
            qtype=row["id"].split("__")[0] if "__" in row["id"] else "",
        )
        out.append(ex)
        if n is not None and len(out) >= n:
            break
    return out


if __name__ == "__main__":
    for e in load_musique(3):
        print(e.qid, e.qtype, "|", e.question[:70], "|", e.answer)
        print("  passages:", len(e.passages), "gold:", sum(p.is_gold for p in e.passages))
