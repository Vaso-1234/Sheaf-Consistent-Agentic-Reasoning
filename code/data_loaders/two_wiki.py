"""2WikiMultihopQA loader via cmriat/2wikimultihopqa mirror."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from typing import Optional

from datasets import load_dataset

from .schema import Example, Passage


def load_two_wiki(n: Optional[int] = None, split: str = "validation"):
    ds = load_dataset("cmriat/2wikimultihopqa", split=split)
    out: list[Example] = []
    for row in ds:
        meta = row.get("metadata") or {}
        sf = meta.get("supporting_facts") or {}
        supporting_titles = set(sf.get("title") or [])
        ctx = meta.get("context") or {}
        titles = ctx.get("title") or []
        sents_lists = ctx.get("content") or ctx.get("sentences") or []
        passages: list[Passage] = []
        for t, sents in zip(titles, sents_lists):
            if isinstance(sents, list):
                text = " ".join(str(s).strip() for s in sents).strip()
            else:
                text = str(sents)
            passages.append(Passage(title=t, text=text, is_gold=(t in supporting_titles)))
        gold_list = row.get("golden_answers") or []
        answer = gold_list[0] if gold_list else ""
        ex = Example(
            qid=row["id"],
            question=row["question"].strip(),
            answer=answer.strip(),
            answer_aliases=[a for a in gold_list[1:]],
            passages=passages,
            supporting_titles=sorted(supporting_titles),
            dataset="2wikimultihop",
            qtype=(meta.get("type") or ""),
        )
        out.append(ex)
        if n is not None and len(out) >= n:
            break
    return out


if __name__ == "__main__":
    for e in load_two_wiki(3):
        print(e.qid, e.qtype, "|", e.question[:70], "|", e.answer)
        print("  passages:", len(e.passages), "gold:", sum(p.is_gold for p in e.passages))
