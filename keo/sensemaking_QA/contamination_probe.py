#!/usr/bin/env python3
"""Closed-book contamination probe for the OMIn / MaintNet benchmark.

ACL_STRATEGY §4.2 asks whether answer models have memorised the corpus. A full
probe needs the answerer; this script is the *corpus-side* half that can run
while the machine is busy: it measures how much of each question's content is
literally recoverable from the retrieval corpus by n-gram overlap. High overlap
means the item is a retrieval lookup, not a generalisation test — and any later
closed-book LLM probe should be read against that baseline.

Refuses to invent a "contamination rate" from the overlap alone: overlap is
reported as itself, not as a guilty verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Iterable, List, Set, Tuple

import pandas as pd

TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokens(text: str) -> List[str]:
    return TOKEN.findall((text or "").lower())


def ngrams(toks: List[str], n: int) -> Set[Tuple[str, ...]]:
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def load_corpus_text(paths: List[str], text_cols: List[str]) -> str:
    parts = []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        cols = [c for c in text_cols if c in df.columns]
        if not cols:
            # Prefer object columns with long mean string length — FAA dumps put the
            # narrative in c119 and the record id in c5; both are dtype object, so a
            # naive "first object cols" pick silently analysed identifiers.
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            scored = []
            for c in obj_cols:
                sample = df[c].dropna().astype(str).head(200)
                if sample.empty:
                    continue
                scored.append((sample.map(len).mean(), c))
            scored.sort(reverse=True)
            cols = [c for _, c in scored[:3]]
        if not cols:
            raise ValueError(f"{path}: no usable text columns (tried {text_cols})")
        for c in cols:
            parts.extend(str(x) for x in df[c].dropna().tolist())
    return "\n".join(parts)


def probe(questions: List[dict], corpus_text: str, n: int = 5) -> dict:
    corp_toks = tokens(corpus_text)
    corp_ng = ngrams(corp_toks, n)
    if not corp_ng:
        raise ValueError("corpus produced no n-grams — check --text-cols")

    rows = []
    for q in questions:
        qtoks = tokens(q.get("question") or "")
        qng = ngrams(qtoks, n)
        if not qng:
            overlap = 0.0
            hits = 0
        else:
            hits = sum(1 for g in qng if g in corp_ng)
            overlap = hits / len(qng)
        rows.append(
            {
                "question_id": q.get("id"),
                "n_grams": len(qng),
                "hits_in_corpus": hits,
                "overlap": overlap,
                "type": q.get("type"),
                "category": q.get("category"),
            }
        )

    overlaps = [r["overlap"] for r in rows]
    return {
        "n_questions": len(rows),
        "ngram_n": n,
        "corpus_ngrams": len(corp_ng),
        "mean_overlap": sum(overlaps) / len(overlaps),
        "median_overlap": sorted(overlaps)[len(overlaps) // 2],
        "frac_gt_0.5": sum(1 for o in overlaps if o > 0.5) / len(overlaps),
        "frac_gt_0.8": sum(1 for o in overlaps if o > 0.8) / len(overlaps),
        "reading": (
            "High 5-gram overlap means many questions are near-paraphrases of the "
            "retrieval corpus; a closed-book LLM probe on those items is not a clean "
            "memorisation test. Low overlap means the items ask for synthesis the "
            "corpus does not literally contain."
        ),
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--corpus", action="append", default=[], required=True)
    parser.add_argument(
        "--text-cols",
        default="c5,narrative,text,problem,action,description",
        help="Comma-separated CSV columns to treat as free text",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    questions = json.load(open(args.questions))
    corpus = load_corpus_text(args.corpus, args.text_cols.split(","))
    result = probe(questions, corpus, n=args.n)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(
        f"n={result['n_questions']}  mean {args.n}-gram overlap="
        f"{result['mean_overlap']:.3f}  "
        f">0.5={result['frac_gt_0.5']:.1%}  >0.8={result['frac_gt_0.8']:.1%}"
    )
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
