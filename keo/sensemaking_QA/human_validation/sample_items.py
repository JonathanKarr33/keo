#!/usr/bin/env python3
"""Sample questions + KG triples for human validation."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def sample_questions(questions: List[Dict], n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for q in questions:
        key = q.get("type") or q.get("category") or "other"
        by_cat[str(key)].append(q)
    # proportional then fill
    picked: List[Dict] = []
    cats = list(by_cat.keys())
    if not cats:
        return []
    per = max(1, n // len(cats))
    for c in cats:
        pool = by_cat[c][:]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    rng.shuffle(picked)
    if len(picked) < n:
        rest = [q for q in questions if q not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def sample_triples(kg_path: str, n: int, seed: int) -> List[Dict[str, str]]:
    import networkx as nx

    rng = random.Random(seed)
    g = nx.read_gml(kg_path)
    edges = []
    for u, v, data in g.edges(data=True):
        rel = (
            data.get("relation") or data.get("relationship") or data.get("relations")
            or data.get("label") or data.get("type") or "related_to"
        )
        edges.append({"head": str(u), "relation": str(rel), "tail": str(v)})
    rng.shuffle(edges)
    return edges[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--kg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-questions", type=int, default=40)
    p.add_argument("--n-triples", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-id", default="batch001")
    args = p.parse_args()

    qs = sample_questions(load_questions(args.questions), args.n_questions, args.seed)
    triples = sample_triples(args.kg, args.n_triples, args.seed)

    payload = {
        "batch_id": args.batch_id,
        "annotator": "",
        "questions": [
            {
                "question_id": q.get("id", ""),
                "question": q.get("question", ""),
                "clarity": None,
                "difficulty": None,
                "domain_relevance": None,
                "answerability_without_leak": None,
                "notes": "",
            }
            for q in qs
        ],
        "triples": [
            {**t, "plausible": None, "relation_fit": None, "notes": ""} for t in triples
        ],
        "answer_preferences": [],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(qs)} questions and {len(triples)} triples → {args.out}")


if __name__ == "__main__":
    main()
