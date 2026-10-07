#!/usr/bin/env python3
"""Aggregate completed human ratings."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional


def mean(xs: List[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rated", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.rated) as f:
        data = json.load(f)

    q_scores = {"clarity": [], "difficulty": [], "domain_relevance": [], "answerability_without_leak": []}
    for q in data.get("questions", []):
        for k in q_scores:
            v = q.get(k)
            if isinstance(v, (int, float)):
                q_scores[k].append(float(v))

    plausible = Counter(t.get("plausible") for t in data.get("triples", []) if t.get("plausible"))
    prefs = Counter(a.get("winner") for a in data.get("answer_preferences", []) if a.get("winner"))

    summary: Dict[str, Any] = {
        "batch_id": data.get("batch_id"),
        "annotator": data.get("annotator"),
        "n_questions_rated": len(q_scores["clarity"]),
        "question_means": {k: mean(v) for k, v in q_scores.items()},
        "triple_plausible_counts": dict(plausible),
        "answer_preference_counts": dict(prefs),
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
