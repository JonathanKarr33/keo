#!/usr/bin/env python3
"""Downstream hazard-mention check for risk vs MST answers (lexicon-relative).

The risk Pareto shows selectors differ on lexicon critical recall. This script
asks a weaker but answer-level question: do risk-selected contexts make the
*answer* mention more severity-lexicon terms than MST?

This is NOT adjudicated ground truth — same lexicon circularity as critical
recall — but it is the offline half of the downstream check while experts label
severity_batch001.json. Refuses to invent a win from underpowered n.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

from severity_lexicon import HIGH_SEVERITY_THRESHOLD, matched_terms, severity_of_text


def _answer_text(row: dict, field: str) -> Optional[str]:
    blob = row.get(field) or {}
    if not isinstance(blob, dict):
        return None
    if blob.get("error"):
        return None
    ans = blob.get("answer")
    if not ans or str(ans).startswith("Error"):
        return None
    return str(ans)


def load_arm(path: str, field: str) -> Dict[str, str]:
    rows = json.load(open(path))
    out: Dict[str, str] = {}
    for row in rows:
        qid = row.get("question_id")
        text = _answer_text(row, field)
        if qid and text is not None:
            out[qid] = text
    return out


def summarise(ref: Dict[str, str], arm: Dict[str, str]) -> dict:
    common = sorted(set(ref) & set(arm))
    rows = []
    for qid in common:
        a, b = ref[qid], arm[qid]
        ta, tb = matched_terms(a), matched_terms(b)
        sa, sb = severity_of_text(a), severity_of_text(b)
        rows.append(
            {
                "question_id": qid,
                "n_terms_ref": len(ta),
                "n_terms_arm": len(tb),
                "delta_terms": len(tb) - len(ta),
                "sev_ref": sa,
                "sev_arm": sb,
                "delta_sev": sb - sa,
                "ref_has_safety": sa >= HIGH_SEVERITY_THRESHOLD,
                "arm_has_safety": sb >= HIGH_SEVERITY_THRESHOLD,
            }
        )
    n = len(rows)
    if n == 0:
        return {"n": 0, "error": "no common answered questions"}
    mean_d_terms = sum(r["delta_terms"] for r in rows) / n
    mean_d_sev = sum(r["delta_sev"] for r in rows) / n
    arm_only = sum(1 for r in rows if r["arm_has_safety"] and not r["ref_has_safety"])
    ref_only = sum(1 for r in rows if r["ref_has_safety"] and not r["arm_has_safety"])
    both = sum(1 for r in rows if r["arm_has_safety"] and r["ref_has_safety"])
    neither = sum(1 for r in rows if not r["arm_has_safety"] and not r["ref_has_safety"])
    return {
        "n": n,
        "mean_delta_lexicon_terms": mean_d_terms,
        "mean_delta_severity_score": mean_d_sev,
        "frac_arm_more_terms": sum(1 for r in rows if r["delta_terms"] > 0) / n,
        "frac_ref_more_terms": sum(1 for r in rows if r["delta_terms"] < 0) / n,
        "safety_mention_contingency": {
            "both": both,
            "arm_only": arm_only,
            "ref_only": ref_only,
            "neither": neither,
        },
        "reading": (
            "Lexicon-relative answer mention rates; same prior as critical recall. "
            "Positive mean_delta_lexicon_terms means the arm answer mentions more "
            "severity terms than the reference. Not adjudicated; not a Claude score."
        ),
        "per_question": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref-answers", required=True, help="MST/KEO answers.json")
    p.add_argument("--arm-answers", required=True, help="Risk (or other) answers.json")
    p.add_argument("--ref-field", default="keo_answer")
    p.add_argument("--arm-field", default="risk_answer")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ref = load_arm(args.ref_answers, args.ref_field)
    arm = load_arm(args.arm_answers, args.arm_field)
    result = summarise(ref, arm)
    result["ref_answers"] = args.ref_answers
    result["arm_answers"] = args.arm_answers
    result["ref_n"] = len(ref)
    result["arm_n"] = len(arm)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"n={result.get('n')}  Δterms={result.get('mean_delta_lexicon_terms'):+.3f}  "
        f"Δsev={result.get('mean_delta_severity_score'):+.3f}  "
        f"arm_more={result.get('frac_arm_more_terms'):.1%}  "
        f"contingency={result.get('safety_mention_contingency')}"
    )
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
