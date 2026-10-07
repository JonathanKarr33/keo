#!/usr/bin/env python3
"""Critical recall against adjudicated severity labels (non-lexicon GT).

Loads a filled severity_batch*.json, runs the same selection policies as
``risk_retrieval.compare_policies`` on the source KG, and scores
high-severity recall where "high" is defined by ``human_class`` on the
adjudicated sample — not by the lexicon prior used inside the risk objective.

Also reports:
  - lexicon vs adjudicated agreement (accuracy, Cohen's κ, confusion)
  - lexicon-prior critical recall on the same 320 for the circular baseline
  - inverse-probability-weighted corpus-level recall estimate using sampling_weight

Refuses to run if any human_class is still null.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from risk_retrieval import (
    compare_policies,
    edge_key,
    greedy_budgeted_coverage,
    mst_baseline,
    top_weight_baseline,
)
from severity_lexicon import HIGH_SEVERITY_THRESHOLD, SEVERITY_CLASSES


def _require_complete(items: Sequence[dict]) -> None:
    missing = sum(1 for it in items if not it.get("human_class"))
    if missing:
        raise SystemExit(
            f"{missing}/{len(items)} items still have null human_class — "
            "run adjudicate_severity_batch.py first (or fill by hand)."
        )


def _cohen_kappa(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(labels)}
    mat = [[0] * len(labels) for _ in labels]
    for a, b in zip(y_true, y_pred):
        if a not in idx or b not in idx:
            continue
        mat[idx[a]][idx[b]] += 1
    po = sum(mat[i][i] for i in range(len(labels))) / n
    row = [sum(mat[i][j] for j in range(len(labels))) for i in range(len(labels))]
    col = [sum(mat[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(len(labels)))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else float("nan")
    return (po - pe) / (1.0 - pe)


def agreement_report(items: Sequence[dict]) -> dict:
    labels = list(SEVERITY_CLASSES)
    prior = [it["prior_class"] for it in items]
    human = [it["human_class"] for it in items]
    agree = sum(1 for a, b in zip(prior, human) if a == b)
    conf = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(prior, human):
        if a in conf and b in conf[a]:
            conf[a][b] += 1
    # safety binary: none vs anything else
    prior_safe = [p != "none" for p in prior]
    human_safe = [h != "none" for h in human]
    tp = sum(1 for p, h in zip(prior_safe, human_safe) if p and h)
    fp = sum(1 for p, h in zip(prior_safe, human_safe) if p and not h)
    fn = sum(1 for p, h in zip(prior_safe, human_safe) if (not p) and h)
    tn = sum(1 for p, h in zip(prior_safe, human_safe) if (not p) and not h)
    return {
        "n": len(items),
        "accuracy_5way": agree / len(items),
        "cohen_kappa_5way": _cohen_kappa(human, prior, labels),
        "prior_counts": dict(Counter(prior)),
        "human_counts": dict(Counter(human)),
        "confusion_prior_rows_human_cols": conf,
        "safety_binary": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "lexicon_false_negatives": fn,
            "lexicon_false_positives": fp,
        },
    }


def _gt_keys(
    items: Sequence[dict],
    *,
    use_human: bool,
    safety_classes: Set[str],
) -> Tuple[Set[Tuple[str, str, str]], Dict[Tuple[str, str, str], float]]:
    """Return (high-severity key set, key→sampling_weight)."""
    high: Set[Tuple[str, str, str]] = set()
    weights: Dict[Tuple[str, str, str], float] = {}
    for it in items:
        cls = it["human_class"] if use_human else it["prior_class"]
        key = edge_key(str(it["head"]), str(it["relation"]), str(it["tail"]))
        w = float(it.get("sampling_weight") or 1.0)
        weights[key] = w
        if cls in safety_classes:
            high.add(key)
    return high, weights


def _selection_keys(selection: dict) -> Set[Tuple[str, str, str]]:
    return {
        edge_key(r["head"], r["relation"], r["tail"])
        for r in selection["selected"]
    }


def recall_on_gt(
    selection: dict,
    gt_high: Set[Tuple[str, str, str]],
    weights: Optional[Dict[Tuple[str, str, str], float]] = None,
) -> dict:
    chosen = _selection_keys(selection)
    recalled = gt_high & chosen
    raw = len(recalled) / len(gt_high) if gt_high else None
    out = {
        "high_severity_total": len(gt_high),
        "high_severity_retrieved": len(recalled),
        "high_severity_recall": raw,
        "critical_omission_rate": (1.0 - raw) if raw is not None else None,
        "edges_selected": len(selection["selected"]),
        "tokens_used": selection["tokens_used"],
    }
    if weights is not None and gt_high:
        w_tot = sum(weights.get(k, 1.0) for k in gt_high)
        w_hit = sum(weights.get(k, 1.0) for k in recalled)
        out["high_severity_recall_ipw"] = w_hit / w_tot if w_tot else None
        out["ipw_weight_total"] = w_tot
        out["ipw_weight_retrieved"] = w_hit
    return out


def score_policies(
    graph: nx.Graph,
    items: Sequence[dict],
    budgets: Sequence[int],
    risk_floor: float = 0.25,
) -> dict:
    # Import local edge records builder via compare_policies side — rebuild once.
    from risk_retrieval import _edge_records

    records = _edge_records(graph)
    safety_classes = {c for c, m in SEVERITY_CLASSES.items() if m >= HIGH_SEVERITY_THRESHOLD}
    # "none" has 1.0; minor+ are safety-relevant under the same threshold as risk_retrieval.
    human_high, human_w = _gt_keys(items, use_human=True, safety_classes=safety_classes)
    prior_high, prior_w = _gt_keys(items, use_human=False, safety_classes=safety_classes)

    budget_rows = []
    for B in budgets:
        policies = {
            "top_weight": top_weight_baseline(records, B),
            "mst": mst_baseline(graph, B),
            "budgeted_coverage": greedy_budgeted_coverage(records, B, use_severity=False),
            "risk_budgeted_coverage": greedy_budgeted_coverage(records, B, use_severity=True),
            f"risk_floor_{risk_floor:g}": greedy_budgeted_coverage(
                records, B, use_severity=True,
                risk_floor=risk_floor,
                high_severity_threshold=HIGH_SEVERITY_THRESHOLD,
            ),
        }
        row = {
            "budget_tokens": B,
            "adjudicated": {
                name: recall_on_gt(sel, human_high, human_w) for name, sel in policies.items()
            },
            "lexicon_prior_on_sample": {
                name: recall_on_gt(sel, prior_high, prior_w) for name, sel in policies.items()
            },
            # Full-graph lexicon recall (the circular number already in the paper).
            "lexicon_full_graph": compare_policies(graph, B)["policies"],
        }
        budget_rows.append(row)
    return {
        "n_adjudicated": len(items),
        "n_human_high": len(human_high),
        "n_prior_high": len(prior_high),
        "safety_classes": sorted(safety_classes),
        "budgets": budget_rows,
    }


def hazard_mention_vs_adjudicated(
    ref_answers: str,
    arm_answers: str,
    items: Sequence[dict],
    *,
    ref_field: str = "keo_answer",
    arm_field: str = "risk_answer",
) -> dict:
    """
    Downstream check: do answers mention head/tail strings from adjudicated
    safety-relevant triples more often under the risk arm than under MST?

    Uses adjudicated labels (not lexicon) to define the phrase set — the
    "hazard-mention rubrics" half of #1 when experts are unavailable.
    """
    from hazard_mention_probe import load_arm

    safety_classes = {c for c, m in SEVERITY_CLASSES.items() if m >= HIGH_SEVERITY_THRESHOLD}
    phrases: List[str] = []
    for it in items:
        if it["human_class"] not in safety_classes:
            continue
        for part in (it["head"], it["tail"]):
            s = str(part).strip()
            if len(s) >= 4:
                phrases.append(s.lower())
    phrases = sorted(set(phrases))

    ref = load_arm(ref_answers, ref_field)
    arm = load_arm(arm_answers, arm_field)
    common = sorted(set(ref) & set(arm))
    rows = []
    for qid in common:
        a, b = ref[qid].lower(), arm[qid].lower()
        na = sum(1 for p in phrases if p in a)
        nb = sum(1 for p in phrases if p in b)
        rows.append({"question_id": qid, "n_ref": na, "n_arm": nb, "delta": nb - na})
    n = len(rows)
    if n == 0:
        return {"n": 0, "error": "no common answers"}
    return {
        "n": n,
        "n_adjudicated_phrases": len(phrases),
        "mean_delta_mentions": sum(r["delta"] for r in rows) / n,
        "frac_arm_more": sum(1 for r in rows if r["delta"] > 0) / n,
        "frac_ref_more": sum(1 for r in rows if r["delta"] < 0) / n,
        "reading": (
            "Mentions of adjudicated safety-triple head/tail strings in answers. "
            "Phrase set comes from human_class (or llm_proxy), not the lexicon prior."
        ),
        "per_question": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch",
        default="human_validation/ratings/severity_batch001.json",
    )
    p.add_argument(
        "--kg",
        default="../kg/output/kg_llm/gs100_total500/gemma3_4b_it/kg_gemma3_4b_it.gml",
    )
    p.add_argument("--budget", type=int, action="append", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--ref-answers", default=None, help="Optional MST answers for hazard mention")
    p.add_argument("--arm-answers", default=None, help="Optional risk answers for hazard mention")
    p.add_argument("--arm-field", default="risk_answer")
    args = p.parse_args()

    batch = json.load(open(args.batch))
    items = batch["items"]
    _require_complete(items)

    budgets = args.budget or [300, 600, 1200, 2400]
    graph = nx.read_gml(args.kg)
    agree = agreement_report(items)
    scored = score_policies(graph, items, budgets)

    report = {
        "batch": args.batch,
        "kg": args.kg,
        "adjudication": batch.get("adjudication"),
        "agreement_lexicon_vs_adjudicated": agree,
        "critical_recall": scored,
        "reading": (
            "adjudicated.high_severity_recall is the non-circular number: policies "
            "still *select* using the lexicon prior, but recall is scored against "
            "independent labels. lexicon_prior_on_sample is the circular baseline "
            "on the same 320. lexicon_full_graph is the paper's existing full-KG figure."
        ),
    }

    if args.ref_answers and args.arm_answers:
        report["hazard_mention_adjudicated"] = hazard_mention_vs_adjudicated(
            args.ref_answers,
            args.arm_answers,
            items,
            arm_field=args.arm_field,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary at 1200 tokens
    target = next((b for b in scored["budgets"] if b["budget_tokens"] == 1200), scored["budgets"][-1])
    print(f"agreement 5-way acc={agree['accuracy_5way']:.3f}  κ={agree['cohen_kappa_5way']:.3f}")
    sb = agree["safety_binary"]
    print(
        f"safety binary  P={sb['precision']}  R={sb['recall']}  "
        f"FN(lexicon miss)={sb['lexicon_false_negatives']}  FP={sb['lexicon_false_positives']}"
    )
    print(f"at budget={target['budget_tokens']}  human_high={scored['n_human_high']}")
    for name, m in target["adjudicated"].items():
        circ = target["lexicon_prior_on_sample"][name]["high_severity_recall"]
        print(
            f"  {name:<28} adj={m['high_severity_recall']:.1%}  "
            f"ipw={m.get('high_severity_recall_ipw')}  "
            f"circ_sample={circ:.1%}"
        )
    if "hazard_mention_adjudicated" in report:
        h = report["hazard_mention_adjudicated"]
        print(
            f"hazard-mention (adj phrases): n={h.get('n')}  "
            f"Δ={h.get('mean_delta_mentions'):+.3f}  arm_more={h.get('frac_arm_more'):.1%}"
        )
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
