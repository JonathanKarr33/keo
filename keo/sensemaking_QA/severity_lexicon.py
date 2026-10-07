#!/usr/bin/env python3
"""
Graded severity lexicon for aviation maintenance facts.

Single source of truth for "how safety-critical is this fact", used by
`risk_retrieval.py` (risk-weighted budgeted coverage, long-tail safety recall) and by
the `rarity` edge-weight ablation in `answer_generator.py`.

Tiers follow the failure-condition classes used in airworthiness practice
(FAA AC 25.1309-1 / SAE ARP4761): Catastrophic, Hazardous, Major, Minor, and No
Safety Effect. Encoding the tier rather than a bare number keeps the mapping
auditable: a reviewer can dispute which tier a term belongs to, which is a
reviewable claim, instead of disputing an unexplained multiplier.

Every multiplier for a term that already existed in `answer_generator.SEVERITY_TERMS`
is preserved exactly, so the existing `--edge-weight rarity` behaviour is unchanged.

These are *priors for retrieval*, not adjudicated labels. `export_for_adjudication()`
emits the candidate set that human annotators rate; the paper should report expert
agreement against these priors rather than treating them as ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

# Failure-condition classes → retrieval multiplier.
# The scale is deliberately compressed relative to the 1e-9/1e-7 probability targets
# in AC 25.1309; it orders facts, it does not estimate risk.
SEVERITY_CLASSES: Dict[str, float] = {
    "catastrophic": 3.0,   # loss of aircraft / multiple fatalities plausible
    "hazardous": 2.5,      # large reduction in safety margin, serious injury plausible
    "major": 2.0,          # significant reduction in margins, crew workload
    "minor": 1.5,          # slight reduction in margins, routine crew response
    "none": 1.0,           # no safety effect
}

# term → class. Substring match against "head relation tail", lowercased.
# Terms marked (legacy) came from the original lexicon; their effective multiplier is
# pinned below so the rarity ablation does not shift.
_TERM_CLASSES: Dict[str, str] = {
    # Catastrophic
    "fire": "catastrophic",            # legacy 3.0
    "flame": "catastrophic",           # legacy 3.0
    "explosion": "catastrophic",       # legacy 3.0
    "crash": "catastrophic",           # legacy 3.0
    "in-flight breakup": "catastrophic",
    "structural failure": "catastrophic",
    "spar crack": "catastrophic",
    "control surface separation": "catastrophic",
    # Hazardous
    "engine quit": "hazardous",        # legacy 2.5
    "engine failure": "hazardous",     # legacy 2.5
    "fuel starvation": "hazardous",    # legacy 2.5
    "uncontained": "hazardous",        # legacy 2.5
    "loss of control": "hazardous",    # legacy 2.5
    "power loss": "hazardous",
    "engine shutdown": "hazardous",
    "fuel contamination": "hazardous",
    "water in fuel": "hazardous",
    "propeller strike": "hazardous",
    "rotor": "hazardous",
    "jammed control": "hazardous",
    "runaway trim": "hazardous",
    "gear collapse": "hazardous",
    "brake failure": "hazardous",
    "depressurization": "hazardous",
    "carbon monoxide": "hazardous",
    # Major
    "smoke": "major",                  # legacy 2.0
    "fumes": "major",
    "icing": "major",                  # legacy pinned to 1.6 below
    "overheat": "major",
    "overtemp": "major",
    "cracked cylinder": "major",
    "metal contamination": "major",
    "chip detector": "major",
    "magneto": "major",
    "vibration": "major",
    "fuel leak": "major",
    "oil leak": "major",
    "birdstrike": "major",
    "lightning strike": "major",
    "hard landing": "major",
    "flap asymmetry": "major",
    # Minor
    "hydraulic": "minor",              # legacy 1.5
    "frozen": "minor",                 # legacy 1.5
    "electrical": "minor",             # legacy pinned to 1.4 below
    "corrosion": "minor",
    "chafing": "minor",
    "worn": "minor",
    "loose": "minor",
    "gasket": "minor",
    "seal": "minor",
    "placard": "minor",
    "light inoperative": "minor",
}

# Legacy multipliers that do not sit exactly on a class value. Pinned so that
# `--edge-weight rarity` keeps producing the weights the earlier runs used.
_PINNED_MULTIPLIERS: Dict[str, float] = {
    "icing": 1.6,
    "electrical": 1.4,
}

HIGH_SEVERITY_THRESHOLD = 1.5  # "minor" and above counts as safety-relevant


def term_multiplier(term: str) -> float:
    """Multiplier for one lexicon term."""
    if term in _PINNED_MULTIPLIERS:
        return _PINNED_MULTIPLIERS[term]
    return SEVERITY_CLASSES[_TERM_CLASSES[term]]


# Flat term → multiplier view, for callers that want the original shape
SEVERITY_TERMS: Dict[str, float] = {t: term_multiplier(t) for t in _TERM_CLASSES}


def severity_of_text(text: str) -> float:
    """Highest severity multiplier implied by any term in the text (1.0 if none)."""
    lowered = str(text).lower()
    score = 1.0
    for term, multiplier in SEVERITY_TERMS.items():
        if term in lowered:
            score = max(score, multiplier)
    return score


def severity_class_of_text(text: str) -> str:
    """Highest-severity class named in the text, or "none"."""
    lowered = str(text).lower()
    best, best_value = "none", 1.0
    for term, cls in _TERM_CLASSES.items():
        if term in lowered:
            value = term_multiplier(term)
            if value > best_value:
                best, best_value = cls, value
    return best


def matched_terms(text: str) -> List[str]:
    """Which lexicon terms fired — needed so annotators can audit a label."""
    lowered = str(text).lower()
    return sorted(t for t in _TERM_CLASSES if t in lowered)


def edge_severity(head: str, relation: str, tail: str) -> float:
    """Severity multiplier for a triple."""
    return severity_of_text(f"{head} {relation} {tail}")


def edge_severity_class(head: str, relation: str, tail: str) -> str:
    return severity_class_of_text(f"{head} {relation} {tail}")


def coverage_report(texts: Iterable[str]) -> Dict:
    """How much of a corpus/graph the lexicon actually touches."""
    counts: Dict[str, int] = {cls: 0 for cls in SEVERITY_CLASSES}
    total = 0
    for text in texts:
        counts[severity_class_of_text(text)] += 1
        total += 1
    return {
        "total": total,
        "by_class": counts,
        "share_safety_relevant": (
            sum(v for k, v in counts.items() if k != "none") / total if total else 0.0
        ),
        "n_terms": len(_TERM_CLASSES),
        "n_classes": len(SEVERITY_CLASSES),
    }


def stratified_sample(
    items: List[Dict],
    n_target: int,
    seed: int = 17,
) -> Tuple[List[Dict], Dict]:
    """
    Stratify the candidate labels by prior class before handing them to annotators.

    A uniform sample would be ~84% "none" on our KG, spending expert time on triples
    nobody disputes while leaving too few high-severity items to estimate agreement on
    the classes that matter. So: take the safety-relevant strata as fully as the budget
    allows, then fill the rest with "none" to measure the lexicon's false negatives.

    Each item carries its `sampling_weight` (stratum size / number sampled), so
    corpus-level rates must be computed as weighted estimates, not raw sample means.
    """
    rng = random.Random(seed)
    by_class: Dict[str, List[Dict]] = {}
    for item in items:
        by_class.setdefault(item["prior_class"], []).append(item)

    safety_classes = [c for c in SEVERITY_CLASSES if c != "none"]
    per_class = max(1, n_target // (len(safety_classes) + 1))

    chosen: List[Dict] = []
    allocation: Dict[str, Dict[str, int]] = {}
    for cls in safety_classes:
        group = by_class.get(cls, [])
        take = min(len(group), per_class)
        chosen.extend(rng.sample(group, take) if take < len(group) else list(group))
        allocation[cls] = {"stratum_size": len(group), "sampled": take}

    # "none" fills whatever the safety strata could not supply
    none_group = by_class.get("none", [])
    take_none = min(len(none_group), max(0, n_target - len(chosen)))
    chosen.extend(rng.sample(none_group, take_none) if take_none < len(none_group) else list(none_group))
    allocation["none"] = {"stratum_size": len(none_group), "sampled": take_none}

    for item in chosen:
        stratum = allocation[item["prior_class"]]
        item["sampling_stratum"] = item["prior_class"]
        item["sampling_weight"] = (
            stratum["stratum_size"] / stratum["sampled"] if stratum["sampled"] else None
        )

    rng.shuffle(chosen)  # so annotators cannot infer the prior from item order
    return chosen, allocation


def export_for_adjudication(
    triples: Iterable[Tuple[str, str, str]],
    out_path: str,
    include_unmatched: bool = True,
    n_sample: Optional[int] = None,
    seed: int = 17,
) -> Dict:
    """
    Write the candidate severity labels for human adjudication.

    Unmatched triples are included by default: the lexicon's false negatives are the
    interesting part, and only annotators can find them.
    """
    items = []
    for head, relation, tail in triples:
        text = f"{head} {relation} {tail}"
        terms = matched_terms(text)
        if not terms and not include_unmatched:
            continue
        items.append({
            "head": head,
            "relation": relation,
            "tail": tail,
            "prior_class": severity_class_of_text(text),
            "prior_multiplier": severity_of_text(text),
            "matched_terms": terms,
            "human_class": None,      # annotator fills: catastrophic|hazardous|major|minor|none
            "annotator_id": None,
            "notes": None,
        })

    total_candidates = len(items)
    allocation = None
    if n_sample is not None and n_sample < total_candidates:
        items, allocation = stratified_sample(items, n_sample, seed=seed)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "instructions": (
            "Assign human_class using the airworthiness failure-condition classes "
            "(catastrophic, hazardous, major, minor, none) for the worst plausible "
            "outcome of the fact in this triple. Ignore prior_class when it looks wrong; "
            "disagreement with the lexicon is the measurement we want."
        ),
        "classes": list(SEVERITY_CLASSES),
        "n_items": len(items),
        "n_candidates_total": total_candidates,
        "sampling": (
            {
                "scheme": "stratified by prior_class, over-sampling safety classes",
                "seed": seed,
                "allocation": allocation,
                "note": (
                    "Weight by sampling_weight for corpus-level rates; raw sample means "
                    "over-represent the safety classes by construction."
                ),
            }
            if allocation else {"scheme": "census (all candidates)"}
        ),
        "items": items,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return {"n_items": len(items), "path": out_path, "allocation": allocation}


def main():
    parser = argparse.ArgumentParser(description="Severity lexicon coverage / label export")
    parser.add_argument("--kg", default="../kg/output/knowledge_graph.gml")
    parser.add_argument(
        "--export", default=None,
        help="Write candidate labels for human adjudication to this path",
    )
    parser.add_argument(
        "--only-matched", action="store_true",
        help="Export only triples the lexicon already flags (default includes unmatched)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Stratified sample size for annotators (e.g. 300); default exports all",
    )
    parser.add_argument("--seed", type=int, default=17, help="Sampling seed")
    args = parser.parse_args()

    import networkx as nx  # local import so the lexicon itself stays dependency-free

    graph = nx.read_gml(args.kg)
    triples = []
    edges = graph.edges(keys=True, data=True) if graph.is_multigraph() else graph.edges(data=True)
    for edge in edges:
        data = edge[-1]
        relation = str(
            data.get("relation") or data.get("label") or data.get("type") or "related_to"
        )
        triples.append((str(edge[0]), relation, str(edge[1])))

    report = coverage_report(f"{h} {r} {t}" for h, r, t in triples)
    print(f"KG: {args.kg}")
    print(f"Lexicon: {report['n_terms']} terms across {report['n_classes']} classes")
    print(f"Triples: {report['total']}")
    for cls in SEVERITY_CLASSES:
        n = report["by_class"][cls]
        print(f"  {cls:<14}{n:>6}  ({n / max(report['total'], 1):.1%})")
    print(f"Safety-relevant share: {report['share_safety_relevant']:.1%}")

    if args.export:
        result = export_for_adjudication(
            triples, args.export,
            include_unmatched=not args.only_matched,
            n_sample=args.sample,
            seed=args.seed,
        )
        print(f"\n✓ {result['n_items']} candidate labels → {result['path']}")
        if result["allocation"]:
            print("  Stratified allocation (sampled / stratum size):")
            for cls, alloc in result["allocation"].items():
                print(f"    {cls:<14}{alloc['sampled']:>5} / {alloc['stratum_size']}")
            print("  Weight by sampling_weight for corpus-level rates.")
        print("  Annotators fill human_class; report agreement against prior_class.")


if __name__ == "__main__":
    main()
