#!/usr/bin/env python3
"""
Risk-sensitive budgeted retrieval (papers/ACL_STRATEGY.md §3, Pivot C).

Retrieval for safety-critical QA should not maximise similarity. It should maximise
risk-weighted coverage of the corpus subject to the context budget the answer model
actually has. Formally, for a candidate edge set \\(E\\) and token budget \\(B\\):

    maximise   f(S) = Σ_{c ∈ cover(S)} max_{e ∈ S, c ∈ cover(e)} rel(e,q) · sev(e)
    subject to Σ_{e ∈ S} tokens(e) ≤ B

`cover(e)` is the set of entities an edge touches, so each entity is credited once at
its best-scoring edge. f is monotone submodular and the constraint is a knapsack, so
the cost-benefit greedy rule (take the edge with the best marginal gain per token)
carries the standard constant-factor guarantee for budgeted submodular maximisation
(Khuller, Moss & Naor 1999; Sviridenko 2004).

This reframes the MST heuristic instead of replacing it: a maximum spanning tree is
the special case that maximises edge weight per unit of structure while keeping the
component connected, with sev ≡ 1 and coverage credited by node. Stating the
objective explains why MST helped and predicts where it should fail — namely when
high-severity facts are rare, because frequency weighting buries them.

Severity comes from `severity_lexicon.py`, which is graded and auditable rather than
a flat keyword list.

Usage:
  python risk_retrieval.py --kg ../kg/output/knowledge_graph.gml          # geometry report
  python risk_retrieval.py --kg ../kg/output/knowledge_graph.gml --budget 1200
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from severity_lexicon import edge_severity

# Serialised edges look like "head — relation → tail"; ~4 chars per token
CHARS_PER_TOKEN = 4.0


def edge_tokens(head: str, relation: str, tail: str) -> int:
    """Token cost of serialising one triple into the prompt."""
    text = f"{head} {relation} {tail}"
    return max(1, int(math.ceil(len(text) / CHARS_PER_TOKEN)))


def edge_relation(data: Dict) -> str:
    """
    Resolve a relation label from the several attribute names these KGs use.

    Shared deliberately: mst_baseline previously resolved relations from a *shorter* list
    that omitted ``relationship``, so those edges came out labelled "related_to" while
    _edge_records gave them their real label. The two then failed to key-match in
    safety_metrics and MST's recall was scored as zero for every such edge — an error that
    only ever flattered the policy being compared against it.
    """
    return str(
        data.get("relation") or data.get("relationship") or data.get("relations")
        or data.get("label") or data.get("type") or "related_to"
    )


def edge_key(head: str, relation: str, tail: str) -> Tuple[str, str, str]:
    """
    Orientation-insensitive identity for a triple.

    mst_baseline routes through ``to_undirected()``, which may return an edge with its
    endpoints swapped relative to the source graph. Recall asks whether a fact reached the
    context, so endpoint order is not part of the question — and keying on the ordered
    tuple silently undercounted MST.
    """
    first, second = sorted((head, tail))
    return (first, relation, second)


def _edge_records(graph: nx.Graph) -> List[Dict]:
    """Flatten a graph into scoring records, tolerating multi/simple and di/undirected."""
    records = []
    edges = graph.edges(keys=True, data=True) if graph.is_multigraph() else graph.edges(data=True)
    for edge in edges:
        head, tail, data = edge[0], edge[1], edge[-1]
        relation = edge_relation(data)
        try:
            weight = float(data.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        records.append({
            "head": str(head),
            "relation": relation,
            "tail": str(tail),
            "weight": weight,
            "severity": edge_severity(str(head), relation, str(tail)),
            "tokens": edge_tokens(str(head), relation, str(tail)),
        })
    return records


def greedy_budgeted_coverage(
    records: Sequence[Dict],
    budget_tokens: int,
    relevance: Optional[Callable[[Dict], float]] = None,
    use_severity: bool = True,
    risk_floor: float = 0.0,
    high_severity_threshold: float = 1.5,
    fill_budget: bool = False,
) -> Dict:
    """
    Cost-benefit greedy for budgeted submodular coverage.

    Each entity is credited once, at the best-scoring edge that covers it, so adding a
    second edge about an already-covered entity yields little marginal gain. That is
    what makes the objective submodular and what makes the greedy rule sensible: it
    spreads the budget across distinct entities instead of piling onto one hub.

    `risk_floor` reserves that fraction of the budget for high-severity edges before
    general selection begins. It exists because the severity multiplier alone is
    fragile: a fact with severity s and relevance r loses to a mundane fact whenever
    s·r < r', so a rare critical fact is still dropped when the mundane backbone is
    frequent enough. The reserved pass converts a soft preference into a floor —
    critical recall is at least min(1, reserved tokens / tokens needed for them) —
    at the cost of some coverage. See test_risk_retrieval_offline.py for both cases.

    `fill_budget` spends whatever the objective leaves over. Coverage saturates: once
    every entity sits at its best-scoring edge, further edges have zero marginal gain
    and the greedy stops, often well short of B. That is a genuine efficiency result,
    but it means the arm ships *shorter* context than MST, and judged comparisons are
    length-sensitive (judge_length_bias.py: the longer answer wins 93% of the time).
    Filling holds context length roughly constant across arms so a judged difference is
    attributable to *which* edges were chosen. Leave it off to report the objective;
    turn it on for a fair judged comparison.
    """
    if budget_tokens <= 0 or not records:
        return {
            "selected": [], "tokens_used": 0, "objective": 0.0,
            "entities_covered": 0, "reserved_tokens_used": 0, "filler_edges": 0,
        }

    def score(rec: Dict) -> float:
        rel = relevance(rec) if relevance else rec["weight"]
        sev = rec["severity"] if use_severity else 1.0
        return max(rel, 0.0) * max(sev, 0.0)

    best_per_entity: Dict[str, float] = {}
    selected: List[Dict] = []
    remaining = list(records)
    tokens_used = 0
    objective = 0.0

    # Reserved pass: spend the risk floor on high-severity edges, most severe first,
    # cheapest first within a severity level so the floor buys as many as possible.
    reserved_tokens_used = 0
    if risk_floor > 0:
        reserve = int(budget_tokens * min(risk_floor, 1.0))
        # Index-based, not value-based: records may carry graph attribute dicts, and
        # comparing those by equality can blow up on array-valued attributes.
        critical_indices = sorted(
            (i for i, r in enumerate(remaining) if r["severity"] >= high_severity_threshold),
            key=lambda i: (-remaining[i]["severity"], remaining[i]["tokens"]),
        )
        taken = set()
        for i in critical_indices:
            rec = remaining[i]
            if reserved_tokens_used + rec["tokens"] > reserve:
                continue
            value = score(rec)
            # Deduplicated: a self-loop would otherwise credit one entity's gain twice.
            endpoints = {rec["head"], rec["tail"]}
            gain = sum(
                max(0.0, value - best_per_entity.get(entity, 0.0)) for entity in endpoints
            )
            for entity in endpoints:
                best_per_entity[entity] = max(best_per_entity.get(entity, 0.0), value)
            selected.append(rec)
            taken.add(i)
            reserved_tokens_used += rec["tokens"]
            tokens_used += rec["tokens"]
            objective += gain
        remaining = [r for i, r in enumerate(remaining) if i not in taken]

    while remaining:
        best_index, best_ratio, best_gain = -1, 0.0, 0.0
        for i, rec in enumerate(remaining):
            if tokens_used + rec["tokens"] > budget_tokens:
                continue
            value = score(rec)
            gain = 0.0
            for entity in {rec["head"], rec["tail"]}:
                gain += max(0.0, value - best_per_entity.get(entity, 0.0))
            ratio = gain / rec["tokens"]
            if ratio > best_ratio:
                best_index, best_ratio, best_gain = i, ratio, gain

        if best_index < 0:
            break  # nothing affordable adds value

        chosen = remaining.pop(best_index)
        value = score(chosen)
        for entity in {chosen["head"], chosen["tail"]}:
            best_per_entity[entity] = max(best_per_entity.get(entity, 0.0), value)
        selected.append(chosen)
        tokens_used += chosen["tokens"]
        objective += best_gain

    # Optional fill pass: the objective is already maximised, so these edges add no
    # coverage value. They exist only to equalise context length across arms.
    filler_edges = 0
    if fill_budget:
        for index in sorted(range(len(remaining)), key=lambda i: -score(remaining[i])):
            rec = remaining[index]
            if tokens_used + rec["tokens"] > budget_tokens:
                continue
            selected.append(rec)
            tokens_used += rec["tokens"]
            filler_edges += 1

    return {
        "selected": selected,
        "tokens_used": tokens_used,
        "objective": objective,
        "entities_covered": len(best_per_entity),
        "reserved_tokens_used": reserved_tokens_used,
        "filler_edges": filler_edges,
    }


def top_weight_baseline(records: Sequence[Dict], budget_tokens: int) -> Dict:
    """Frequency-ranked selection under the same budget — today's default policy."""
    chosen, tokens = [], 0
    for rec in sorted(records, key=lambda r: -r["weight"]):
        if tokens + rec["tokens"] > budget_tokens:
            continue
        chosen.append(rec)
        tokens += rec["tokens"]
    entities = {e for r in chosen for e in (r["head"], r["tail"])}
    return {
        "selected": chosen,
        "tokens_used": tokens,
        "objective": sum(r["weight"] for r in chosen),
        "entities_covered": len(entities),
    }


def mst_baseline(graph: nx.Graph, budget_tokens: int) -> Dict:
    """
    Maximum spanning tree per component, serialised until the budget runs out.

    This is the current KEO selector, expressed in the same units so the three
    policies can be compared on one axis.
    """
    undirected = graph.to_undirected() if graph.is_directed() else graph
    simple = nx.Graph()
    for head, tail, data in undirected.edges(data=True):
        try:
            weight = float(data.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if simple.has_edge(head, tail):
            simple[head][tail]["weight"] += weight
        else:
            simple.add_edge(head, tail, weight=weight, **{
                k: v for k, v in data.items() if k != "weight"
            })

    tree_edges = []
    for component in nx.connected_components(simple):
        sub = simple.subgraph(component)
        if sub.number_of_edges() == 0:
            continue
        for head, tail, data in nx.maximum_spanning_tree(sub, weight="weight").edges(data=True):
            relation = edge_relation(data)
            tree_edges.append({
                "head": str(head),
                "relation": relation,
                "tail": str(tail),
                "weight": float(data.get("weight", 1.0)),
                "severity": edge_severity(str(head), relation, str(tail)),
                "tokens": edge_tokens(str(head), relation, str(tail)),
            })

    chosen, tokens = [], 0
    for rec in sorted(tree_edges, key=lambda r: -r["weight"]):
        if tokens + rec["tokens"] > budget_tokens:
            continue
        chosen.append(rec)
        tokens += rec["tokens"]
    entities = {e for r in chosen for e in (r["head"], r["tail"])}
    return {
        "selected": chosen,
        "tokens_used": tokens,
        "objective": sum(r["weight"] for r in chosen),
        "entities_covered": len(entities),
    }


def safety_metrics(
    selection: Dict,
    all_records: Sequence[Dict],
    high_severity_threshold: float = 1.5,
) -> Dict:
    """
    Long-tail safety recall for one selection.

    The metric that matters operationally is not mean judged quality but whether the
    safety-critical facts made it into the context at all. `critical_omission_rate`
    is the complement: the share of high-severity facts the policy dropped.
    """
    high = [r for r in all_records if r["severity"] >= high_severity_threshold]
    chosen_keys = {edge_key(r["head"], r["relation"], r["tail"]) for r in selection["selected"]}
    recalled = [r for r in high if edge_key(r["head"], r["relation"], r["tail"]) in chosen_keys]
    return {
        "high_severity_total": len(high),
        "high_severity_retrieved": len(recalled),
        "high_severity_recall": len(recalled) / len(high) if high else None,
        "critical_omission_rate": 1 - len(recalled) / len(high) if high else None,
        "mean_severity_of_selection": (
            sum(r["severity"] for r in selection["selected"]) / len(selection["selected"])
            if selection["selected"] else 0.0
        ),
        "edges_selected": len(selection["selected"]),
        "tokens_used": selection["tokens_used"],
        "entities_covered": selection["entities_covered"],
    }


def rarity_of_high_severity(records: Sequence[Dict], high_severity_threshold: float = 1.5) -> Dict:
    """
    Are high-severity facts rare? This is the premise the whole pivot rests on.

    Replaces the n=5 anecdote in the draft with a count over every labelled edge.
    """
    if not records:
        return {}
    weights = sorted(r["weight"] for r in records)
    min_weight = weights[0]
    cutoff_index = int(0.8 * (len(weights) - 1))
    top_quintile_cutoff = weights[cutoff_index]
    high = [r for r in records if r["severity"] >= high_severity_threshold]
    if not high:
        return {"high_severity_total": 0}

    # The quintile cutoff is meaningless when weights are so tied that the 80th
    # percentile equals the minimum: nothing can fall "below" it. Report the share of
    # singleton-weight facts instead — those are the ones frequency ranking cannot see.
    quintile_identifiable = top_quintile_cutoff > min_weight
    below = [r for r in high if r["weight"] < top_quintile_cutoff]
    singletons = [r for r in high if r["weight"] <= min_weight]
    return {
        "high_severity_total": len(high),
        "share_of_all_edges": len(high) / len(records),
        "min_weight": min_weight,
        "top_quintile_weight_cutoff": top_quintile_cutoff,
        "quintile_cutoff_identifiable": quintile_identifiable,
        "share_below_top_quintile": (len(below) / len(high)) if quintile_identifiable else None,
        "high_severity_at_min_weight": len(singletons),
        "share_at_min_weight": len(singletons) / len(high),
        "mean_weight_high_severity": sum(r["weight"] for r in high) / len(high),
        "mean_weight_all": sum(r["weight"] for r in records) / len(records),
    }


def weight_degeneracy(records: Sequence[Dict]) -> Dict:
    """
    Are edge weights informative on this graph?

    Several KGs in this repo assign every edge weight 1.0. There, frequency ranking and
    maximum spanning tree are decided by tie-breaks rather than by evidence, so any
    comparison against them says nothing. The severity-on/severity-off contrast stays
    valid either way, because it toggles one term of the same objective.
    """
    weights = [r["weight"] for r in records]
    distinct = len(set(weights))
    mean = sum(weights) / len(weights) if weights else 0.0
    variance = sum((w - mean) ** 2 for w in weights) / len(weights) if weights else 0.0
    return {
        "distinct_weights": distinct,
        "std": math.sqrt(variance),
        "degenerate": distinct <= 1,
    }


def compare_policies(
    graph: nx.Graph,
    budget_tokens: int,
    high_severity_threshold: float = 1.5,
    risk_floor: float = 0.25,
) -> Dict:
    """Run every selection policy at one budget and score them on the safety metrics."""
    records = _edge_records(graph)
    policies = {
        "top_weight": top_weight_baseline(records, budget_tokens),
        "mst": mst_baseline(graph, budget_tokens),
        "budgeted_coverage": greedy_budgeted_coverage(records, budget_tokens, use_severity=False),
        "risk_budgeted_coverage": greedy_budgeted_coverage(records, budget_tokens, use_severity=True),
        f"risk_floor_{risk_floor:g}": greedy_budgeted_coverage(
            records, budget_tokens, use_severity=True,
            risk_floor=risk_floor, high_severity_threshold=high_severity_threshold,
        ),
    }
    return {
        "budget_tokens": budget_tokens,
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "candidate_records": len(records),
        },
        "weight_degeneracy": weight_degeneracy(records),
        "severity_prevalence": rarity_of_high_severity(records, high_severity_threshold),
        "policies": {
            name: safety_metrics(sel, records, high_severity_threshold)
            for name, sel in policies.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Risk-sensitive budgeted retrieval report")
    parser.add_argument("--kg", default="../kg/output/knowledge_graph.gml")
    parser.add_argument(
        "--budget", type=int, action="append", default=None,
        help="Token budget; repeatable (default: 300 600 1200 2400)",
    )
    parser.add_argument("--severity-threshold", type=float, default=1.5)
    parser.add_argument("--out", default=None, help="Write the report JSON here")
    args = parser.parse_args()

    budgets = args.budget or [300, 600, 1200, 2400]
    graph = nx.read_gml(args.kg)
    print(f"KG: {args.kg}  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")

    report = {"kg": args.kg, "severity_threshold": args.severity_threshold, "budgets": []}
    for budget in budgets:
        result = compare_policies(graph, budget, args.severity_threshold)
        report["budgets"].append(result)

        prevalence = result["severity_prevalence"]
        if budget == budgets[0]:
            degeneracy = result["weight_degeneracy"]
            print(
                f"High-severity edges: {prevalence['high_severity_total']}"
                f" of {result['graph']['candidate_records']}"
                f" ({prevalence['share_of_all_edges']:.1%})"
            )
            if degeneracy["degenerate"]:
                print(
                    "WARNING: every edge weight is identical on this graph. top_weight and "
                    "mst are decided by tie-breaks here, so treat only the "
                    "budgeted_coverage vs risk_budgeted_coverage contrast as meaningful.\n"
                )
            else:
                print(
                    f"  {prevalence['share_at_min_weight']:.0%} appear at the minimum"
                    f" edge weight (w={prevalence['min_weight']:.1f}), i.e. invisible to"
                    f" frequency ranking."
                )
                if prevalence["share_below_top_quintile"] is not None:
                    print(
                        f"  {prevalence['share_below_top_quintile']:.0%} fall below the"
                        f" top-quintile cutoff"
                        f" (w={prevalence['top_quintile_weight_cutoff']:.1f})."
                    )
                print(
                    f"  Mean weight: high-severity"
                    f" {prevalence['mean_weight_high_severity']:.2f}"
                    f" vs all edges {prevalence['mean_weight_all']:.2f}\n"
                )

        print(f"Budget {budget} tokens:")
        print(
            f"  {'policy':<24}{'edges':>7}{'tokens':>8}{'entities':>10}"
            f"{'crit.recall':>13}{'omission':>10}{'mean sev':>10}"
        )
        for name, m in result["policies"].items():
            recall = "--" if m["high_severity_recall"] is None else f"{m['high_severity_recall']:.1%}"
            omission = (
                "--" if m["critical_omission_rate"] is None else f"{m['critical_omission_rate']:.1%}"
            )
            print(
                f"  {name:<24}{m['edges_selected']:>7}{m['tokens_used']:>8}"
                f"{m['entities_covered']:>10}{recall:>13}{omission:>10}"
                f"{m['mean_severity_of_selection']:>10.2f}"
            )
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report → {args.out}")


if __name__ == "__main__":
    main()
