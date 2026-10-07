#!/usr/bin/env python3
"""
Offline checks for the mechanism-control arms (no API calls, no LLM).

Controls under test (papers/ACL_STRATEGY.md §4):
  keo_shuf     - _permute_graph_edges keeps entities/degrees, destroys pairings
  keo_norel    - _format_relationship_description drops relation semantics
  chunk_dedup  - _dedup_scored_chunks removes near-duplicate chunks
  chunk_tok    - _chunk_budget matches the chunk budget to KEO's context length

Run: python test_controls_offline.py
"""

from __future__ import annotations

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_generator import SensemakingAnswerGenerator  # noqa: E402
from offline_harness import run_suite  # noqa: E402


def make_generator(**controls) -> SensemakingAnswerGenerator:
    """Bare instance with only the attributes the control paths touch (no client)."""
    gen = SensemakingAnswerGenerator.__new__(SensemakingAnswerGenerator)
    gen.shuffle_edges = controls.get("shuffle_edges", False)
    gen.strip_relations = controls.get("strip_relations", False)
    gen.dedup_chunks = controls.get("dedup_chunks", False)
    gen.match_context_budget = controls.get("match_context_budget", False)
    gen.control_seed = controls.get("control_seed", 13)
    gen.knowledge_graph = controls.get("knowledge_graph")
    return gen


def build_toy_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edge("engine quit", "fuel sumps frozen", weight=21.0, relation="caused_by")
    g.add_edge("engine quit", "water in fuel", weight=8.0, relation="caused_by")
    g.add_edge("fuel sumps frozen", "winter ops", weight=5.0, relation="associated_with")
    g.add_edge("water in fuel", "winter ops", weight=3.0, relation="associated_with")
    g.add_edge("landing gear", "hydraulic leak", weight=12.0, relation="exhibits")
    g.add_edge("hydraulic leak", "gear collapse", weight=9.0, relation="caused")
    return g


def test_edge_permutation():
    g = build_toy_graph()
    gen = make_generator(shuffle_edges=True)
    permuted = gen._permute_graph_edges(g)

    assert set(permuted.nodes()) == set(g.nodes()), "node set must be preserved"
    assert permuted.number_of_edges() <= g.number_of_edges()
    assert permuted.number_of_edges() >= g.number_of_edges() - 1, "should retain nearly all edges"

    orig_relations = sorted(d.get("relation") for _, _, d in g.edges(data=True))
    new_relations = sorted(d.get("relation") for _, _, d in permuted.edges(data=True))
    assert new_relations == orig_relations, "relation label multiset must be preserved"

    orig_pairs = set(g.edges())
    new_pairs = set(permuted.edges())
    assert new_pairs != orig_pairs, "permutation must actually change some pairings"
    assert not any(u == v for u, v in new_pairs), "no self loops"

    # Reproducible for a fixed seed, different across seeds
    again = make_generator(shuffle_edges=True, control_seed=13)._permute_graph_edges(g)
    assert set(again.edges()) == new_pairs, "same seed must reproduce the permutation"
    other = make_generator(shuffle_edges=True, control_seed=99)._permute_graph_edges(g)
    assert set(other.edges()) != new_pairs, "different seed should give a different permutation"

    overlap = len(new_pairs & orig_pairs) / max(1, len(orig_pairs))
    print(f"  permutation: {len(new_pairs)} edges, {overlap:.0%} original pairs retained")


def test_relation_stripping():
    edge_data = {"relation": "caused_by"}
    keep = make_generator()
    strip = make_generator(strip_relations=True)

    assert keep._format_relationship_description("a", "b", edge_data) == "caused by"
    assert strip._format_relationship_description("a", "b", edge_data) == "is connected to"
    # Connectivity survives, semantics do not
    assert "caused" not in strip._format_relationship_description("a", "b", edge_data)
    print("  relation stripping: 'caused by' → 'is connected to'")


def test_chunk_dedup():
    boilerplate = "removed and replaced fuel pump per maintenance manual chapter 28"
    scored = [
        (0, 0.91, boilerplate),
        (1, 0.90, boilerplate + " revision b"),      # near-duplicate
        (2, 0.88, "engine quit during climb, fuel sumps found frozen"),
        (3, 0.80, boilerplate),                       # exact duplicate
        (4, 0.70, "hydraulic leak at left main landing gear actuator"),
    ]
    gen = make_generator(dedup_chunks=True)
    kept = gen._dedup_scored_chunks(scored)
    kept_ids = [i for i, _, _ in kept]

    assert 0 in kept_ids and 2 in kept_ids and 4 in kept_ids, "distinct chunks must survive"
    assert 3 not in kept_ids, "exact duplicate must be dropped"
    assert len(kept) < len(scored), "dedup must remove something here"
    assert kept_ids == sorted(kept_ids, key=lambda i: -dict((j, s) for j, s, _ in scored)[i]), \
        "similarity order must be preserved"
    print(f"  dedup: {len(scored)} → {len(kept)} chunks (kept {kept_ids})")


def test_matched_budget():
    default_chars = 4000

    # No KG loaded → fall back to the default budget, unmatched
    off = make_generator()
    assert off._chunk_budget("q", {}, default_chars) == (default_chars, 5, False)

    # KG present → budget tracks the KEO context length, chunk count scales with it
    on = make_generator(match_context_budget=True, knowledge_graph=build_toy_graph())
    on.keo_context_chars = lambda *a, **k: 6200  # stand in for the graph path
    budget, top_k, matched = on._chunk_budget("q", {}, default_chars)
    assert (budget, top_k, matched) == (6200, 31, True), (budget, top_k, matched)

    # Degenerate graph context → fall back rather than starve the chunk arm
    on.keo_context_chars = lambda *a, **k: 0
    assert on._chunk_budget("q", {}, default_chars) == (default_chars, 5, False)
    print(f"  matched budget: 6200 chars → {top_k} chunks; 0 chars → default")


def test_budget_escalation():
    """The matched arm must actually spend its budget, not just be granted one."""
    on = make_generator(match_context_budget=True, knowledge_graph=build_toy_graph())
    on.keo_context_chars = lambda *a, **k: 6000

    # Corpus that yields ~100 chars per requested chunk: 31 chunks undershoots 6000
    calls = []

    def fake_retrieve(question, datasets, max_chars, top_k=3, **kwargs):
        calls.append(top_k)
        return "x" * min(max_chars, top_k * 100)

    on._retrieve_basic_context = fake_retrieve
    context, budget, top_k = on._chunk_context_for("q", {}, 4000)
    assert budget == 6000
    assert len(context) >= 0.9 * budget, (len(context), budget)
    assert top_k > calls[0], "must escalate past the first estimate"

    # A corpus that cannot fill the budget must terminate instead of looping
    on._retrieve_basic_context = lambda *a, **k: "short"
    context, budget, top_k = on._chunk_context_for("q", {}, 4000)
    assert context == "short" and top_k <= 80
    print(f"  escalation: top_k {calls[0]}→{calls[-1]} to fill 6000 chars; exhausted corpus stops")


def test_control_config_provenance():
    gen = make_generator(shuffle_edges=True, control_seed=7)
    cfg = gen.control_config()
    assert cfg["shuffle_edges"] is True and cfg["control_seed"] == 7
    assert cfg["strip_relations"] is False
    # Seed is only meaningful when the permutation is active
    assert make_generator().control_config()["control_seed"] is None
    print(f"  provenance: {cfg}")


def test_real_kg_permutation():
    """Sanity-check the control on the real OMIn KG when it is present."""
    kg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../kg/output/knowledge_graph.gml")
    if not os.path.exists(kg_path):
        print("  skipped (no KG at keo/kg/output/knowledge_graph.gml)")
        return
    g = nx.read_gml(kg_path)
    permuted = make_generator(shuffle_edges=True)._permute_graph_edges(g)

    assert set(permuted.nodes()) == set(g.nodes())
    retained = len(set(permuted.edges()) & set(g.edges())) / max(1, g.number_of_edges())
    assert retained < 0.1, f"permutation retained too many real pairs ({retained:.0%})"

    def components(graph):
        und = graph.to_undirected() if graph.is_directed() else graph
        return list(nx.connected_components(und))

    before, after = components(g), components(permuted)
    print(
        f"  real KG: {g.number_of_edges()}→{permuted.number_of_edges()} edges, "
        f"{retained:.1%} original pairs retained, "
        f"components {len(before)}→{len(after)}"
    )


def test_control_arms_are_judgeable():
    """Every control arm we can generate must be recognised by the evaluation loader."""
    from generate_ablation_answers import CONTROL_METHODS
    from run_evaluation import CHUNK_FAMILY, GRAPH_FAMILY, METHOD_FIELDS

    registered = {field for _, field in METHOD_FIELDS}
    families = set(CHUNK_FAMILY) | set(GRAPH_FAMILY)
    for arm, spec in CONTROL_METHODS.items():
        assert spec["field"] in registered, f"{arm} would be silently unjudged"
        assert arm in families, f"{arm} cannot stand in for its family in pairwise comparisons"
    # chunk_tok is the length-matched reference; it must outrank the unmatched arm's siblings
    assert CHUNK_FAMILY.index("chunk_tok") < CHUNK_FAMILY.index("chunk_sum")
    print(f"  {len(CONTROL_METHODS)} control arms registered for judging: {sorted(CONTROL_METHODS)}")


def main():
    run_suite("mechanism control", [
        ("edge permutation (keo_shuf)", test_edge_permutation),
        ("relation stripping (keo_norel)", test_relation_stripping),
        ("chunk dedup (chunk_dedup)", test_chunk_dedup),
        ("matched budget (chunk_tok)", test_matched_budget),
        ("budget escalation (chunk_tok)", test_budget_escalation),
        ("control provenance", test_control_config_provenance),
        ("controls are judgeable", test_control_arms_are_judgeable),
        ("real KG permutation", test_real_kg_permutation),
    ])


if __name__ == "__main__":
    main()
