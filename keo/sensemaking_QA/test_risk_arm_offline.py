#!/usr/bin/env python3
"""
Offline checks for the `risk` generation arm (no API calls, no LLM).

This exercises the real retrieval path — `_retrieve_graph_context` with
selector="risk" — on a hash embedder, so the selection, serialisation, evidence and
provenance are all tested end to end. Only the final LLM call is absent.

What must hold:
  - the risk arm selects a *different* edge set than MST at the same budget
  - it respects its token budget and reserves the risk floor
  - it recalls more high-severity edges than MST on the same question
  - its prompt surface is identical to KEO's, so a judge cannot reward format
  - it is registered everywhere it needs to be, or it would be silently unjudged

Run: python test_risk_arm_offline.py
"""

from __future__ import annotations

import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer_generator import SELECTORS, SensemakingAnswerGenerator  # noqa: E402
from local_embeddings import EmbeddingBackend  # noqa: E402
from offline_harness import run_suite  # noqa: E402
from severity_lexicon import HIGH_SEVERITY_THRESHOLD, edge_severity  # noqa: E402

KG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../kg/output/knowledge_graph.gml"
)


def make_generator(selector="risk", **kwargs) -> SensemakingAnswerGenerator:
    """
    Generator wired for retrieval only: hash embeddings, no API client.

    Bypasses __init__ so no network or key is needed, then sets exactly the attributes
    the graph-retrieval path reads.
    """
    gen = SensemakingAnswerGenerator.__new__(SensemakingAnswerGenerator)
    gen.selector = selector
    gen.use_mst = kwargs.get("use_mst", True)
    gen.risk_budget_tokens = kwargs.get("risk_budget_tokens", 1200)
    gen.risk_floor = kwargs.get("risk_floor", 0.25)
    gen.risk_fill_budget = kwargs.get("risk_fill_budget", True)
    gen.shuffle_edges = False
    gen.strip_relations = kwargs.get("strip_relations", False)
    gen.dedup_chunks = False
    gen.match_context_budget = False
    gen.control_seed = 13
    gen.edge_weight = "freq"
    gen.embedding_backend_name = "hash"
    gen.embedder = EmbeddingBackend(backend="hash")
    gen.embedding_client = gen.embedder
    gen._risk_selection_stats = []
    gen._last_evidence = {}
    gen.knowledge_graph = kwargs.get("knowledge_graph")
    gen.graph_retriever = kwargs.get("graph_retriever")
    return gen


class StubRetriever:
    """Minimal stand-in for GraphRetriever: hash embeddings + cosine similarity."""

    def __init__(self, graph, embedder):
        self.embeddings = {node: embedder.embed(str(node)) for node in graph.nodes()}

    @staticmethod
    def calculate_semantic_similarity(a, b):
        import numpy as np

        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)


def build_safety_graph() -> nx.Graph:
    """
    A graph with the structure the pivot is about: a frequent mundane backbone plus
    rare high-severity facts hanging off it.
    """
    g = nx.Graph()
    for i in range(30):
        g.add_edge("inspection", f"routine item {i}", weight=12.0, relation="covered")
    for i in range(6):
        g.add_edge(f"aircraft {i}", "engine fire", weight=1.0, relation="experienced")
        g.add_edge(f"aircraft {i}", "inspection", weight=4.0, relation="underwent")
    g.add_edge("fuel line", "uncontained failure", weight=1.0, relation="led to")
    g.add_edge("fuel line", "inspection", weight=6.0, relation="covered by")
    return g


def wire(graph, **kwargs):
    gen = make_generator(knowledge_graph=graph, **kwargs)
    gen.graph_retriever = StubRetriever(graph, gen.embedder)
    return gen


def _selected_edges(evidence):
    return {(t["head"], t["tail"]) for t in evidence["triples"]}


def test_selector_validation():
    """A typo in the selector must fail loudly, not silently fall back to MST."""
    assert SELECTORS == ("mst", "subgraph", "risk")
    try:
        SensemakingAnswerGenerator(api_key="x", provider="local", selector="rsk")
    except ValueError as e:
        assert "selector must be one of" in str(e), e
        print("  invalid selector raises ValueError")
        return
    raise AssertionError("invalid selector was accepted")


def test_risk_selects_differently_than_mst():
    graph = build_safety_graph()
    question = "What engine fire hazards appear in the fleet?"

    mst_gen = wire(graph, selector="mst")
    risk_gen = wire(graph, selector="risk", risk_budget_tokens=400)

    mst_context = mst_gen._retrieve_graph_context(question, 2, 10)
    risk_context = risk_gen._retrieve_graph_context(question, 2, 10)

    assert "Error retrieving graph context" not in mst_context, mst_context
    assert "Error retrieving graph context" not in risk_context, risk_context

    mst_edges = _selected_edges(mst_gen._last_evidence)
    risk_edges = _selected_edges(risk_gen._last_evidence)
    assert mst_edges, "MST arm produced no evidence"
    assert risk_edges, "risk arm produced no evidence"
    assert mst_edges != risk_edges, "risk selector reproduced the MST selection exactly"
    print(
        f"  MST selected {len(mst_edges)} edges, risk selected {len(risk_edges)}; "
        f"{len(mst_edges & risk_edges)} shared"
    )


def test_risk_recalls_more_critical_edges():
    """The claim, on the real retrieval path rather than on synthetic records."""
    graph = build_safety_graph()
    question = "What engine fire hazards appear in the fleet?"

    def critical_recall(gen):
        gen._retrieve_graph_context(question, 2, 10)
        triples = gen._last_evidence["triples"]
        chosen = sum(
            1 for t in triples
            if edge_severity(t["head"], t["relation"], t["tail"]) >= HIGH_SEVERITY_THRESHOLD
        )
        return chosen, len(triples)

    mst_hits, mst_total = critical_recall(wire(graph, selector="mst"))
    risk_hits, risk_total = critical_recall(
        wire(graph, selector="risk", risk_budget_tokens=400)
    )
    assert risk_hits > mst_hits, (mst_hits, risk_hits)
    print(
        f"  critical edges surfaced: MST {mst_hits}/{mst_total} vs "
        f"risk {risk_hits}/{risk_total}"
    )


def test_budget_and_floor_are_respected():
    graph = build_safety_graph()
    gen = wire(graph, selector="risk", risk_budget_tokens=300, risk_floor=0.3)
    gen._retrieve_graph_context("engine fire", 2, 10)

    stats = gen._last_evidence["risk_selection"]
    assert stats["tokens_used"] <= 300, stats
    assert stats["reserved_tokens_used"] <= int(300 * 0.3), stats
    assert stats["selected_edges"] > 0
    assert stats["budget_tokens"] == 300 and stats["risk_floor"] == 0.3
    print(
        f"  used {stats['tokens_used']}/{stats['budget_tokens']} tokens, "
        f"{stats['reserved_tokens_used']} reserved, "
        f"critical recall {stats['high_severity_recall']:.0%}"
    )


def test_floor_zero_still_works():
    """risk_floor=0 must reduce to the plain multiplier objective, not crash."""
    graph = build_safety_graph()
    gen = wire(graph, selector="risk", risk_budget_tokens=300, risk_floor=0.0)
    gen._retrieve_graph_context("engine fire", 2, 10)
    stats = gen._last_evidence["risk_selection"]
    assert stats["reserved_tokens_used"] == 0, stats
    assert stats["selected_edges"] > 0
    print(f"  risk_floor=0: {stats['selected_edges']} edges, no reserved spend")


def test_fill_budget_controls_context_length():
    """
    Coverage saturates well short of B, so without filling the risk arm ships much
    shorter context than MST — which would hand the length-preferring judge a reason to
    prefer MST regardless of content. Filling removes that asymmetry.
    """
    graph = build_safety_graph()
    question = "What engine fire hazards appear in the fleet?"

    saturating = wire(graph, selector="risk", risk_budget_tokens=600, risk_fill_budget=False)
    filling = wire(graph, selector="risk", risk_budget_tokens=600, risk_fill_budget=True)
    lean_context = saturating._retrieve_graph_context(question, 2, 10)
    full_context = filling._retrieve_graph_context(question, 2, 10)

    lean = saturating._last_evidence["risk_selection"]
    full = filling._last_evidence["risk_selection"]

    assert lean["filler_edges"] == 0, lean
    assert full["filler_edges"] > 0, "nothing was filled despite leftover budget"
    assert full["tokens_used"] > lean["tokens_used"], (lean, full)
    assert full["tokens_used"] <= 600
    assert len(full_context) > len(lean_context)
    # Filling must not cost safety recall: it only adds edges
    assert full["high_severity_recall"] >= lean["high_severity_recall"]
    print(
        f"  saturated at {lean['tokens_used']}/600 tokens ({len(lean_context)} chars); "
        f"filled to {full['tokens_used']} tokens ({len(full_context)} chars) with "
        f"{full['filler_edges']} filler edges, recall unchanged at "
        f"{full['high_severity_recall']:.0%}"
    )


def test_prompt_surface_matches_keo():
    """
    Format parity: the risk arm must emit the same section labels as KEO, so a judge
    cannot reward it (or penalise it) for surface form.
    """
    graph = build_safety_graph()
    question = "What engine fire hazards appear in the fleet?"
    mst_context = wire(graph, selector="mst")._retrieve_graph_context(question, 2, 10)
    risk_context = wire(
        graph, selector="risk", risk_budget_tokens=400
    )._retrieve_graph_context(question, 2, 10)

    header = "Graph-based Knowledge Context:"
    assert mst_context.startswith(header) and risk_context.startswith(header)
    assert "Knowledge Path" in risk_context, risk_context[:200]
    assert "Knowledge Subgraph" not in risk_context
    print("  risk arm uses the same 'Knowledge Path' surface as KEO")


def test_selection_provenance():
    graph = build_safety_graph()
    gen = wire(graph, selector="risk", risk_budget_tokens=800, risk_floor=0.4)
    config = gen.selection_config()
    assert config == {
        "selector": "risk", "use_mst": True,
        "risk_budget_tokens": 800, "risk_floor": 0.4, "risk_fill_budget": True,
    }, config

    mst_config = wire(graph, selector="mst").selection_config()
    assert mst_config == {"selector": "mst", "use_mst": True}, mst_config
    assert "risk_floor" not in mst_config, "MST runs must not claim risk settings"

    for question in ("engine fire", "hydraulic leak", "winter operations"):
        gen._retrieve_graph_context(question, 2, 8)
    summary = gen.risk_selection_summary()
    assert summary["questions"] == 3, summary
    assert summary["budget_tokens"] == 800
    assert summary["mean_tokens_used"] <= 800
    assert not wire(graph, selector="mst").risk_selection_summary()
    print(
        f"  summary over {summary['questions']} questions: "
        f"{summary['mean_selected_edges']:.1f} edges, "
        f"{summary['mean_tokens_used']:.0f} tokens mean"
    )


def test_risk_arm_is_judgeable():
    """Registration check: an unregistered arm is generated and then silently ignored."""
    from generate_ablation_answers import ANSWER_FIELDS, METHOD_CHOICES, SELECTION_METHODS
    from run_evaluation import GRAPH_FAMILY, METHOD_FIELDS
    from judge_length_bias import ARM_TO_FIELD
    from paired_stats import _ARM_PRIORITY

    registered = {field for _, field in METHOD_FIELDS}
    for arm, spec in SELECTION_METHODS.items():
        assert arm in METHOD_CHOICES, f"{arm} not selectable on the CLI"
        assert spec["field"] in ANSWER_FIELDS, f"{arm} field not written to answers files"
        assert spec["field"] in registered, f"{arm} would be silently unjudged"
        assert arm in GRAPH_FAMILY, f"{arm} cannot stand in for the graph family"
        assert ARM_TO_FIELD.get(arm) == spec["field"], f"{arm} missing from length audit"
        assert arm in _ARM_PRIORITY, f"{arm} missing from paired-stats arm priority"
    print(f"  {sorted(SELECTION_METHODS)} registered across generation, judging and analysis")


def test_real_kg_end_to_end():
    """Run the arm on the real OMIn KG when it is present."""
    if not os.path.exists(KG_PATH):
        print("  skipped (no KG at keo/kg/output/knowledge_graph.gml)")
        return
    graph = nx.read_gml(KG_PATH)
    question = "What recurring engine problems led to in-flight failures?"

    mst_gen, risk_gen = wire(graph, selector="mst"), wire(
        graph, selector="risk", risk_budget_tokens=1200
    )
    mst_context = mst_gen._retrieve_graph_context(question, 2, 15)
    risk_context = risk_gen._retrieve_graph_context(question, 2, 15)

    for name, context in (("mst", mst_context), ("risk", risk_context)):
        assert "Error retrieving graph context" not in context, (name, context)

    stats = risk_gen._last_evidence["risk_selection"]
    assert stats["tokens_used"] <= 1200
    mst_edges = len(mst_gen._last_evidence["triples"])
    print(
        f"  real KG: MST surfaced {mst_edges} triples ({len(mst_context)} chars); "
        f"risk selected {stats['selected_edges']}/{stats['candidate_edges']} candidate "
        f"edges in {stats['tokens_used']} tokens ({len(risk_context)} chars), "
        f"critical recall {stats['high_severity_recall']:.0%} "
        f"of {stats['high_severity_total']}"
    )


def main():
    tests = [
        ("selector validation", test_selector_validation),
        ("risk selects differently", test_risk_selects_differently_than_mst),
        ("risk recalls more critical edges", test_risk_recalls_more_critical_edges),
        ("budget and floor respected", test_budget_and_floor_are_respected),
        ("risk_floor=0 degrades cleanly", test_floor_zero_still_works),
        ("fill_budget controls length", test_fill_budget_controls_context_length),
        ("prompt surface parity", test_prompt_surface_matches_keo),
        ("selection provenance", test_selection_provenance),
        ("risk arm is judgeable", test_risk_arm_is_judgeable),
        ("real KG end to end", test_real_kg_end_to_end),
    ]
    run_suite("risk arm", tests)


if __name__ == "__main__":
    main()
