#!/usr/bin/env python3
"""Offline smoke test for MST / edge-weight ablations (no API calls)."""

from __future__ import annotations

import os
import sys
import tempfile

import networkx as nx
import numpy as np

# Allow importing sibling modules without loading OpenAI clients for graph-only tests
sys.path.insert(0, os.path.dirname(__file__))

from offline_harness import run_suite  # noqa: E402


def build_toy_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edge("engine quit", "fuel sumps frozen", weight=21.0, relation="caused_by")
    g.add_edge("engine quit", "water in fuel", weight=8.0, relation="caused_by")
    g.add_edge("fuel sumps frozen", "winter ops", weight=5.0, relation="associated_with")
    g.add_edge("water in fuel", "winter ops", weight=3.0, relation="associated_with")
    g.add_edge("landing gear", "hydraulic leak", weight=12.0, relation="exhibits")
    return g


def reweight(g: nx.DiGraph, scheme: str) -> dict:
    """Mirror freq/tfidf/pmi formulas without SensemakingAnswerGenerator."""
    base = {(u, v): float(d.get("weight", 1.0)) for u, v, d in g.edges(data=True)}
    n = max(g.number_of_nodes(), 1)
    degrees = dict(g.degree())
    total = sum(base.values()) or 1.0
    out = {}
    for (u, v), b in base.items():
        if scheme == "freq":
            w = b
        elif scheme == "tfidf":
            idf_u = np.log((n + 1) / (degrees.get(u, 1) + 1)) + 1.0
            idf_v = np.log((n + 1) / (degrees.get(v, 1) + 1)) + 1.0
            w = b * 0.5 * (idf_u + idf_v)
        elif scheme == "pmi":
            p_uv = (b + 1e-3) / (total + 1e-3)
            deg_sum = sum(degrees.values()) + 1e-3
            p_u = (degrees.get(u, 1) + 1e-3) / deg_sum
            p_v = (degrees.get(v, 1) + 1e-3) / deg_sum
            w = max(np.log(p_uv / (p_u * p_v + 1e-12)), 0.0) * b
        else:
            raise ValueError(scheme)
        out[(u, v)] = float(w)
    return out


def mst_vs_full(g: nx.DiGraph):
    u = g.to_undirected()
    # collapse parallel by summing weights
    simple = nx.Graph()
    for a, b, d in u.edges(data=True):
        w = float(d.get("weight", 1.0))
        if simple.has_edge(a, b):
            simple[a][b]["weight"] += w
        else:
            simple.add_edge(a, b, weight=w)
    components = list(nx.connected_components(simple))
    stats = []
    for comp in components:
        sub = simple.subgraph(comp).copy()
        if sub.number_of_edges() == 0:
            continue
        mst = nx.maximum_spanning_tree(sub, weight="weight")
        stats.append(
            {
                "n_nodes": sub.number_of_nodes(),
                "full_edges": sub.number_of_edges(),
                "mst_edges": mst.number_of_edges(),
            }
        )
    return stats


def test_gml_roundtrip():
    g = build_toy_graph()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "toy.gml")
        nx.write_gml(g, path)
        loaded = nx.read_gml(path)
        assert loaded.number_of_nodes() == g.number_of_nodes(), loaded.number_of_nodes()
        assert loaded.number_of_edges() == g.number_of_edges(), loaded.number_of_edges()
    print(f"  {g.number_of_nodes()} nodes / {g.number_of_edges()} edges survive GML")


def test_reweighting_is_non_negative_and_distinct():
    """Weights stay usable as MST inputs, and the schemes are not all identical."""
    g = build_toy_graph()
    produced = {}
    for scheme in ("freq", "tfidf", "pmi"):
        weights = reweight(g, scheme)
        assert all(w >= 0 for w in weights.values()), f"{scheme}: negative weight"
        assert len(weights) == g.number_of_edges(), scheme
        produced[scheme] = tuple(round(weights[e], 6) for e in sorted(weights))
        print(f"  {scheme}: { {f'{u}->{v}': round(w, 3) for (u, v), w in weights.items()} }")
    assert len(set(produced.values())) == 3, "two schemes produced identical weights"
    # Note: all three happen to rank this toy graph the same way, because its degrees are
    # nearly uniform. Reordering is tested below on a graph built to expose it.


def test_tfidf_downweights_hub_edges():
    """The point of tfidf/pmi: a heavy edge between hubs can lose to a lighter rare edge.

    Without this the ablation could be weight-preserving in rank and the MST would be
    identical under every scheme, making the whole track a no-op.
    """
    g = nx.DiGraph()
    g.add_edge("hub a", "hub b", weight=10.0, relation="associated_with")
    for i in range(5):
        g.add_edge("hub a", f"filler a{i}", weight=1.0, relation="associated_with")
        g.add_edge("hub b", f"filler b{i}", weight=1.0, relation="associated_with")
    g.add_edge("rare x", "rare y", weight=9.0, relation="caused_by")

    hub, rare = ("hub a", "hub b"), ("rare x", "rare y")
    freq = reweight(g, "freq")
    assert freq[hub] > freq[rare], "raw frequency should prefer the heavier hub edge"

    tfidf = reweight(g, "tfidf")
    assert tfidf[rare] > tfidf[hub], (
        f"tfidf did not downweight the hub edge: hub={tfidf[hub]:.2f} rare={tfidf[rare]:.2f}"
    )
    print(f"  freq prefers hub ({freq[hub]:.1f} > {freq[rare]:.1f}); "
          f"tfidf flips it ({tfidf[rare]:.1f} > {tfidf[hub]:.1f})")


def test_mst_actually_drops_edges():
    """The previous assertion was `mst < full or mst == full` — i.e. `<=`, always true."""
    g = build_toy_graph()
    stats = mst_vs_full(g)
    assert stats, "no components with edges"
    print(f"  MST ablation stats: {stats}")

    for s in stats:
        # Spanning tree of a connected component: exactly n-1 edges.
        assert s["mst_edges"] == s["n_nodes"] - 1, s
        assert s["mst_edges"] <= s["full_edges"], s

    cyclic = [s for s in stats if s["full_edges"] > s["n_nodes"] - 1]
    assert cyclic, f"toy graph has no cyclic component to prune: {stats}"
    for s in cyclic:
        assert s["mst_edges"] < s["full_edges"], f"MST pruned nothing from {s}"
    print(f"  {len(cyclic)} cyclic component(s) genuinely pruned")


def test_real_kg_if_present():
    real = os.path.join(os.path.dirname(__file__), "../kg/output/knowledge_graph.gml")
    if not os.path.exists(real):
        print(f"  skipped: {real} not present")
        return
    rg = nx.read_gml(real)
    assert rg.number_of_nodes() > 0, "real KG loaded but empty"
    und = rg.to_undirected() if rg.is_directed() else rg
    ccs = list(nx.connected_components(und))
    print(f"  real KG: {rg.number_of_nodes()} nodes, {rg.number_of_edges()} edges, "
          f"multi={rg.is_multigraph()}, components={len(ccs)}")


def main():
    run_suite("offline ablation", [
        ("GML roundtrip", test_gml_roundtrip),
        ("reweighting non-negative and distinct", test_reweighting_is_non_negative_and_distinct),
        ("tfidf downweights hub edges", test_tfidf_downweights_hub_edges),
        ("MST drops edges", test_mst_actually_drops_edges),
        ("real KG if present", test_real_kg_if_present),
    ])


if __name__ == "__main__":
    main()
