#!/usr/bin/env python3
"""
Graph-context-only ablation smoke (no chat LLM).
Requires OPENAI_API_KEY only if embeddings are missing from cache;
prefers precomputed kg_cache when present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def serialize_component(g: nx.Graph, use_mst: bool) -> int:
    """Return number of edges used in context serialization."""
    und = g.to_undirected() if g.is_directed() else g
    total = 0
    for comp in nx.connected_components(und):
        sub = und.subgraph(comp).copy()
        if sub.number_of_edges() == 0:
            continue
        for u, v, d in sub.edges(data=True):
            try:
                d["weight"] = float(d.get("weight", 1.0))
            except Exception:
                d["weight"] = 1.0
        structure = nx.maximum_spanning_tree(sub, weight="weight") if use_mst else sub
        total += structure.number_of_edges()
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg", default="../kg/output/knowledge_graph.gml")
    p.add_argument("--out", default="evaluation_results/ablations/graph_context_smoke.json")
    args = p.parse_args()

    g = nx.read_gml(args.kg)
    on = serialize_component(g, True)
    off = serialize_component(g, False)
    payload = {
        "kg": args.kg,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "serialized_edges_mst_on": on,
        "serialized_edges_mst_off": off,
        "reduction": (off - on) / max(off, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
