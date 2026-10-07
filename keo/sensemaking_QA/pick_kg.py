#!/usr/bin/env python3
"""
Choose a usable knowledge graph, or say plainly that there is not one.

The KG directories contain many 23-byte stubs — literally `graph [ directed 1 ]` — written
by earlier `*_fixed` repair scripts, sitting alongside the real graph under names that only
differ by suffix (`..._with_entity_mentions_fixed.gml` has 160 nodes;
`..._with_entity_mentions_fixed_edges_fixed.gml` has none). The sweep used to take whatever
`find` listed first, which was a stub, so every graph arm ran against an empty graph and
produced null answers that looked like a method performing badly.

Usage:
  python pick_kg.py DIR                         # print the path with the most nodes
  python pick_kg.py --check PATH                # exit 0 if PATH has at least one node
  python pick_kg.py --check --require-annotated PATH
                                                # also require relation labels and >1 weight
Exit codes: 0 usable, 1 nothing usable or not annotated.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def graph_size(path: str) -> tuple:
    """(nodes, edges) for a GML file, or (0, 0) if it cannot be read."""
    import networkx as nx

    try:
        g = nx.read_gml(path)
    except Exception:
        return (0, 0)
    return (g.number_of_nodes(), g.number_of_edges())


_GENERIC_RELATIONS = {"", "related_to", "relates to", "connected to", "none"}


def annotation_status(path: str) -> dict:
    """
    Whether a graph can support the mechanism controls.

    The judged control graph stores bare source--target edges. On that file every
    weight defaults to 1 and every relation falls back to a generic phrase, so
    ``keo_norel`` and a frequency spanning tree do not test what their names say.
    """
    import networkx as nx

    g = nx.read_gml(path)
    weights = set()
    labeled = 0
    edges = g.edges(keys=True, data=True) if g.is_multigraph() else (
        (u, v, 0, d) for u, v, d in g.edges(data=True)
    )
    n_edges = 0
    for _u, _v, _k, data in edges:
        n_edges += 1
        try:
            weights.add(float(data.get("weight", 1.0)))
        except (TypeError, ValueError):
            weights.add(1.0)
        rel = (
            data.get("relation") or data.get("relationship") or data.get("relations")
            or data.get("label") or data.get("type") or ""
        )
        if str(rel).strip().lower() not in _GENERIC_RELATIONS:
            labeled += 1
    return {
        "nodes": g.number_of_nodes(),
        "edges": n_edges,
        "distinct_weights": len(weights),
        "labeled_edges": labeled,
    }


def is_annotated(status: dict) -> bool:
    """True when some edge has a real relation label and weights are not all equal."""
    return status["labeled_edges"] > 0 and status["distinct_weights"] > 1


def best_graph(directory: str) -> str | None:
    """The .gml under `directory` with the most nodes, ignoring empties."""
    candidates = []
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.gml"), recursive=True)):
        if os.path.getsize(path) < 64:  # stubs, skipped without paying to parse them
            continue
        nodes, edges = graph_size(path)
        if nodes:
            candidates.append((nodes, edges, path))
    if not candidates:
        return None
    # Prefer most nodes; break ties on edges, then on the shorter name, which is the
    # original rather than a ..._fixed_fixed re-derivation.
    candidates.sort(key=lambda c: (c[0], c[1], -len(c[2])), reverse=True)
    return candidates[0][2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="directory to search, or file to check")
    parser.add_argument("--check", action="store_true",
                        help="treat target as a file and verify it has nodes")
    parser.add_argument("--require-annotated", action="store_true",
                        help="with --check, also require relation labels and more than one weight")
    args = parser.parse_args()

    if args.check:
        if not os.path.exists(args.target):
            print(f"KG not found: {args.target}", file=sys.stderr)
            return 1
        nodes, edges = graph_size(args.target)
        if not nodes:
            print(
                f"KG is empty: {args.target} ({os.path.getsize(args.target)} bytes, "
                f"{nodes} nodes). This is almost certainly one of the *_fixed stubs.",
                file=sys.stderr,
            )
            return 1
        print(f"✓ KG OK: {args.target} ({nodes} nodes, {edges} edges)")
        if args.require_annotated:
            status = annotation_status(args.target)
            if not is_annotated(status):
                print(
                    f"KG is not annotated: {args.target} "
                    f"({status['labeled_edges']} labeled edges, "
                    f"{status['distinct_weights']} distinct weights). "
                    "Mechanism controls on this file do not test relation labels or "
                    "frequency weights. Pass the 1,468-edge gemma3 graph, or set "
                    "KEO_ALLOW_UNLABELED_KG=1 to keep this file.",
                    file=sys.stderr,
                )
                return 1
            print(
                f"✓ KG annotated: {status['labeled_edges']} labeled edges, "
                f"{status['distinct_weights']} distinct weights"
            )
        return 0

    chosen = best_graph(args.target)
    if not chosen:
        print(f"No non-empty .gml under {args.target}", file=sys.stderr)
        return 1
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
