#!/usr/bin/env python3
"""Build a tiny undirected/directed KG from a CSV text column (second-domain pilot)."""

from __future__ import annotations

import argparse
import re
from collections import Counter

import networkx as nx
import pandas as pd


STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "from", "by", "was", "were", "is", "are", "be", "been", "this", "that",
}


def tokens(text: str):
    return [t for t in re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower()) if t not in STOP and len(t) > 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-col", default="narrative")
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--top-terms", type=int, default=40)
    args = p.parse_args()

    df = pd.read_csv(args.csv).head(args.max_rows)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column {args.text_col} not in {list(df.columns)}")

    term_counts = Counter()
    row_terms = []
    for text in df[args.text_col].dropna().astype(str):
        ts = tokens(text)
        row_terms.append(ts)
        term_counts.update(set(ts))

    vocab = {t for t, _ in term_counts.most_common(args.top_terms)}
    g = nx.DiGraph()
    for ts in row_terms:
        present = [t for t in ts if t in vocab]
        for i, a in enumerate(present):
            for b in present[i + 1 : i + 4]:
                if a == b:
                    continue
                if g.has_edge(a, b):
                    g[a][b]["weight"] += 1
                else:
                    g.add_edge(a, b, weight=1.0, relation="co_occurs_with")

    nx.write_gml(g, args.out)
    print(f"Wrote {g.number_of_nodes()} nodes / {g.number_of_edges()} edges → {args.out}")


if __name__ == "__main__":
    main()
