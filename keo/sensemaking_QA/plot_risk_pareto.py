#!/usr/bin/env python3
"""Draw the judge-vs-safety Pareto figure from an existing risk_retrieval JSON.

No LLM, no embeddings — just the numbers already measured. MST is the judged-
quality selector and the worst safety policy; that contrast is the paper's
sharpest generator-free claim and it deserves a figure, not only a table.
"""

from __future__ import annotations

import argparse
import json
import os

# Keep matplotlib off the user's home cache while the sweep holds the machine.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POLICY_ORDER = [
    ("mst", "MST (KEO)", "#b22222"),
    ("top_weight", "Freq. top-weight", "#888888"),
    ("budgeted_coverage", "Coverage", "#4682b4"),
    ("risk_budgeted_coverage", "Risk", "#2e8b57"),
    ("risk_floor_0.25", r"Risk$_{\rho=.25}$", "#006400"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, help="risk_retrieval_*.json")
    parser.add_argument("--out", required=True, help="PDF or PNG path")
    args = parser.parse_args()

    blob = json.load(open(args.stats))
    budgets = blob.get("budgets") or []
    if not budgets:
        raise SystemExit(f"{args.stats} has no budgets[]")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = [b["budget_tokens"] for b in budgets]
    for key, label, color in POLICY_ORDER:
        ys = []
        for b in budgets:
            pol = (b.get("policies") or {}).get(key)
            if not pol:
                ys.append(None)
            else:
                ys.append(100.0 * float(pol["high_severity_recall"]))
        if any(v is None for v in ys):
            continue
        ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=2.0)

    ax.set_xlabel("Token budget")
    ax.set_ylabel("Critical recall (%)")
    ax.set_title("Safety recall at equal context budget")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, max(70, ax.get_ylim()[1]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
