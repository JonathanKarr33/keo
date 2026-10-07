#!/usr/bin/env python3
"""Schematic of the judged experiments. Does not replace figures/workflow.pdf.

Writes papers/latex/figures/experiment_schematic.pdf. The paper still uses the
original pipeline figure and says so in the caption.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def _box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor="#222", linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("What the judged experiments actually compare", fontsize=11)

    _box(ax, 0.3, 3.3, 2.6, 1.2, "Labeled graph\n(weights + relations)\nnot yet re-judged", "#f4e4c8")
    _box(ax, 3.5, 3.3, 2.8, 1.2, "Control graph\nno weights, no labels\njudged arms live here", "#f6d5d5")
    _box(ax, 6.9, 3.3, 2.8, 1.2, "Length-matched\nchunks\nmeaningful contrast", "#d9ead3")
    _box(ax, 0.3, 0.6, 4.4, 1.6, "Same Phi-4 answers\nHaiku vs Sonnet\n(3 arms today)", "#d6e3f0")
    _box(ax, 5.2, 0.6, 4.5, 1.6, "Severity floor vs tie-break\nscored on the same lexicon\nexpert labels still blank", "#e6e6e6")

    out = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../papers/latex/figures/experiment_schematic.pdf",
    ))
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
