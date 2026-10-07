#!/usr/bin/env python3
"""Theme score against entity yield, and yield against answer length.

The length panel is the check: if yield were verbosity, the right-hand cloud
would slope. It does not. No LLM call; reads the judged Phi-4 stats plus the
answer files those stats were computed from.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _answers_by_id(path: str) -> dict:
    rows = json.load(open(path))
    out = {}
    for row in rows:
        ans = row.get("keo_answer") or {}
        qid = ans.get("question_id") or row.get("question_id")
        if qid and ans.get("answer"):
            out[qid] = ans
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        default=os.path.join(
            here, "evaluation_results", "stats",
            "prior_residual_phi4_matched_prior.json",
        ),
    )
    parser.add_argument(
        "--answers",
        default=os.path.join(
            here, "evaluation_results", "ablations",
            "phi4", "mst_on_freq", "answers.json",
        ),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            here, "..", "..", "papers", "latex", "figures", "entity_yield.pdf",
        ),
    )
    args = parser.parse_args()
    blob = json.load(open(args.stats))
    rows = blob.get("entity_yield_per_question") or []
    if not rows:
        raise SystemExit(f"{args.stats} has no entity_yield_per_question")
    answers = _answers_by_id(args.answers)
    xs, themes, lengths = [], [], []
    for rec in rows:
        ans = answers.get(rec["question_id"])
        if not ans:
            continue
        xs.append(float(rec["Y"]))
        themes.append(float(rec["theme_identification"]))
        lengths.append(len(ans.get("answer") or ""))
    if len(xs) < 8:
        raise SystemExit(f"only {len(xs)} questions aligned")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    axes[0].scatter(xs, themes, s=18, color="#1f4e79", alpha=0.75, linewidths=0)
    axes[0].set_xlabel(r"Entity yield $Y$")
    axes[0].set_ylabel("Theme identification")
    axes[0].set_title(r"Theme, $\rho=+0.37$")
    axes[1].scatter(xs, lengths, s=18, color="#1f4e79", alpha=0.75, linewidths=0)
    axes[1].set_xlabel(r"Entity yield $Y$")
    axes[1].set_ylabel("Answer length (characters)")
    axes[1].set_title(r"Length, $\rho=-0.03$")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} n={len(xs)}")


if __name__ == "__main__":
    main()
