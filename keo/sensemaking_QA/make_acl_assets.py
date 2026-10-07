#!/usr/bin/env python3
"""Offline ACL assets from judged runs: forest plot, dim flips, generated tex.

No LLM calls. Reads evaluation_results/ablations/*/global_sensemaking_*.json
and writes papers/latex/figures + papers/latex/generated + evaluation_results/stats.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

CR = (
    "global_perspective",
    "theme_identification",
    "synthesis_quality",
    "strategic_value",
    "pattern_recognition",
)
SHORT = {
    "global_perspective": "global",
    "theme_identification": "theme",
    "synthesis_quality": "synth",
    "strategic_value": "strategic",
    "pattern_recognition": "pattern",
}

ROOT = Path(__file__).resolve().parent
STATS = ROOT / "evaluation_results" / "stats"
FIGS = ROOT.parents[1] / "papers" / "latex" / "figures"
TEXGEN = ROOT.parents[1] / "papers" / "latex" / "generated"

STACKS = [
    ("Gemma E4B / Haiku", "_gemma4e4b"),
    ("Gemma E4B / Sonnet", "gemma4e4b_sonnet_all"),
    ("Phi-4 / Haiku", "phi4"),
    ("Phi-4 / Sonnet", "phi4_sonnet_all"),
    ("Phi-4 labeled / Haiku", "labeled_graph"),
    ("Gemma 31B / Haiku", "openrouter_gemma4_31b"),
    ("Gemma 31B / Sonnet", "openrouter_gemma4_31b_sonnet"),
]


def jdump(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=conv))


def load_scores(arm_dir: Path, method: str):
    files = sorted(
        p
        for p in arm_dir.glob("global_sensemaking_evaluation_*.json")
        if "judge_meta" not in p.name
    )
    if not files:
        return None
    d = json.loads(files[-1].read_text())
    bm = d.get("by_method") or {}
    key = method if method in bm else next(iter(bm))
    out = {}
    for e in bm[key].get("individual_evaluations") or []:
        if e.get("error"):
            continue
        q = e.get("question_id")
        gm = e.get("global_metrics") or {}
        dims = {k: float(gm[k]) for k in CR if isinstance(gm.get(k), (int, float))}
        if q and len(dims) == 5:
            out[q] = {"overall": float(np.mean(list(dims.values()))), "dims": dims}
    return out


def paired_chunk_minus_keo(keo, chunk):
    common = sorted(set(keo) & set(chunk))
    d = np.array([chunk[q]["overall"] - keo[q]["overall"] for q in common])
    rng = np.random.default_rng(0)
    obs = float(d.mean())
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    null = (rng.choice([-1.0, 1.0], size=(15000, len(d))) * d).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(obs)))
    return {
        "n": len(common),
        "delta": obs,
        "ci": [float(lo_ci), float(hi_ci)],
        "p_perm": p,
        "dz": float(obs / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
    }


def win_rate(keo, chunk):
    common = sorted(set(keo) & set(chunk))
    wins = sum(1 for q in common if chunk[q]["overall"] > keo[q]["overall"] + 1e-9)
    ties = sum(1 for q in common if abs(chunk[q]["overall"] - keo[q]["overall"]) <= 1e-9)
    return {
        "n": len(common),
        "chunk_wins": wins,
        "keo_wins": len(common) - wins - ties,
        "ties": ties,
        "chunk_win_rate": wins / len(common),
    }


def dim_disagreement(haiku_run, sonnet_run, name):
    h_root = ROOT / "evaluation_results" / "ablations" / haiku_run
    s_root = ROOT / "evaluation_results" / "ablations" / sonnet_run
    abs_by_dim = {k: [] for k in CR}
    h_keo = load_scores(h_root / "mst_on_freq", "keo")
    s_keo = load_scores(s_root / "mst_on_freq", "keo")
    h_ct = load_scores(h_root / "ctrl_chunk_tok", "chunk_tok")
    s_ct = load_scores(s_root / "ctrl_chunk_tok", "chunk_tok")
    sign_flip_chunk = {}
    for dim in CR:
        for arm, method in [("mst_on_freq", "keo"), ("ctrl_chunk_tok", "chunk_tok")]:
            h = load_scores(h_root / arm, method)
            s = load_scores(s_root / arm, method)
            for q in set(h) & set(s):
                abs_by_dim[dim].append(abs(s[q]["dims"][dim] - h[q]["dims"][dim]))
        common = sorted(set(h_keo) & set(h_ct) & set(s_keo) & set(s_ct))
        dh = float(
            np.mean([h_ct[q]["dims"][dim] - h_keo[q]["dims"][dim] for q in common])
        )
        ds = float(
            np.mean([s_ct[q]["dims"][dim] - s_keo[q]["dims"][dim] for q in common])
        )
        sign_flip_chunk[dim] = {
            "haiku_delta": dh,
            "sonnet_delta": ds,
            "flips": bool(
                np.sign(dh) != np.sign(ds) and abs(dh) > 0.02 and abs(ds) > 0.02
            ),
        }
    return {
        "generator": name,
        "median_abs_by_dim": {k: float(np.median(v)) for k, v in abs_by_dim.items()},
        "mean_abs_by_dim": {k: float(np.mean(v)) for k, v in abs_by_dim.items()},
        "chunk_tok_delta_by_dim": sign_flip_chunk,
    }


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    TEXGEN.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)

    rows = []
    win_rows = []
    for label, run in STACKS:
        base = ROOT / "evaluation_results" / "ablations" / run
        keo = load_scores(base / "mst_on_freq", "keo")
        chunk = load_scores(base / "ctrl_chunk_tok", "chunk_tok")
        if not keo or not chunk:
            print("SKIP", label)
            continue
        stats = paired_chunk_minus_keo(keo, chunk)
        common = set(keo) & set(chunk)
        rows.append(
            {
                "label": label,
                "run": run,
                **stats,
                "keo_mean": float(np.mean([keo[q]["overall"] for q in common])),
                "chunk_mean": float(np.mean([chunk[q]["overall"] for q in common])),
            }
        )
        wr = win_rate(keo, chunk)
        wr["label"] = label
        win_rows.append(wr)

    ps = np.array([r["p_perm"] for r in rows])
    order = list(np.argsort(ps))
    adj = [1.0] * len(ps)
    m = len(ps)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, ps[idx] * (m - i))
    for i in range(1, m):
        adj[order[i]] = max(adj[order[i]], adj[order[i - 1]])
    for r, h in zip(rows, adj):
        r["p_holm"] = float(h)
        r["sig"] = bool(h < 0.05)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y = np.arange(len(rows))[::-1]
    colors = []
    for r in rows:
        if r["delta"] > 0 and r["sig"]:
            colors.append("#2a6f97")
        elif r["delta"] < 0 and r["sig"]:
            colors.append("#9b2226")
        else:
            colors.append("#6c757d")
    for i, r in enumerate(rows):
        ax.plot(r["ci"], [y[i], y[i]], color=colors[i], lw=2.2, solid_capstyle="round")
        ax.plot(r["delta"], y[i], "o", color=colors[i], ms=7.5, zorder=3)
    ax.axvline(0, color="#222", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlabel(
        r"Paired $\Delta$: chunk_tok $-$ KEO  (positive = chunks higher)", fontsize=9
    )
    ax.set_title("Length-matched graph vs chunks across stacks", fontsize=11, pad=8)
    ax.set_xlim(-0.45, 0.45)
    ax.grid(axis="x", alpha=0.35)
    handles = [
        Line2D(
            [0], [0], marker="o", color="#2a6f97", lw=2, label="chunks higher (Holm < .05)"
        ),
        Line2D(
            [0], [0], marker="o", color="#9b2226", lw=2, label="KEO higher (Holm < .05)"
        ),
        Line2D([0], [0], marker="o", color="#6c757d", lw=2, label="underpowered / n.s."),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGS / "chunk_tok_forest.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "chunk_tok_forest.png", dpi=160, bbox_inches="tight")
    plt.close()

    dim_report = {
        "phi4": dim_disagreement("phi4", "phi4_sonnet_all", "phi4"),
        "gemma4e4b": dim_disagreement("_gemma4e4b", "gemma4e4b_sonnet_all", "gemma4e4b"),
        "gemma4_31b": dim_disagreement(
            "openrouter_gemma4_31b", "openrouter_gemma4_31b_sonnet", "gemma4_31b"
        ),
    }
    jdump(dim_report, STATS / "judge_disagreement_by_dimension.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, key, title in zip(axes, ["phi4", "gemma4e4b"], ["Phi-4", "Gemma-4 E4B"]):
        med = dim_report[key]["median_abs_by_dim"]
        labs = [SHORT[k] for k in CR]
        vals = [med[k] for k in CR]
        flips = [dim_report[key]["chunk_tok_delta_by_dim"][k]["flips"] for k in CR]
        cols = ["#9b2226" if f else "#4a4e69" for f in flips]
        ax.barh(labs[::-1], vals[::-1], color=cols[::-1])
        ax.set_xlabel("median |Haiku−Sonnet|")
        ax.set_title(title)
        ax.set_xlim(0, 0.8)
        ax.axvline(0.2, color="#aaa", ls=":", lw=1)
    axes[0].set_ylabel("rubric dimension")
    fig.suptitle(
        "Where judges diverge (red = chunk_tok Δ sign flips on that dim)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "judge_dim_disagreement.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "judge_dim_disagreement.png", dpi=160, bbox_inches="tight")
    plt.close()

    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Stack & KEO & chunk\_tok & $\Delta$ & 95\% CI & Reading \\",
        r"\midrule",
    ]
    for r in rows:
        if r["delta"] > 0 and r["sig"]:
            reading = "chunks higher"
        elif r["delta"] < 0 and r["sig"]:
            reading = "KEO higher"
        else:
            reading = "underpowered"
        lines.append(
            f"{r['label']} & {r['keo_mean']:.3f} & {r['chunk_mean']:.3f} & "
            f"${r['delta']:+.3f}$ & $[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]$ & {reading} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEXGEN / "tab_chunk_tok_forest.tex").write_text("\n".join(lines) + "\n")

    dlines = [
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Generator & Dimension & Haiku $\Delta$ & Sonnet $\Delta$ & Sign flip? \\",
        r"\midrule",
    ]
    for gen, key in [
        ("Phi-4", "phi4"),
        ("Gemma-4 E4B", "gemma4e4b"),
        ("Gemma-4 31B", "gemma4_31b"),
    ]:
        for dim in CR:
            v = dim_report[key]["chunk_tok_delta_by_dim"][dim]
            dlines.append(
                f"{gen} & {SHORT[dim]} & ${v['haiku_delta']:+.3f}$ & "
                f"${v['sonnet_delta']:+.3f}$ & {'yes' if v['flips'] else 'no'} \\\\"
            )
    dlines += [r"\bottomrule", r"\end{tabular}"]
    (TEXGEN / "tab_dim_sign_flips.tex").write_text("\n".join(dlines) + "\n")

    jdump(
        {
            "chunk_tok_deltas": rows,
            "win_rates": win_rows,
            "dim_disagreement": dim_report,
        },
        STATS / "acl_cross_stack_summary.json",
    )
    print(f"Wrote {FIGS}/chunk_tok_forest.pdf and related assets")
    for w in win_rows:
        print(
            f"  {w['label']:28} chunk_win={w['chunk_win_rate']:.1%} "
            f"({w['chunk_wins']}/{w['n']})"
        )


if __name__ == "__main__":
    main()
