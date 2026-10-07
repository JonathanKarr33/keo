#!/usr/bin/env python3
"""Offline answer-specificity audit: KEO vs length-matched chunk_tok.

Beyond the n=40 preference pack: runs on every judged global item (n≈83)
across generator×judge stacks. Measures whether KEO answers carry more
concrete/identifier mass, and whether Claude overall tracks that signal.

No LLM calls.

  python specificity_audit.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
STATS = ROOT / "evaluation_results" / "stats"
ABL = ROOT / "evaluation_results" / "ablations"
HUMAN = ROOT / "human_validation" / "ratings"
FIGS = ROOT.parents[1] / "papers" / "latex" / "figures"
TEXGEN = ROOT.parents[1] / "papers" / "latex" / "generated"

CR = (
    "global_perspective",
    "theme_identification",
    "synthesis_quality",
    "strategic_value",
    "pattern_recognition",
)

STOP = set(
    """a an the and or but if then of to in on for with from by as is are was were be been being
    this that these those it its their his her our your you we they them at not no so than into
    over under about after before between through during without within can could should would
    may might will shall do does did done have has had having more most less least also just
    only very when where which who whom what how why such other per via etc also however therefore
    thus hence because while although though since until unless whether either neither both each
    every any all some many few several same different similar possible potential likely common
    general overall finally additionally moreover further furthermore including include includes
    based using used use uses example examples result results issue issues problem problems
    system systems component components condition conditions failure failures""".split()
)

AVIATION = set(
    """aircraft airplane helicopter engine propeller landing gear brake hydraulic fuel pump
    valve gasket seal bearing shaft wing fuselage cockpit avionics turbine compressor
    ignition starter generator battery wiring harness connector circuit breaker actuator
    maintenance inspection overhaul leak leakage corrosion fatigue crack fracture
    torque pressure psi rpm ntsb faa ata altimeter gyro servo solenoid manifold""".lower().split()
)


def content_tokens(text: str) -> List[str]:
    return [
        t
        for t in re.findall(r"[A-Za-z][A-Za-z0-9\-/]{2,}", text or "")
        if t.lower() not in STOP
    ]


def specificity_features(text: str) -> Dict[str, float]:
    raw = text or ""
    toks = content_tokens(raw)
    lower = [t.lower() for t in toks]
    n = max(len(toks), 1)
    digits = len(re.findall(r"\d", raw))
    id_like = [
        t
        for t in toks
        if re.search(r"\d", t)
        or "-" in t
        or (t.isupper() and 2 <= len(t) <= 6 and t.isalpha())
    ]
    avi = sum(
        1
        for t in lower
        if t in AVIATION or any(a in t for a in AVIATION if len(a) >= 5)
    )
    types = len(set(lower))
    long_types = len({t for t in lower if len(t) >= 6})
    # Length-robust densities (primary) + soft caps (secondary).
    id_density = len(id_like) / n
    avi_density = avi / n
    long_density = long_types / n
    digit_density = digits / max(len(raw), 1)
    spec_score = float(
        0.40 * id_density
        + 0.25 * min(long_density / 0.55, 1.0)
        + 0.20 * min(avi_density / 0.12, 1.0)
        + 0.15 * min(digit_density / 0.04, 1.0)
    )
    return {
        "n_content": float(len(toks)),
        "n_types": float(types),
        "id_like": float(len(id_like)),
        "id_density": float(id_density),
        "aviation_hits": float(avi),
        "avi_density": float(avi_density),
        "long_types": float(long_types),
        "long_density": float(long_density),
        "digit_chars": float(digits),
        "digit_density": float(digit_density),
        "chars": float(len(raw)),
        "spec_score": spec_score,
    }


def load_answers(arm_dir: Path, field: str) -> Dict[str, str]:
    path = arm_dir / "answers.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out = {}
    for row in rows:
        qid = row.get("question_id")
        blob = row.get(field) or {}
        if qid and isinstance(blob, dict) and blob.get("answer"):
            out[qid] = blob["answer"]
        elif qid and isinstance(blob, str) and blob.strip():
            out[qid] = blob
    return out


def load_judge_overall(arm_dir: Path, method: str) -> Dict[str, float]:
    files = sorted(
        p
        for p in arm_dir.glob("global_sensemaking_evaluation_*.json")
        if "judge_meta" not in p.name
    )
    if not files:
        return {}
    d = json.loads(files[-1].read_text())
    bm = d.get("by_method") or {}
    key = method if method in bm else next(iter(bm), None)
    if not key:
        return {}
    out = {}
    for e in bm[key].get("individual_evaluations") or []:
        if e.get("error"):
            continue
        q = e.get("question_id")
        gm = e.get("global_metrics") or {}
        dims = [float(gm[k]) for k in CR if isinstance(gm.get(k), (int, float))]
        if q and len(dims) == 5:
            out[q] = float(np.mean(dims))
    return out


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0):
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def signflip_p(x: np.ndarray, n_perm: int = 15000, seed: int = 0) -> float:
    if len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(float(x.mean()))
    null = (rng.choice([-1.0, 1.0], size=(n_perm, len(x))) * x).mean(axis=1)
    return float(np.mean(np.abs(null) >= obs))


def paired_block(deltas: np.ndarray, seed: int = 0) -> dict:
    if len(deltas) == 0:
        return {"n": 0}
    return {
        "n": int(len(deltas)),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "ci95": bootstrap_mean_ci(deltas, seed=seed),
        "p_signflip": signflip_p(deltas, seed=seed + 1),
        "frac_positive": float(np.mean(deltas > 1e-12)),
        "frac_negative": float(np.mean(deltas < -1e-12)),
    }


def audit_stack(label: str, run: str, judge_run: Optional[str] = None) -> dict:
    base = ABL / run
    keo_ans = load_answers(base / "mst_on_freq", "keo_answer")
    ct_ans = load_answers(base / "ctrl_chunk_tok", "chunk_tok_answer")
    jbase = ABL / (judge_run or run)
    keo_j = load_judge_overall(jbase / "mst_on_freq", "keo")
    ct_j = load_judge_overall(jbase / "ctrl_chunk_tok", "chunk_tok")

    # Primary analysis: judged global items only (same n as mechanism tables).
    judged_ids = sorted(set(keo_j) & set(ct_j) & set(keo_ans) & set(ct_ans))
    rows = []
    for q in judged_ids:
        fk = specificity_features(keo_ans[q])
        fc = specificity_features(ct_ans[q])
        rows.append(
            {
                "question_id": q,
                "keo_spec": fk["spec_score"],
                "chunk_spec": fc["spec_score"],
                "delta_spec": fk["spec_score"] - fc["spec_score"],
                "delta_id_density": fk["id_density"] - fc["id_density"],
                "delta_avi_density": fk["avi_density"] - fc["avi_density"],
                "delta_long_density": fk["long_density"] - fc["long_density"],
                "delta_digit_density": fk["digit_density"] - fc["digit_density"],
                "delta_id_like_count": fk["id_like"] - fc["id_like"],
                "delta_log_chars": float(
                    np.log(fk["chars"] + 1) - np.log(fc["chars"] + 1)
                ),
                "keo_judge": keo_j[q],
                "chunk_judge": ct_j[q],
                "delta_judge": ct_j[q] - keo_j[q],
            }
        )

    d_spec = np.array([r["delta_spec"] for r in rows])
    d_id = np.array([r["delta_id_density"] for r in rows])
    d_count = np.array([r["delta_id_like_count"] for r in rows])
    dj = np.array([r["delta_judge"] for r in rows])
    dlen = np.array([r["delta_log_chars"] for r in rows])

    out: dict = {
        "label": label,
        "run": run,
        "judge_run": judge_run or run,
        "n_judged": len(rows),
        # Primary paper metric: absolute identifier mass (tracks interim humans).
        "id_count_keo_minus_chunk": paired_block(d_count, seed=10),
        # Robustness: length-divided density (weaker; do not conflate with counts).
        "spec_density_keo_minus_chunk": paired_block(d_spec, seed=11),
        "id_density_keo_minus_chunk": paired_block(d_id, seed=12),
        "component_means": {
            "delta_id_density": float(np.mean([r["delta_id_density"] for r in rows])) if rows else None,
            "delta_avi_density": float(np.mean([r["delta_avi_density"] for r in rows])) if rows else None,
            "delta_long_density": float(np.mean([r["delta_long_density"] for r in rows])) if rows else None,
            "delta_digit_density": float(np.mean([r["delta_digit_density"] for r in rows])) if rows else None,
        },
        "mean_keo_spec": float(np.mean([r["keo_spec"] for r in rows])) if rows else None,
        "mean_chunk_spec": float(np.mean([r["chunk_spec"] for r in rows])) if rows else None,
    }
    if len(rows) >= 3 and d_count.std() > 0 and dj.std() > 0:
        out["corr_delta_id_count_vs_delta_judge"] = float(np.corrcoef(d_count, dj)[0, 1])
        if dlen.std() > 0:
            b_c = np.polyfit(dlen, d_count, 1)
            b_j = np.polyfit(dlen, dj, 1)
            rc = d_count - (b_c[0] * dlen + b_c[1])
            rj = dj - (b_j[0] * dlen + b_j[1])
            out["corr_id_count_vs_judge_length_partial"] = (
                float(np.corrcoef(rc, rj)[0, 1]) if rc.std() > 0 and rj.std() > 0 else None
            )
    else:
        out["corr_delta_id_count_vs_delta_judge"] = None
        out["corr_id_count_vs_judge_length_partial"] = None

    more = d_count > 0
    if more.any():
        out["among_keo_more_id_count"] = {
            "n": int(more.sum()),
            "mean_chunk_minus_keo_judge": float(dj[more].mean()),
            "frac_judge_favors_chunk": float(np.mean(dj[more] > 0)),
            "frac_judge_favors_keo": float(np.mean(dj[more] < 0)),
        }
        out["id_count_judge_sign_disagree_frac"] = float(
            np.mean((d_count > 0) != (dj < 0))
        )
    out["per_question"] = rows
    return out


def human_interim() -> dict:
    path = HUMAN / "human_pref_votes_annotator1.json"
    if not path.exists():
        return {"n_rated": 0}
    votes = json.loads(path.read_text()).get("votes") or {}
    c = Counter(v.get("preference_system") for v in votes.values())
    return {
        "annotator": "1",
        "n_rated": len(votes),
        "n_pack": 40,
        "preference_system_counts": dict(c),
        "keo_frac_of_rated": (c.get("keo") or 0) / max(len(votes), 1),
        "note": "Interim single-annotator lean; not consensus.",
    }


def human_pack_specificity(phi4_rows: List[dict]) -> dict:
    blind = json.loads((HUMAN / "answer_calibration_n40.json").read_text())
    key = {
        it["item_id"]: it
        for it in json.loads((HUMAN / "answer_calibration_n40_key.json").read_text())["items"]
    }
    votes = {}
    vp = HUMAN / "human_pref_votes_annotator1.json"
    if vp.exists():
        for v in (json.loads(vp.read_text()).get("votes") or {}).values():
            votes[v["item_id"]] = v.get("preference_system")
    by_q = {r["question_id"]: r for r in phi4_rows}

    rows = []
    for raw in blind.get("items") or []:
        qid = raw["item_id"]
        k = key[qid]
        left, right = raw.get("answer_left") or "", raw.get("answer_right") or ""
        keo_t, ct_t = (left, right) if k["left_system"] == "keo" else (right, left)
        fk, fc = specificity_features(keo_t), specificity_features(ct_t)
        rows.append(
            {
                "item_id": qid,
                "delta_id_count": fk["id_like"] - fc["id_like"],
                "delta_spec": fk["spec_score"] - fc["spec_score"],
                "human_pref": votes.get(qid),
                "delta_judge_haiku": (by_q[qid]["delta_judge"] if qid in by_q else None),
            }
        )
    dc = np.array([r["delta_id_count"] for r in rows])
    ds = np.array([r["delta_spec"] for r in rows])
    hits_c, hits_s = [], []
    for r in rows:
        if r["human_pref"] in (None, "tie"):
            continue
        hum = r["human_pref"] == "keo"
        hits_c.append((r["delta_id_count"] > 0) == hum)
        hits_s.append((r["delta_spec"] > 0) == hum)
    return {
        "n": len(rows),
        "id_count_keo_minus_chunk": paired_block(dc, seed=41),
        "spec_density_keo_minus_chunk": paired_block(ds, seed=42),
        "keo_more_id_count_frac": float(np.mean(dc > 0)) if len(dc) else None,
        "human_agree_id_count": float(np.mean(hits_c)) if hits_c else None,
        "human_agree_density": float(np.mean(hits_s)) if hits_s else None,
        "n_human_non_tie": len(hits_c),
    }


def write_figure(stacks: List[dict]) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    seen = set()
    forest = []
    for s in stacks:
        if s["run"] in seen:
            continue
        seen.add(s["run"])
        forest.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15))
    ax = axes[0]
    y = np.arange(len(forest))[::-1]
    for i, s in enumerate(forest):
        blk = s["id_count_keo_minus_chunk"]
        lo, hi = blk["ci95"]
        m = blk["mean"]
        col = (
            "#1b7a4e"
            if m > 0 and blk["p_signflip"] < 0.05
            else ("#9b2226" if m < 0 and blk["p_signflip"] < 0.05 else "#6c757d")
        )
        ax.plot([lo, hi], [y[i], y[i]], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(m, y[i], "o", color=col, ms=7)
    ax.axvline(0, color="#222", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([s["label"].split(" / ")[0] for s in forest], fontsize=9)
    ax.set_xlabel(r"$\Delta$ identifier count: KEO $-$ chunk\_tok")
    ax.set_title("Absolute concreteness (judged $n{=}83$)", fontsize=10)
    ax.grid(axis="x", alpha=0.35)

    ax = axes[1]
    phi = next(s for s in stacks if s["label"] == "Phi-4 / Haiku")
    xs = np.array([r["delta_id_like_count"] for r in phi["per_question"]])
    ys = np.array([r["delta_judge"] for r in phi["per_question"]])
    ax.scatter(xs, ys, s=18, alpha=0.55, c="#4a4e69", edgecolors="none")
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.axvline(0, color="#888", lw=1, ls="--")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.fill_between(
        [0, max(xlim[1], 1)], 0, max(ylim[1], 0.1), color="#f4a261", alpha=0.14, zorder=0
    )
    ax.set_xlabel(r"$\Delta$ id count (KEO more concrete $\rightarrow$)")
    ax.set_ylabel(r"$\Delta_{\mathrm{judge}}$ (chunks higher $\rightarrow$)")
    ax.set_title("Phi-4 / Haiku mismatch zone", fontsize=10)
    disagree = phi.get("id_count_judge_sign_disagree_frac")
    if disagree is not None:
        ax.text(
            0.02,
            0.98,
            f"sign disagree {100 * disagree:.0f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#6c757d",
        )
    fig.tight_layout()
    fig.savefig(FIGS / "specificity_audit.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "specificity_audit.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    TEXGEN.mkdir(parents=True, exist_ok=True)

    stacks = [
        audit_stack("Phi-4 / Haiku", "phi4"),
        audit_stack("Phi-4 / Sonnet", "phi4", judge_run="phi4_sonnet_all"),
        audit_stack("Gemma E4B / Haiku", "_gemma4e4b"),
        audit_stack("Gemma E4B / Sonnet", "_gemma4e4b", judge_run="gemma4e4b_sonnet_all"),
        audit_stack("Gemma 31B / Haiku", "openrouter_gemma4_31b"),
        audit_stack(
            "Gemma 31B / Sonnet",
            "openrouter_gemma4_31b",
            judge_run="openrouter_gemma4_31b_sonnet",
        ),
    ]
    write_figure(stacks)
    pack = human_pack_specificity(stacks[0]["per_question"])
    human = human_interim()
    summary = [{k: v for k, v in s.items() if k != "per_question"} for s in stacks]

    blob = {
        "scope": "Judged global n≈83; primary=absolute id count; density=robustness only.",
        "human_interim_ann1": human,
        "calibration_pack": pack,
        "stacks": summary,
        "phi4_haiku_per_question": stacks[0]["per_question"],
        "paper_impact": (
            "Moderate supporting audit for human–judge divergence. "
            "Does not greatly change headline Graph-RAG claims without finished humans."
        ),
    }
    (STATS / "specificity_audit.json").write_text(json.dumps(blob, indent=2))

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Stack & $n$ & Mean $\Delta$ id count & 95\% CI & KEO more ids & Judge$\uparrow$chunk $|$ KEO more ids \\",
        r"\midrule",
    ]
    for s in summary:
        b = s["id_count_keo_minus_chunk"]
        among = s.get("among_keo_more_id_count") or {}
        jfrac = among.get("frac_judge_favors_chunk")
        jfrac_s = f"{100 * jfrac:.0f}\\%" if jfrac is not None else "---"
        lo, hi = b["ci95"]
        lines.append(
            f"{s['label']} & {b['n']} & ${b['mean']:+.2f}$ & "
            f"$[{lo:+.2f},{hi:+.2f}]$ & {100 * b['frac_positive']:.0f}\\% & {jfrac_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TEXGEN / "tab_specificity_audit.tex").write_text("\n".join(lines) + "\n")

    print(f"Wrote {STATS / 'specificity_audit.json'}")
    print(f"Wrote {FIGS / 'specificity_audit.pdf'}")
    for s in summary:
        b = s["id_count_keo_minus_chunk"]
        print(
            f"  {s['label']:26} Δid={b['mean']:+.2f} "
            f"KEO_more={b['frac_positive']:.0%} p={b['p_signflip']:.3f}"
        )


if __name__ == "__main__":
    main()
