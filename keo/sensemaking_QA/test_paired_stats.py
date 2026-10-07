#!/usr/bin/env python3
"""
Offline checks for paired_stats.py (no API calls).

Uses synthetic judge scores where the right answer is known: a real effect must be
detected, pure noise must not be, and a small effect on a small sample must be
reported as underpowered rather than as "no difference".

Run: python test_paired_stats.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_harness import run_suite  # noqa: E402
from paired_stats import (  # noqa: E402
    CRITERIA,
    OVERALL,
    compare_all,
    format_latex,
    format_text,
    holm,
    load_arms,
    paired_compare,
    _z,
)


def synth_scores(n: int, base: float, shift: float = 0.0, seed: int = 0):
    """Per-question criterion scores on the judge's 1-5 scale."""
    rng = np.random.default_rng(seed)
    scores = {}
    for i in range(n):
        qid = f"q{i:03d}"
        row = {}
        for c in CRITERIA:
            val = base + shift + rng.normal(0, 0.5)
            row[c] = float(np.clip(round(val * 2) / 2, 1.0, 5.0))
        row[OVERALL] = sum(row[c] for c in CRITERIA) / len(CRITERIA)
        scores[qid] = row
    return scores


def test_normal_quantile():
    assert abs(_z(0.975) - 1.959964) < 1e-4
    assert abs(_z(0.80) - 0.841621) < 1e-4
    assert abs(_z(0.5)) < 1e-9
    assert _z(0.025) < 0 < _z(0.975)
    print(f"  z(0.975)={_z(0.975):.6f}  z(0.80)={_z(0.80):.6f}")


def test_detects_real_effect():
    n = 133
    ref = np.array([3.0] * n)
    arm = ref + 0.5  # a clear, consistent improvement
    r = paired_compare(ref, arm + np.random.default_rng(1).normal(0, 0.2, n), n_boot=2000, n_perm=2000)
    assert r["p_value"] < 0.01, r["p_value"]
    assert r["mean_diff"] > 0.4
    assert r["ci95"][0] > 0, "CI should exclude zero"
    assert not r["underpowered"]
    assert r["wins"] > r["losses"]
    print(f"  real effect: Δ={r['mean_diff']:+.3f} p={r['p_value']:.4f} d_z={r['cohens_dz']:+.2f}")


def test_rejects_noise():
    rng = np.random.default_rng(7)
    n = 133
    ref = rng.normal(3.0, 0.5, n)
    arm = ref + rng.normal(0.0, 0.5, n)  # no true shift
    r = paired_compare(ref, arm, n_boot=2000, n_perm=2000)
    assert r["p_value"] > 0.05, r["p_value"]
    assert r["ci95"][0] < 0 < r["ci95"][1], "CI should straddle zero"
    print(f"  noise: Δ={r['mean_diff']:+.3f} p={r['p_value']:.3f} CI={r['ci95']}")


def test_flags_underpowered():
    rng = np.random.default_rng(11)
    n = 12
    ref = rng.normal(3.0, 0.6, n)
    arm = ref + 0.1 + rng.normal(0, 0.6, n)  # small effect, tiny sample
    r = paired_compare(ref, arm, n_boot=2000, n_perm=2000)
    assert r["underpowered"], "small effect on n=12 must be flagged"
    assert r["mde_80"] > abs(r["mean_diff"])
    print(f"  underpowered: Δ={r['mean_diff']:+.3f} MDE={r['mde_80']:.3f} (n={n})")

    # Larger n shrinks the MDE
    big = paired_compare(
        rng.normal(3.0, 0.6, 400),
        rng.normal(3.0, 0.6, 400),
        n_boot=500, n_perm=500,
    )
    assert big["mde_80"] < r["mde_80"], "MDE must shrink with n"
    print(f"  MDE {r['mde_80']:.3f} (n=12) → {big['mde_80']:.3f} (n=400)")


def test_degenerate_pairs():
    ref = np.array([3.0, 3.0, 3.0])
    r = paired_compare(ref, ref.copy(), n_boot=100, n_perm=100)
    assert r["degenerate"] and r["p_value"] == 1.0 and r["underpowered"]
    single = paired_compare(np.array([3.0]), np.array([4.0]), n_boot=100, n_perm=100)
    assert single["degenerate"], "a single question cannot support a test"
    print("  degenerate cases return p=1 instead of crashing")


def test_constant_difference_is_tested_not_dismissed():
    """
    A constant non-zero paired difference is the strongest evidence the design can give.

    Regression: sd==0 was lumped in with "identical arms" and returned p=1.0, so an arm
    that beat the reference on every single question was reported as no result. Coarse
    integer judge scores make this case realistic, not hypothetical.
    """
    n = 30
    ref = np.array([3.0] * n)
    arm = ref + 1.0  # every question improves by exactly one point

    r = paired_compare(ref, arm, n_boot=500, n_perm=2000)

    assert not r["degenerate"], "a constant non-zero difference is testable"
    assert r["constant_difference"], r
    assert r["p_value"] < 0.01, f"unanimous improvement should be significant: {r['p_value']}"
    assert r["wins"] == n and r["losses"] == 0, r
    assert math.isinf(r["cohens_dz"]) and r["cohens_dz"] > 0, r["cohens_dz"]
    assert not r["underpowered"], r
    assert r["ci95"] == [1.0, 1.0], r["ci95"]
    print(f"  constant Δ=+1 on n={n}: p={r['p_value']:.4f}, d_z=+inf (was p=1.0)")

    # The sign is respected, and truly identical arms remain degenerate
    worse = paired_compare(ref, ref - 1.0, n_boot=500, n_perm=2000)
    assert worse["p_value"] < 0.01 and worse["cohens_dz"] < 0, worse
    same = paired_compare(ref, ref.copy(), n_boot=100, n_perm=100)
    assert same["degenerate"] and not same["constant_difference"], same
    print("  sign preserved; identical arms still degenerate")


def test_small_n_constant_difference_is_not_oversold():
    """With n=3 the exact sign-flip p is 2/8, so it must not read as significant."""
    ref = np.array([3.0, 3.0, 3.0])
    r = paired_compare(ref, ref + 1.0, n_boot=200, n_perm=20000)

    assert not r["degenerate"], r
    assert 0.15 < r["p_value"] < 0.35, f"expected ≈0.25 for n=3, got {r['p_value']}"
    print(f"  n=3 constant Δ: p={r['p_value']:.3f} (exact 2/8 = 0.250)")


def test_untested_arms_do_not_inflate_the_family():
    """Degenerate comparisons were never tested, so Holm must not count them."""
    ref = synth_scores(40, 3.4, seed=5)
    arms = {
        "ref": ref,
        "real": {q: {k: v + 0.6 for k, v in row.items()} for q, row in ref.items()},
        "clone_a": {q: dict(row) for q, row in ref.items()},
        "clone_b": {q: dict(row) for q, row in ref.items()},
        "clone_c": {q: dict(row) for q, row in ref.items()},
    }

    out = compare_all(arms, reference="ref", metrics=(OVERALL,), n_boot=400, n_perm=2000)
    block = out["metrics"][OVERALL]

    assert block["family_size"] == 1, f"only one real test was performed: {block['family_size']}"
    real = block["comparisons"]["real"]
    assert real["p_holm"] < 0.01, real["p_holm"]
    for clone in ("clone_a", "clone_b", "clone_c"):
        assert block["comparisons"][clone]["degenerate"], clone
        assert block["comparisons"][clone]["verdict"] == "n/a", clone
    print(f"  family_size=1 despite 4 arms; real effect keeps p_holm={real['p_holm']:.4f}")


def test_holm_correction():
    raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
    adj = holm(raw)
    assert abs(adj["a"] - 0.004) < 1e-9, adj["a"]
    assert abs(adj["b"] - 0.06) < 1e-9, adj["b"]
    assert all(adj[k] >= raw[k] for k in raw), "adjusted p must never decrease"
    ordered = sorted(raw, key=lambda k: raw[k])
    assert all(adj[ordered[i]] <= adj[ordered[i + 1]] + 1e-12 for i in range(len(ordered) - 1)), \
        "Holm must stay monotone"
    # b survives uncorrected but not corrected: the whole point of the correction
    assert raw["b"] < 0.05 < adj["b"]
    print(f"  holm: {raw} → { {k: round(v, 4) for k, v in adj.items()} }")


def _write_eval(dirpath: str, arm_scores: dict, stamp: str = "20260919_120000"):
    """Write a judged-run file shaped like run_evaluation.py's output."""
    os.makedirs(dirpath, exist_ok=True)
    by_method = {}
    for arm, scores in arm_scores.items():
        by_method[arm] = {
            "global_questions_evaluated": len(scores),
            "global_metrics": {
                c: sum(s[c] for s in scores.values()) / len(scores) for c in CRITERIA
            },
            "individual_evaluations": [
                {"question_id": qid, "global_metrics": {c: row[c] for c in CRITERIA}}
                for qid, row in scores.items()
            ],
        }
    path = os.path.join(dirpath, f"global_sensemaking_evaluation_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"timestamp": stamp, "by_method": by_method}, f)
    return path


def test_end_to_end_on_judged_files():
    """Load real-shaped judged output, compare arms, render both outputs."""
    with tempfile.TemporaryDirectory() as td:
        keo_dir = os.path.join(td, "mst_on_freq")
        tok_dir = os.path.join(td, "ctrl_chunk_tok")
        shuf_dir = os.path.join(td, "ctrl_keo_shuf_s13")
        _write_eval(keo_dir, {"keo": synth_scores(60, 3.6, seed=1)})
        _write_eval(tok_dir, {"chunk_tok": synth_scores(60, 3.6, shift=-0.6, seed=2)})
        _write_eval(shuf_dir, {"keo_shuf": synth_scores(60, 3.6, shift=0.0, seed=3)})

        arms, sources = load_arms(
            [f"keo={keo_dir}", f"chunk_tok={tok_dir}", f"keo_shuf={shuf_dir}"]
        )
        assert set(arms) == {"keo", "chunk_tok", "keo_shuf"}
        assert all(p.endswith(".json") for p in sources.values())

        results = compare_all(arms, reference="keo", n_boot=2000, n_perm=2000)
        assert results["n_common_questions"] == 60
        comps = results["metrics"][OVERALL]["comparisons"]

        # chunk_tok was built 0.6 lower: must be detected as lower
        assert comps["chunk_tok"]["verdict"] == "lower", comps["chunk_tok"]
        # keo_shuf was built identical: must NOT be called different
        assert comps["keo_shuf"]["verdict"].endswith("n.s."), comps["keo_shuf"]
        # every criterion is scored, not just the overall
        assert set(results["metrics"]) == set(CRITERIA) | {OVERALL}

        text = format_text(results, OVERALL)
        assert "verdict" in text and "chunk_tok" in text
        latex = format_latex(results, OVERALL)
        assert "\\begin{tabular}" in latex and "chunk\\_tok" in latex
        assert latex.count("\\\\") >= 4
        print("  end-to-end: chunk_tok → lower, keo_shuf → n.s., LaTeX + text rendered")


def test_reads_legacy_layout():
    """The 2025 runs used flat *_global keys; they must load without re-judging."""
    with tempfile.TemporaryDirectory() as td:
        run = os.path.join(td, "legacy_run")
        os.makedirs(run)
        scores = {
            "vanilla": synth_scores(30, 3.2, seed=1),
            "textchunk": synth_scores(30, 3.4, seed=2),
            "graphrag": synth_scores(30, 3.9, seed=3),
        }
        blob = {"timestamp": "20250717_125342"}
        for arm, s in scores.items():
            blob[f"{arm}_global"] = {
                "individual_evaluations": [
                    {"question_id": qid, "global_metrics": {c: row[c] for c in CRITERIA}}
                    for qid, row in s.items()
                ]
            }
        with open(os.path.join(run, "global_sensemaking_evaluation_20250717_125342.json"), "w") as f:
            json.dump(blob, f)

        arms, _ = load_arms([], all_from=[run])
        # "textchunk" is renamed to the current arm name so comparisons line up
        assert set(arms) == {"vanilla", "textchunkrag", "graphrag"}, sorted(arms)
        assert len(arms["graphrag"]) == 30
        results = compare_all(arms, reference="graphrag", n_boot=1000, n_perm=1000)
        assert results["n_common_questions"] == 30
        assert set(results["metrics"][OVERALL]["comparisons"]) == {"vanilla", "textchunkrag"}
        print("  legacy *_global layout loads; textchunk → textchunkrag")


def test_missing_judgement_is_loud():
    with tempfile.TemporaryDirectory() as td:
        empty = os.path.join(td, "never_judged")
        os.makedirs(empty)
        try:
            load_arms([f"keo={empty}"])
        except FileNotFoundError as e:
            assert "judged" in str(e)
            print("  unjudged arm raises instead of reporting an empty comparison")
            return
        raise AssertionError("expected FileNotFoundError for an unjudged arm")


def main():
    run_suite("paired stats", [
        ("normal quantile", test_normal_quantile),
        ("detects a real effect", test_detects_real_effect),
        ("rejects noise", test_rejects_noise),
        ("flags underpowered comparisons", test_flags_underpowered),
        ("degenerate pairs", test_degenerate_pairs),
        ("constant difference is tested", test_constant_difference_is_tested_not_dismissed),
        ("small-n constant not oversold", test_small_n_constant_difference_is_not_oversold),
        ("untested arms excluded from family", test_untested_arms_do_not_inflate_the_family),
        ("holm correction", test_holm_correction),
        ("end to end on judged files", test_end_to_end_on_judged_files),
        ("legacy evaluation layout", test_reads_legacy_layout),
        ("missing judgement", test_missing_judgement_is_loud),
    ])


if __name__ == "__main__":
    main()
