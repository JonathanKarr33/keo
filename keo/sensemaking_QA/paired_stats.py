#!/usr/bin/env python3
"""
Paired significance tests for KEO method arms (papers/ACL_STRATEGY.md §4.2).

Judge scores are matched per question, so arms must be compared with paired tests
rather than by eyeballing two means. For every arm-vs-reference pair this reports:

  * mean paired difference with a bootstrap 95% CI
  * two-sided sign-flip permutation p-value, Holm-corrected within each metric
  * paired effect size (Cohen's d_z) and Cliff's delta
  * win / tie / loss counts
  * the minimum difference the sample could detect (MDE at 80% power)

Differences that are not significant are labelled "n.s.", and comparisons whose
MDE exceeds the observed difference are labelled "underpowered" — with n=133 many
plausible effects are simply undetectable, and saying so is part of the result.

Usage:
  # Mechanism controls against the keo arm
  python paired_stats.py \
    --arm keo=evaluation_results/ablations/mst_on_freq \
    --arm chunk_tok=evaluation_results/ablations/ctrl_chunk_tok \
    --arm chunk_sum=evaluation_results/ablations/ctrl_chunk_sum \
    --arm keo_shuf=evaluation_results/ablations/ctrl_keo_shuf_s13 \
    --reference keo \
    --out evaluation_results/stats/controls.json \
    --latex evaluation_results/stats/controls.tex

  # Every arm inside one judged run
  python paired_stats.py --all-from evaluation_results/paper_min --reference keo
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CRITERIA = (
    "global_perspective",
    "theme_identification",
    "synthesis_quality",
    "strategic_value",
    "pattern_recognition",
)
OVERALL = "overall"
METRICS = CRITERIA + (OVERALL,)

EVAL_GLOB = "global_sensemaking_evaluation_*.json"
JUDGE_META_SUFFIX = "_judge_meta.json"

# Preference order when a judged file holds several arms and none matches the label.
# "graphrag" is last because keo runs duplicate their answers into it for back-compat.
_ARM_PRIORITY = (
    "keo", "risk", "keo_shuf", "keo_norel", "routed", "hybrid", "community",
    "chunk_tok", "chunk_sum", "chunk_dedup", "textchunkrag", "vanilla", "graphrag",
)

# 2025 runs wrote flat "<name>_global" keys; map them onto current arm names
_LEGACY_ARM_NAMES = {"textchunk": "textchunkrag"}

# Scores per question, per metric: {arm: {question_id: {metric: score}}}
ArmScores = Dict[str, Dict[str, float]]


def _z(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation; avoids a scipy dep)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _newest_eval_file(path: str) -> str:
    """Resolve a directory to its most recent global-sensemaking evaluation file."""
    if os.path.isfile(path):
        return path
    matches = sorted(glob.glob(os.path.join(path, EVAL_GLOB)))
    # The judge writes a `..._judge_meta.json` sidecar next to each evaluation. It matches the glob
    # and sorts *after* the file it describes, so taking the last match picks the sidecar — which
    # holds no arms and fails with a message about unrecognisable arms rather than the real cause.
    matches = [m for m in matches if not m.endswith(JUDGE_META_SUFFIX)]
    if not matches:
        raise FileNotFoundError(f"No {EVAL_GLOB} under {path} — has this arm been judged?")
    return matches[-1]


def _arms_in_blob(blob: Dict) -> Dict[str, Dict]:
    """
    Arms in a judged file, from either layout.

    Current runs nest them under by_method; the 2025 runs used flat keys like
    ``graphrag_global``. Reading both lets the archived judgments be re-analysed
    without paying to re-judge them.
    """
    by_method = blob.get("by_method")
    if by_method:
        return dict(by_method)
    legacy = {}
    for key, value in blob.items():
        if not key.endswith("_global") or not isinstance(value, dict):
            continue
        if "individual_evaluations" not in value:
            continue
        name = key[: -len("_global")]
        legacy[_LEGACY_ARM_NAMES.get(name, name)] = value
    return legacy


def _per_question_scores(method_result: Dict) -> ArmScores:
    """Pull per-question criterion scores out of one arm's evaluation result."""
    scores: ArmScores = {}
    for item in method_result.get("individual_evaluations") or []:
        if not isinstance(item, dict) or item.get("error"):
            continue
        qid = item.get("question_id")
        metrics = item.get("global_metrics") or {}
        if not qid or not metrics:
            continue
        row = {c: float(metrics[c]) for c in CRITERIA if isinstance(metrics.get(c), (int, float))}
        if len(row) < len(CRITERIA):
            continue  # partial parse; excluding it beats silently averaging over missing criteria
        row[OVERALL] = sum(row[c] for c in CRITERIA) / len(CRITERIA)
        scores[qid] = row
    return scores


def load_arms(
    arm_specs: Sequence[str],
    all_from: Sequence[str] = (),
) -> Tuple[Dict[str, ArmScores], Dict[str, str]]:
    """Load per-question scores for each arm → ({arm: scores}, {arm: source_file})."""
    arms: Dict[str, ArmScores] = {}
    sources: Dict[str, str] = {}

    for path in all_from:
        eval_file = _newest_eval_file(path)
        with open(eval_file) as f:
            blob = json.load(f)
        by_method = _arms_in_blob(blob)
        if not by_method:
            raise ValueError(f"{eval_file} contains no recognisable method arms")
        for name, result in by_method.items():
            scores = _per_question_scores(result)
            if scores:
                arms[name] = scores
                sources[name] = eval_file

    for spec in arm_specs:
        if "=" not in spec:
            raise ValueError(f"--arm expects LABEL=PATH[#ARM_KEY], got {spec!r}")
        label, path = spec.split("=", 1)
        arm_key = None
        if "#" in path:
            path, arm_key = path.rsplit("#", 1)
        eval_file = _newest_eval_file(path)
        with open(eval_file) as f:
            blob = json.load(f)
        by_method = _arms_in_blob(blob)
        if not by_method:
            raise ValueError(f"{eval_file} contains no recognisable method arms")
        if arm_key:
            if arm_key not in by_method:
                raise ValueError(
                    f"{eval_file} has no arm {arm_key!r} (found {sorted(by_method)})"
                )
            chosen = by_method[arm_key]
        elif label in by_method:
            chosen = by_method[label]
        elif len(by_method) == 1:
            chosen = next(iter(by_method.values()))
        else:
            # A keo run also writes a duplicate graphrag arm, so prefer the
            # canonical name over the back-compat alias instead of giving up.
            picked = next((k for k in _ARM_PRIORITY if k in by_method), None)
            if not picked:
                raise ValueError(
                    f"{eval_file} holds several arms ({sorted(by_method)}); "
                    f"disambiguate with {label}=PATH#ARM_KEY"
                )
            print(f"  note: {label} → picked arm {picked!r} from {sorted(by_method)}")
            chosen = by_method[picked]
        scores = _per_question_scores(chosen)
        if not scores:
            raise ValueError(f"No per-question scores parsed for {label} from {eval_file}")
        arms[label] = scores
        sources[label] = eval_file

    return arms, sources


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(b > a) - P(a > b); robust to the ordinal 1-5 judge scale."""
    greater = int((b[:, None] > a[None, :]).sum())
    less = int((b[:, None] < a[None, :]).sum())
    total = a.size * b.size
    return (greater - less) / total if total else 0.0


def paired_compare(
    ref: np.ndarray,
    arm: np.ndarray,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    power: float = 0.80,
    seed: int = 1234,
) -> Dict:
    """Paired comparison of one arm against the reference on a single metric."""
    diff = arm - ref
    n = diff.size
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    rng = np.random.default_rng(seed)

    wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    losses = int((diff < 0).sum())

    # Untestable: a single question, or two arms that agree on every question.
    #
    # A *constant non-zero* difference is emphatically not untestable, and treating it as
    # such was a bug: if every question moves by exactly +1, the sign-flip test is exact
    # and gives p = 2/2^n — the strongest evidence the design can produce. On a coarse
    # integer judge scale with few questions, constant differences are entirely plausible,
    # so this case has to run the test.
    if n < 2 or (sd == 0.0 and mean_diff == 0.0):
        return {
            "n_pairs": int(n),
            "mean_reference": float(ref.mean()) if n else 0.0,
            "mean_arm": float(arm.mean()) if n else 0.0,
            "mean_diff": mean_diff,
            "sd_diff": sd,
            "ci95": [mean_diff, mean_diff],
            "p_value": 1.0,
            "cohens_dz": 0.0,
            "cliffs_delta": 0.0,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mde_80": float("inf"),
            "underpowered": True,
            "degenerate": True,
            "constant_difference": False,
        }

    # Percentile bootstrap CI on the mean paired difference
    boot = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided sign-flip permutation test: under H0 the sign of each paired
    # difference is exchangeable, which is the right null for matched judging.
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    null_means = (signs * diff).mean(axis=1)
    p_value = float((1 + int((np.abs(null_means) >= abs(mean_diff) - 1e-12).sum())) / (n_perm + 1))

    if sd > 0.0:
        dz = mean_diff / sd
        mde = (_z(1 - alpha / 2) + _z(power)) * sd / math.sqrt(n)
        underpowered = bool(abs(mean_diff) < mde)
    else:
        # Every pair moved by the same non-zero amount: the permutation test above is
        # valid, but a standardised effect size divides by zero and there is no
        # within-pair variance for an MDE to be defined against.
        dz = math.inf if mean_diff > 0 else -math.inf
        mde = 0.0
        underpowered = False

    return {
        "n_pairs": int(n),
        "mean_reference": float(ref.mean()),
        "mean_arm": float(arm.mean()),
        "mean_diff": mean_diff,
        "sd_diff": sd,
        "ci95": [float(lo), float(hi)],
        "p_value": p_value,
        "cohens_dz": float(dz),
        "cliffs_delta": _cliffs_delta(ref, arm),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mde_80": float(mde),
        "underpowered": underpowered,
        "degenerate": False,
        "constant_difference": sd == 0.0,
    }


def holm(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment (monotone, family-wise error ≤ alpha)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def verdict(result: Dict, alpha: float = 0.05) -> str:
    """One-word reading of a comparison, for humans and for table cells."""
    if result.get("degenerate"):
        return "n/a"
    if result["p_holm"] < alpha:
        return "higher" if result["mean_diff"] > 0 else "lower"
    return "underpowered n.s." if result["underpowered"] else "n.s."


def compare_all(
    arms: Dict[str, ArmScores],
    reference: str,
    metrics: Sequence[str] = METRICS,
    alpha: float = 0.05,
    **kwargs,
) -> Dict:
    """Every arm against the reference, Holm-corrected within each metric."""
    if reference not in arms:
        raise ValueError(f"Reference arm {reference!r} not loaded (have {sorted(arms)})")

    others = [a for a in arms if a != reference]
    if not others:
        raise ValueError("Need at least one arm besides the reference")

    # Restrict to questions every arm answered, so all comparisons use one sample
    common = set(arms[reference])
    for arm in others:
        common &= set(arms[arm])
    qids = sorted(common)
    if not qids:
        raise ValueError("No question IDs shared across all arms")

    out: Dict = {
        "reference": reference,
        "arms": sorted(arms),
        "n_common_questions": len(qids),
        "question_ids": qids,
        "alpha": alpha,
        "metrics": {},
    }

    for metric in metrics:
        ref_vec = np.array([arms[reference][q][metric] for q in qids], dtype=float)
        per_arm = {
            arm: paired_compare(
                ref_vec,
                np.array([arms[arm][q][metric] for q in qids], dtype=float),
                alpha=alpha,
                **kwargs,
            )
            for arm in others
        }
        # Correct only over tests actually performed. A degenerate comparison was never
        # tested, so counting it inflates the family size and costs the real comparisons
        # power — with several identical arms it can push a true effect out of
        # significance.
        tested = {arm: r["p_value"] for arm, r in per_arm.items() if not r["degenerate"]}
        adjusted = holm(tested)
        for arm, result in per_arm.items():
            result["p_holm"] = adjusted.get(arm, 1.0)
            result["family_size"] = len(tested)
            result["verdict"] = verdict(result, alpha)
        out["metrics"][metric] = {
            "mean_reference": float(ref_vec.mean()),
            "family_size": len(tested),
            "comparisons": per_arm,
        }

    return out


def format_text(results: Dict, metric: str = OVERALL) -> str:
    """Readable summary for one metric."""
    block = results["metrics"][metric]
    ref = results["reference"]
    width = max([len(a) for a in block["comparisons"]] + [len("arm")]) + 2
    lines = [
        f"Metric: {metric}   reference: {ref} (mean {block['mean_reference']:.3f}), "
        f"n={results['n_common_questions']} paired questions",
        f"{'arm':<{width}}{'mean':>7}{'Δ':>8}{'95% CI':>18}{'p_holm':>9}{'d_z':>7}{'MDE':>7}  "
        f"{'W/T/L':>10}  verdict",
    ]
    for arm, r in sorted(block["comparisons"].items(), key=lambda kv: -kv[1]["mean_diff"]):
        ci = f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        mde = "  inf" if math.isinf(r["mde_80"]) else f"{r['mde_80']:.3f}"
        dz = "   inf" if math.isinf(r["cohens_dz"]) else f"{r['cohens_dz']:+.2f}"
        lines.append(
            f"{arm:<{width}}{r['mean_arm']:>7.3f}{r['mean_diff']:>+8.3f}{ci:>18}"
            f"{r['p_holm']:>9.4f}{dz:>7}{mde:>7}  {wtl:>10}  {r['verdict']}"
        )
    return "\n".join(lines)


def format_latex(results: Dict, metric: str = OVERALL, caption: Optional[str] = None) -> str:
    """Booktabs table; significance and power stated in-cell, not implied by bolding."""
    block = results["metrics"][metric]
    ref = results["reference"]
    n = results["n_common_questions"]
    cap = caption or (
        f"Paired comparison against {_tex(ref)} on {_tex(metric)} "
        f"($n={n}$ questions, judge scores 1--5). "
        "$\\Delta$ is the mean paired difference with a percentile bootstrap 95\\% CI; "
        "$p$ is a two-sided sign-flip permutation test, Holm-corrected within the metric; "
        "MDE is the smallest difference detectable at 80\\% power."
    )
    rows = []
    for arm, r in sorted(block["comparisons"].items(), key=lambda kv: -kv[1]["mean_diff"]):
        mde = "--" if math.isinf(r["mde_80"]) else f"{r['mde_80']:.3f}"
        # The permutation test floors at 1/(B+1), so report a bound rather than 0.000
        p = "$<$0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"
        # A constant paired difference gives d_z = ±inf; "$+inf$" would typeset as italic
        # prose in the table.
        dz = (
            ("$+\\infty$" if r["cohens_dz"] > 0 else "$-\\infty$")
            if math.isinf(r["cohens_dz"])
            else f"${r['cohens_dz']:+.2f}$"
        )
        rows.append(
            f"{_tex(arm)} & {r['mean_arm']:.3f} & ${r['mean_diff']:+.3f}$ & "
            f"$[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]$ & {p} & "
            f"{dz} & {mde} & {_tex(r['verdict'])} \\\\"
        )
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrl}",
        "\\toprule",
        "Arm & Mean & $\\Delta$ & 95\\% CI & $p_{\\text{Holm}}$ & $d_z$ & MDE & Reading \\\\",
        "\\midrule",
        f"{_tex(ref)} & {block['mean_reference']:.3f} & -- & -- & -- & -- & -- & reference \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{cap}}}",
        f"\\label{{tab:paired-{metric.replace('_', '-')}}}",
        "\\end{table}",
    ])


def _tex(text: str) -> str:
    return str(text).replace("_", "\\_")


def main():
    parser = argparse.ArgumentParser(description="Paired significance tests across KEO arms")
    parser.add_argument(
        "--arm", action="append", default=[], metavar="LABEL=PATH",
        help="Arm label and its judged run (directory or evaluation JSON); repeatable",
    )
    parser.add_argument(
        "--all-from", action="append", default=[], metavar="PATH",
        help="Load every arm found in this judged run; repeatable",
    )
    parser.add_argument("--reference", required=True, help="Arm every other arm is compared against")
    parser.add_argument("--metric", default=OVERALL, choices=list(METRICS), help="Metric to print")
    parser.add_argument("--out", default=None, help="Write full results JSON here")
    parser.add_argument("--latex", default=None, help="Write a LaTeX table for --metric here")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80, help="Target power for the MDE")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not args.arm and not args.all_from:
        parser.error("give at least one --arm or --all-from")

    arms, sources = load_arms(args.arm, args.all_from)
    print(f"Loaded {len(arms)} arms:")
    for name in sorted(arms):
        print(f"  {name:<14} n={len(arms[name]):<4} ← {sources[name]}")

    results = compare_all(
        arms,
        reference=args.reference,
        alpha=args.alpha,
        power=args.power,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    results["sources"] = sources

    print()
    print(format_text(results, args.metric))

    flagged = [
        arm for arm, r in results["metrics"][args.metric]["comparisons"].items()
        if r["verdict"] == "underpowered n.s."
    ]
    if flagged:
        print(
            f"\nUnderpowered at n={results['n_common_questions']}: {', '.join(sorted(flagged))}. "
            "Report these as undetectable, not as equivalent."
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Stats → {args.out}")

    if args.latex:
        os.makedirs(os.path.dirname(args.latex) or ".", exist_ok=True)
        with open(args.latex, "w") as f:
            f.write(format_latex(results, args.metric) + "\n")
        print(f"✓ LaTeX table → {args.latex}")


if __name__ == "__main__":
    main()
