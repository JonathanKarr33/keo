#!/usr/bin/env python3
"""
Is the graph advantage a judge length preference? (papers/ACL_STRATEGY.md §3 Pivot B)

Runs entirely on judgments and answers already on disk — no API calls, no re-judging.

For every run that has both an answers file and a judged evaluation file this reports:

  1. answer length per arm (the graph arm is typically several times longer)
  2. how strongly judge score tracks answer length, within question (Spearman / Pearson)
  3. how often the longer of two answers to the same question wins
  4. the arm gap before and after adjusting for log length, with question fixed effects
  5. the same gap on the length-matched subsample, with a permutation test
  6. a collinearity diagnostic: if arm and length barely overlap, no regression can
     separate them, and only a length-matched *generation* arm (chunk_tok) can

Point 6 is the likely outcome and is itself the result: it is the argument for the
C-tok control rather than a statistical adjustment after the fact.

Usage:
  python judge_length_bias.py --auto --out evaluation_results/stats/length_bias.json
  python judge_length_bias.py \
    --answers output/answers_gemma-3-27b-it_kg_gs100.json \
    --eval evaluation_results/answer_gemma-3-27b-it_evaluator_gpt-4o_kg_gs100
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from paired_stats import (
    CRITERIA,
    OVERALL,
    _arms_in_blob,
    _newest_eval_file,
    _per_question_scores,
    paired_compare,
)

# Judged arm name → the answers-file field holding that arm's text
ARM_TO_FIELD = {
    "vanilla": "vanilla_answer",
    "textchunkrag": "textchunkrag_answer",
    "graphrag": "graphrag_answer",
    "keo": "keo_answer",
    "community": "community_answer",
    "hybrid": "hybrid_answer",
    "routed": "routed_answer",
    "risk": "risk_answer",
    "chunk_tok": "chunk_tok_answer",
    "chunk_sum": "chunk_sum_answer",
    "chunk_dedup": "chunk_dedup_answer",
    "keo_shuf": "keo_shuf_answer",
    "keo_norel": "keo_norel_answer",
}

# Arms compared head to head; the graph arm is second so a positive gap favours graph
DEFAULT_PAIR = ("textchunkrag", "graphrag")

# Ablation dirs store one method each; map eval method name → answers field
METHOD_TO_FIELD = dict(ARM_TO_FIELD)
METHOD_TO_FIELD["textchunk"] = "textchunkrag_answer"  # dir name alias


def _norm(text: str) -> str:
    """Normalise a model/KG tag so directory and filename spellings match."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def discover_pairs(
    results_root: str = "evaluation_results",
    answers_root: str = "output",
) -> List[Tuple[str, str]]:
    """Find (answers_file, eval_dir) pairs by matching model and KG tags in their names."""
    answers_by_key: Dict[str, str] = {}
    for path in glob.glob(os.path.join(answers_root, "**", "answers_*.json"), recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0][len("answers_"):]
        if "_kg_" not in stem:
            continue
        model, kg = stem.split("_kg_", 1)
        answers_by_key[f"{_norm(model)}|{_norm(kg)}"] = path

    pairs = []
    for eval_dir in sorted(glob.glob(os.path.join(results_root, "answer_*"))):
        name = os.path.basename(eval_dir)
        match = re.match(r"answer_(.+)_evaluator_(.+)_kg_(.+)$", name)
        if not match:
            continue
        model, _judge, kg = match.groups()
        answers = answers_by_key.get(f"{_norm(model)}|{_norm(kg)}")
        if answers and glob.glob(os.path.join(eval_dir, "global_sensemaking_evaluation_*.json")):
            pairs.append((answers, eval_dir))
    return pairs


def _answer_chars(record: Dict, field: str) -> Optional[int]:
    blob = record.get(field)
    if isinstance(blob, dict) and not blob.get("error"):
        text = blob.get("answer") or ""
        if text:
            return len(text)
    return None


def load_ablation_pair(
    lo_dir: str,
    lo_method: str,
    hi_dir: str,
    hi_method: str,
) -> Dict:
    """Build the same run dict as load_run, but from two single-arm ablation directories."""
    lo_field = METHOD_TO_FIELD.get(lo_method)
    hi_field = METHOD_TO_FIELD.get(hi_method)
    if not lo_field or not hi_field:
        raise ValueError(f"unknown method(s): {lo_method!r}, {hi_method!r}")

    lo_answers = json.load(open(os.path.join(lo_dir, "answers.json")))
    hi_answers = json.load(open(os.path.join(hi_dir, "answers.json")))
    lo_eval = _newest_eval_file(lo_dir)
    hi_eval = _newest_eval_file(hi_dir)
    lo_blob = json.load(open(lo_eval))
    hi_blob = json.load(open(hi_eval))

    lo_arms = _arms_in_blob(lo_blob)
    hi_arms = _arms_in_blob(hi_blob)
    # Method key in by_method may be keo / chunk_tok / textchunkrag
    lo_key = lo_method if lo_method in lo_arms else (
        "textchunkrag" if lo_method == "textchunk" and "textchunkrag" in lo_arms else None
    )
    hi_key = hi_method if hi_method in hi_arms else (
        "textchunkrag" if hi_method == "textchunk" and "textchunkrag" in hi_arms else None
    )
    if lo_key is None or hi_key is None:
        raise ValueError(
            f"methods not in evals: lo={list(lo_arms)} (want {lo_method}), "
            f"hi={list(hi_arms)} (want {hi_method})"
        )

    lo_scores = _per_question_scores(lo_arms[lo_key])
    hi_scores = _per_question_scores(hi_arms[hi_key])

    lo_len = {
        r["question_id"]: chars
        for r in lo_answers
        if r.get("question_id") and (chars := _answer_chars(r, lo_field)) is not None
    }
    hi_len = {
        r["question_id"]: chars
        for r in hi_answers
        if r.get("question_id") and (chars := _answer_chars(r, hi_field)) is not None
    }

    run = {
        lo_method: {
            qid: {"score": row[OVERALL], "chars": float(lo_len[qid])}
            for qid, row in lo_scores.items()
            if qid in lo_len
        },
        hi_method: {
            qid: {"score": row[OVERALL], "chars": float(hi_len[qid])}
            for qid, row in hi_scores.items()
            if qid in hi_len
        },
    }
    return {
        "answers_file": f"{lo_dir}+{hi_dir}",
        "eval_file": f"{lo_eval}+{hi_eval}",
        "arms": run,
    }


def load_run(answers_file: str, eval_path: str) -> Dict:
    """Per-question score and answer length for each arm of one run."""
    with open(answers_file) as f:
        answers = json.load(f)
    eval_file = _newest_eval_file(eval_path)
    with open(eval_file) as f:
        blob = json.load(f)

    arms = _arms_in_blob(blob)
    if not arms:
        raise ValueError(f"{eval_file} contains no recognisable method arms")

    lengths: Dict[str, Dict[str, int]] = {}
    for record in answers:
        qid = record.get("question_id")
        if not qid:
            continue
        for arm, field in ARM_TO_FIELD.items():
            blob_arm = record.get(field)
            if isinstance(blob_arm, dict) and not blob_arm.get("error"):
                text = blob_arm.get("answer") or ""
                if text:
                    lengths.setdefault(arm, {})[qid] = len(text)

    run: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm, result in arms.items():
        scores = _per_question_scores(result)
        arm_lengths = lengths.get(arm, {})
        rows = {
            qid: {"score": row[OVERALL], "chars": float(arm_lengths[qid])}
            for qid, row in scores.items()
            if qid in arm_lengths
        }
        if rows:
            run[arm] = rows

    return {
        "answers_file": answers_file,
        "eval_file": eval_file,
        "arms": run,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation with average ranks for ties (no scipy dependency)."""
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1]) if x.size > 2 else 0.0


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # average ties so the coefficient is not inflated by the 1-5 score granularity
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def _ols(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def analyse_pair(
    run: Dict,
    pair: Sequence[str] = DEFAULT_PAIR,
    match_tolerance: float = 0.2,
    seed: int = 1234,
) -> Optional[Dict]:
    """Length-vs-score analysis for one arm pair within one run."""
    lo_arm, hi_arm = pair
    arms = run["arms"]
    if lo_arm not in arms or hi_arm not in arms:
        return None

    qids = sorted(set(arms[lo_arm]) & set(arms[hi_arm]))
    if len(qids) < 8:
        return None

    lo_score = np.array([arms[lo_arm][q]["score"] for q in qids])
    hi_score = np.array([arms[hi_arm][q]["score"] for q in qids])
    lo_len = np.array([arms[lo_arm][q]["chars"] for q in qids])
    hi_len = np.array([arms[hi_arm][q]["chars"] for q in qids])

    log_lo, log_hi = np.log(lo_len), np.log(hi_len)

    # Within-question paired view: both the score gap and the length gap
    d_score = hi_score - lo_score
    d_log_len = log_hi - log_lo

    # How often does the longer answer win the same question?
    longer_is_hi = hi_len > lo_len
    longer_wins = int(np.sum((longer_is_hi & (d_score > 0)) | (~longer_is_hi & (d_score < 0))))
    longer_loses = int(np.sum((longer_is_hi & (d_score < 0)) | (~longer_is_hi & (d_score > 0))))
    ties = int(np.sum(d_score == 0))

    # Question fixed effects via within-question demeaning of the stacked 2-arm panel
    scores = np.concatenate([lo_score, hi_score])
    logs = np.concatenate([log_lo, log_hi])
    is_hi = np.concatenate([np.zeros_like(lo_score), np.ones_like(hi_score)])
    q_mean_score = np.tile((lo_score + hi_score) / 2, 2)
    q_mean_log = np.tile((log_lo + log_hi) / 2, 2)
    q_mean_hi = np.tile(np.full(len(qids), 0.5), 2)
    y = scores - q_mean_score
    x_hi = is_hi - q_mean_hi
    x_len = logs - q_mean_log

    gap_raw = float(_ols(x_hi.reshape(-1, 1), y)[0])
    both = _ols(np.column_stack([x_hi, x_len]), y)
    gap_adjusted, length_coefficient = float(both[0]), float(both[1])

    # Collinearity: with question FE, arm and length are the same contrast if the
    # graph arm is longer on every question. Report it rather than trusting the fit.
    arm_length_corr = float(np.corrcoef(x_hi, x_len)[0, 1]) if x_len.std() > 0 else 1.0
    share_hi_longer = float(np.mean(longer_is_hi))

    # Length-matched subsample: questions where the two answers are within tolerance
    matched = np.abs(d_log_len) <= match_tolerance
    matched_result = None
    if matched.sum() >= 8:
        matched_result = paired_compare(
            lo_score[matched], hi_score[matched], n_boot=5000, n_perm=5000, seed=seed
        )
        matched_result["n_matched"] = int(matched.sum())
        matched_result["tolerance_log_chars"] = match_tolerance

    return {
        "pair": [lo_arm, hi_arm],
        "n_questions": len(qids),
        "length_chars": {
            lo_arm: {"mean": float(lo_len.mean()), "median": float(np.median(lo_len))},
            hi_arm: {"mean": float(hi_len.mean()), "median": float(np.median(hi_len))},
            "ratio_hi_over_lo": float(hi_len.mean() / lo_len.mean()) if lo_len.mean() else 0.0,
        },
        "score": {
            lo_arm: float(lo_score.mean()),
            hi_arm: float(hi_score.mean()),
            "gap": float(d_score.mean()),
        },
        "length_score_correlation": {
            "spearman_within_question_diffs": _spearman(d_log_len, d_score),
            "pearson_within_question_diffs": (
                float(np.corrcoef(d_log_len, d_score)[0, 1]) if d_log_len.std() > 0 else 0.0
            ),
            "spearman_pooled": _spearman(logs, scores),
        },
        "longer_answer_wins": {
            "wins": longer_wins, "ties": ties, "losses": longer_loses,
            "rate_excluding_ties": (
                longer_wins / (longer_wins + longer_loses)
                if (longer_wins + longer_loses) else None
            ),
        },
        "fixed_effect_model": {
            "gap_raw": gap_raw,
            "gap_adjusted_for_log_length": gap_adjusted,
            "log_length_coefficient": length_coefficient,
            "share_of_gap_absorbed_by_length": (
                float(1 - gap_adjusted / gap_raw) if abs(gap_raw) > 1e-9 else None
            ),
        },
        "collinearity": {
            "arm_vs_log_length_corr": arm_length_corr,
            "share_of_questions_where_graph_arm_longer": share_hi_longer,
            "separable": bool(abs(arm_length_corr) < 0.9),
        },
        "length_matched_subsample": matched_result,
    }


def summarise(per_run: List[Dict], pair: Sequence[str] = DEFAULT_PAIR) -> Dict:
    """Pool the per-run analyses into headline numbers."""
    usable = [r for r in per_run if r.get("analysis")]
    if not usable:
        return {"runs": 0}
    a = [r["analysis"] for r in usable]
    ratios = [x["length_chars"]["ratio_hi_over_lo"] for x in a]
    gaps = [x["score"]["gap"] for x in a]
    corrs = [x["length_score_correlation"]["spearman_within_question_diffs"] for x in a]
    rates = [
        x["longer_answer_wins"]["rate_excluding_ties"]
        for x in a if x["longer_answer_wins"]["rate_excluding_ties"] is not None
    ]
    separable = sum(1 for x in a if x["collinearity"]["separable"])
    matched = [x["length_matched_subsample"] for x in a if x["length_matched_subsample"]]
    return {
        "runs": len(a),
        "pair": list(pair),
        "length_ratio_graph_over_chunk": {
            "median": float(np.median(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
        },
        "score_gap": {"median": float(np.median(gaps)), "mean": float(np.mean(gaps))},
        "within_question_length_score_spearman": {
            "median": float(np.median(corrs)),
            "min": float(np.min(corrs)),
            "max": float(np.max(corrs)),
        },
        "longer_answer_win_rate": (
            {"median": float(np.median(rates)), "n_runs": len(rates)} if rates else None
        ),
        "runs_where_arm_and_length_are_separable": f"{separable}/{len(a)}",
        "length_matched": _summarise_matched(a, matched),
    }


def _summarise_matched(analyses: List[Dict], matched: List[Dict]) -> Optional[Dict]:
    """
    Matched-subsample summary, including its own power limits.

    These subsamples are small by construction — the arms rarely produce answers of
    similar length — so a non-significant result here means "undetectable", not
    "absent". The shrinkage of the point estimate is the informative part.
    """
    if not matched:
        return None
    full_gaps = [
        x["score"]["gap"] for x in analyses if x["length_matched_subsample"]
    ]
    matched_gaps = [m["mean_diff"] for m in matched]
    median_full = float(np.median(full_gaps))
    median_matched = float(np.median(matched_gaps))
    return {
        "n_runs": len(matched),
        "median_n_per_subsample": float(np.median([m["n_matched"] for m in matched])),
        "median_gap_full_sample": median_full,
        "median_gap_length_matched": median_matched,
        "point_estimate_shrinkage": (
            float(1 - median_matched / median_full) if abs(median_full) > 1e-9 else None
        ),
        "significant_runs": sum(1 for m in matched if m["p_value"] < 0.05),
        "underpowered_runs": sum(1 for m in matched if m.get("underpowered")),
        "median_mde_80": float(np.median([m["mde_80"] for m in matched])),
    }


def main():
    parser = argparse.ArgumentParser(description="Judge length-bias audit on archived runs")
    parser.add_argument("--auto", action="store_true", help="Discover all answers/eval pairs")
    parser.add_argument("--answers", default=None, help="One answers JSON")
    parser.add_argument("--eval", dest="eval_path", default=None, help="Its judged run dir or file")
    parser.add_argument(
        "--ablation-lo",
        default=None,
        metavar="DIR",
        help="Ablation arm dir for the low/chunk side (answers.json + judged eval)",
    )
    parser.add_argument(
        "--ablation-hi",
        default=None,
        metavar="DIR",
        help="Ablation arm dir for the high/graph side",
    )
    parser.add_argument(
        "--lo-method",
        default=None,
        help="Method key inside --ablation-lo (default: chunk_tok)",
    )
    parser.add_argument(
        "--hi-method",
        default=None,
        help="Method key inside --ablation-hi (default: keo)",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        default=None,
        metavar=("CHUNK_ARM", "GRAPH_ARM"),
        help=f"Arms to compare (default: {DEFAULT_PAIR[0]} {DEFAULT_PAIR[1]}, "
             "or lo/hi-method when using --ablation-*)",
    )
    parser.add_argument(
        "--match-tolerance", type=float, default=0.2,
        help="Max |Δ log chars| for the length-matched subsample (0.2 ≈ within 22%%)",
    )
    parser.add_argument("--out", default=None, help="Write full results JSON here")
    parser.add_argument("--limit", type=int, default=None, help="Analyse at most N runs")
    args = parser.parse_args()

    per_run = []
    if args.ablation_lo or args.ablation_hi:
        if not (args.ablation_lo and args.ablation_hi):
            parser.error("--ablation-lo and --ablation-hi must be given together")
        lo_method = args.lo_method or "chunk_tok"
        hi_method = args.hi_method or "keo"
        pair = tuple(args.pair) if args.pair else (lo_method, hi_method)
        args.pair = list(pair)
        print(
            f"Ablation pair: {lo_method} @ {args.ablation_lo}  vs  "
            f"{hi_method} @ {args.ablation_hi}\n"
        )
        try:
            run = load_ablation_pair(
                args.ablation_lo, lo_method, args.ablation_hi, hi_method
            )
            if pair[0] not in run["arms"] or pair[1] not in run["arms"]:
                run["arms"] = {
                    pair[0]: run["arms"][lo_method],
                    pair[1]: run["arms"][hi_method],
                }
            analysis = analyse_pair(run, pair, args.match_tolerance)
        except Exception as e:
            print(f"  abort: {e}")
            return
        per_run.append({
            "run": (
                f"{os.path.basename(args.ablation_lo.rstrip('/'))}"
                f"_vs_{os.path.basename(args.ablation_hi.rstrip('/'))}"
            ),
            "answers_file": run["answers_file"],
            "eval_file": run["eval_file"],
            "analysis": analysis,
        })
    else:
        pair = tuple(args.pair) if args.pair else DEFAULT_PAIR
        args.pair = list(pair)
        if args.auto:
            pairs = discover_pairs()
        elif args.answers and args.eval_path:
            pairs = [(args.answers, args.eval_path)]
        else:
            parser.error(
                "use --auto, --answers+--eval, or --ablation-lo+--ablation-hi"
            )

        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            print("No runs found with both an answers file and a judged evaluation.")
            return

        print(f"Analysing {len(pairs)} run(s) with answers + judgments\n")
        for answers_file, eval_path in pairs:
            try:
                run = load_run(answers_file, eval_path)
                analysis = analyse_pair(run, pair, args.match_tolerance)
            except Exception as e:
                print(f"  skip {os.path.basename(eval_path)}: {e}")
                continue
            per_run.append({
                "run": os.path.basename(eval_path.rstrip("/")),
                "answers_file": answers_file,
                "eval_file": run["eval_file"],
                "analysis": analysis,
            })

    lo_arm, hi_arm = args.pair
    header = (
        f"{'run':<52}{'n':>4}{'len ratio':>11}{'Δscore':>8}"
        f"{'ρ(len,score)':>14}{'longer-wins':>13}{'sep?':>6}"
    )
    print(header)
    for entry in per_run:
        a = entry["analysis"]
        if not a:
            print(f"{entry['run'][:50]:<52}{'--':>4}  (arms {lo_arm}/{hi_arm} not both present)")
            continue
        rate = a["longer_answer_wins"]["rate_excluding_ties"]
        print(
            f"{entry['run'][:50]:<52}{a['n_questions']:>4}"
            f"{a['length_chars']['ratio_hi_over_lo']:>10.2f}x"
            f"{a['score']['gap']:>+8.3f}"
            f"{a['length_score_correlation']['spearman_within_question_diffs']:>+14.3f}"
            f"{(f'{rate:.0%}' if rate is not None else '--'):>13}"
            f"{('yes' if a['collinearity']['separable'] else 'NO'):>6}"
        )

    pooled = summarise(per_run, args.pair)
    print()
    if pooled.get("runs"):
        lr = pooled["length_ratio_graph_over_chunk"]
        print(
            f"Across {pooled['runs']} runs: {hi_arm} answers are "
            f"{lr['median']:.1f}x longer than {lo_arm} (range {lr['min']:.1f}-{lr['max']:.1f}x), "
            f"median score gap {pooled['score_gap']['median']:+.3f}."
        )
        print(
            f"Within-question correlation between Δlog length and Δscore: median "
            f"ρ={pooled['within_question_length_score_spearman']['median']:+.3f}."
        )
        if pooled["longer_answer_win_rate"]:
            print(
                f"The longer answer wins {pooled['longer_answer_win_rate']['median']:.0%} "
                "of non-tied questions."
            )
        print(
            f"Arm and length statistically separable in "
            f"{pooled['runs_where_arm_and_length_are_separable']} runs."
        )
        lm = pooled.get("length_matched")
        if lm:
            print(
                f"\nLength-matched subsamples ({lm['n_runs']} runs, median n="
                f"{lm['median_n_per_subsample']:.0f}): gap "
                f"{lm['median_gap_full_sample']:+.3f} → {lm['median_gap_length_matched']:+.3f} "
                f"({lm['point_estimate_shrinkage']:.0%} smaller), significant in "
                f"{lm['significant_runs']}/{lm['n_runs']}."
            )
            print(
                f"  But {lm['underpowered_runs']}/{lm['n_runs']} are underpowered "
                f"(median MDE {lm['median_mde_80']:.3f}): these subsamples cannot "
                "distinguish a small effect from none."
            )
        print(
            "\nReading: length and arm are too entangled to separate after the fact. "
            "The shrinkage is suggestive, not conclusive. Settle it with the "
            "length-matched generation arm (--method chunk_tok)."
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"pooled": pooled, "runs": per_run}, f, indent=2)
        print(f"\n✓ Length-bias audit → {args.out}")


if __name__ == "__main__":
    main()
