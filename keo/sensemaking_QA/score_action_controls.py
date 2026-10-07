#!/usr/bin/env python3
"""Score knowledge-to-action (action_specific) answers already on disk.

Control ablation trees answer all 133 questions, including 50 action items with
ground_truth_answer, but run_evaluation historically only pairwise-compared
vanilla/textchunk/graphrag — so single-method control dirs never got an action
file. This script scores every arm under a run-id with NLP metrics (no Claude)
and writes paired Δ vs a reference arm (default: mst_on_freq / keo).

  KEO_ACTION_NLP_ONLY=1 python score_action_controls.py --run-id phi4
  KEO_ACTION_NLP_ONLY=1 python score_action_controls.py --run-id labeled_graph

Do not pool NLP-only rows with LLM-judged archived GPT/Llama action tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KEO_ACTION_NLP_ONLY", "1")

from evaluator import SensemakingEvaluator  # noqa: E402
from run_evaluation import METHOD_FIELDS, load_method_answers  # noqa: E402

# Arm directory name → preferred method key inside answers.json
ARM_METHOD = {
    "mst_on_freq": "keo",
    "mst_off_freq": "keo",
    "textchunk": "textchunkrag",
    "ctrl_chunk_tok": "chunk_tok",
    "ctrl_chunk_sum": "chunk_sum",
    "ctrl_chunk_dedup": "chunk_dedup",
    "ctrl_keo_norel": "keo_norel",
    "ctrl_keo_shuf_s13": "keo_shuf",
    "ctrl_keo_shuf_s29": "keo_shuf",
    "ctrl_keo_shuf_s71": "keo_shuf",
    "risk_floor0": "risk",
    "risk_floor0.25": "risk",
    "risk_nofill": "risk",
}


def _load_questions(path: Path) -> List[Dict]:
    blob = json.loads(path.read_text())
    if isinstance(blob, list):
        return blob
    return blob.get("questions") or []


def _pick_method(arm_dir: str, present: Dict[str, List[Dict]]) -> Optional[str]:
    preferred = ARM_METHOD.get(arm_dir)
    if preferred and preferred in present and present[preferred]:
        return preferred
    # fall back: first non-empty present method, preferring keo over graphrag alias
    for name, _ in METHOD_FIELDS:
        if name == "graphrag" and "keo" in present and present["keo"]:
            continue
        if present.get(name):
            return name
    return None


def _per_question_rouge(result: Dict) -> Dict[str, float]:
    out = {}
    for ev in result.get("individual_evaluations") or []:
        qid = getattr(ev, "question_id", None) or (ev.get("question_id") if isinstance(ev, dict) else None)
        if not qid:
            continue
        if hasattr(ev, "nlp_metrics"):
            out[qid] = float(ev.nlp_metrics.rouge_l_f1)
        elif isinstance(ev, dict):
            nlp = ev.get("nlp_metrics") or {}
            out[qid] = float(nlp.get("rouge_l_f1") or 0.0)
    return out


def _dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    return obj


def score_run(
    run_id: str,
    questions_file: Path,
    reference_arm: str,
    out_path: Path,
) -> Dict:
    abl = ROOT / "evaluation_results" / "ablations" / run_id
    if not abl.is_dir():
        raise SystemExit(f"missing ablation root: {abl}")

    questions = _load_questions(questions_file)
    action_qs = [q for q in questions if q.get("category") == "action_specific"]
    if not action_qs:
        raise SystemExit("no action_specific questions with category set")

    # Instantiates Claude CLI login check only; LLM calls are skipped via KEO_ACTION_NLP_ONLY.
    evaluator = SensemakingEvaluator(
        provider=os.getenv("JUDGE_PROVIDER", "claude_cli"),
        model=os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001"),
    )

    arms_out: Dict[str, Any] = {}
    per_q: Dict[str, Dict[str, float]] = {}

    for arm_dir in sorted(p.name for p in abl.iterdir() if p.is_dir()):
        answers_path = abl / arm_dir / "answers.json"
        if not answers_path.is_file():
            continue
        answers_data = json.loads(answers_path.read_text())
        present, _failed = load_method_answers(answers_data)
        present = {k: v for k, v in present.items() if v}
        method = _pick_method(arm_dir, present)
        if not method:
            print(f"  skip {arm_dir}: no method answers")
            continue
        print(f"  → {arm_dir} as {method} (n={len(present[method])})")
        result = evaluator.evaluate_action_specific_answers(present[method], action_qs)
        if result.get("error"):
            print(f"    ❌ {result['error']}")
            continue
        arms_out[arm_dir] = {
            "method": method,
            "aggregate_metrics": result.get("aggregate_metrics"),
            "evaluated_answers": result.get("evaluated_answers"),
            "total_action_questions": result.get("total_action_questions"),
            "evaluation_summary": result.get("evaluation_summary"),
            # keep per-question for pairing; strip long predicted text to keep file smaller
            "per_question_rouge_l": _per_question_rouge(result),
        }
        per_q[arm_dir] = arms_out[arm_dir]["per_question_rouge_l"]
        # also write per-arm action file for provenance
        arm_out = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "run_id": run_id,
            "arm": arm_dir,
            "method": method,
            "nlp_only": True,
            "by_method": {method: _dataclass_to_dict(result)},
        }
        meta = abl / arm_dir / "answers_meta.json"
        if meta.is_file():
            try:
                arm_out["answer_model"] = json.loads(meta.read_text()).get("answer_model")
            except json.JSONDecodeError:
                pass
        out_arm = abl / arm_dir / f"action_specific_nlp_{arm_out['timestamp']}.json"
        out_arm.write_text(json.dumps(arm_out, indent=2, default=str))
        print(f"    wrote {out_arm.name}")

    if reference_arm not in arms_out:
        raise SystemExit(
            f"reference arm {reference_arm!r} not scored; have {sorted(arms_out)}"
        )

    ref_scores = per_q[reference_arm]
    paired = {}
    for arm, scores in per_q.items():
        if arm == reference_arm:
            continue
        common = sorted(set(ref_scores) & set(scores))
        if len(common) < 5:
            paired[arm] = {"error": f"too few paired ids ({len(common)})"}
            continue
        deltas = [scores[q] - ref_scores[q] for q in common]
        import statistics as st

        mean_d = st.mean(deltas)
        paired[arm] = {
            "n": len(common),
            "metric": "rouge_l_f1",
            "mean_arm": st.mean(scores[q] for q in common),
            "mean_ref": st.mean(ref_scores[q] for q in common),
            "delta_vs_ref": mean_d,
            "delta_std": st.pstdev(deltas) if len(deltas) > 1 else 0.0,
        }

    blob = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "reference_arm": reference_arm,
        "reference_method": arms_out[reference_arm]["method"],
        "nlp_only": True,
        "n_action_questions": len(action_qs),
        "note": (
            "Ground-truth strings are short maintenance log phrases; model answers are "
            "longer prose, so absolute BLEU/ROUGE are low. Use paired Δ across arms. "
            "Do not pool with archived GPT/Llama LLM action tables."
        ),
        "arms": arms_out,
        "paired_rouge_l_vs_reference": paired,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n✓ Stats → {out_path}")

    # compact table
    print(f"\n{'arm':<22} {'rougeL':>8} {'bleu':>8} {'exact':>8} {'Δvs ref':>10}")
    ref_agg = arms_out[reference_arm]["aggregate_metrics"]
    print(
        f"{reference_arm + ' (ref)':<22} "
        f"{ref_agg.get('rouge_l_f1_scores_mean', 0):8.3f} "
        f"{ref_agg.get('bleu_scores_mean', 0):8.3f} "
        f"{ref_agg.get('exact_matches_rate', 0):8.3f} "
        f"{'—':>10}"
    )
    for arm, row in sorted(paired.items(), key=lambda x: -(x[1].get("delta_vs_ref") or -99)):
        if row.get("error"):
            print(f"{arm:<22} {row['error']}")
            continue
        agg = arms_out[arm]["aggregate_metrics"]
        print(
            f"{arm:<22} "
            f"{agg.get('rouge_l_f1_scores_mean', 0):8.3f} "
            f"{agg.get('bleu_scores_mean', 0):8.3f} "
            f"{agg.get('exact_matches_rate', 0):8.3f} "
            f"{row['delta_vs_ref']:+10.3f}"
        )
    return blob


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="ablations/<run-id>/ tree")
    p.add_argument(
        "--questions-file",
        default=str(ROOT / "output" / "aviation_sensemaking_questions.json"),
    )
    p.add_argument("--reference-arm", default="mst_on_freq")
    p.add_argument(
        "--out",
        default=None,
        help="default: evaluation_results/stats/action_<run-id>_nlp.json",
    )
    args = p.parse_args()
    out = Path(args.out) if args.out else ROOT / "evaluation_results" / "stats" / f"action_{args.run_id}_nlp.json"
    score_run(args.run_id, Path(args.questions_file), args.reference_arm, out)


if __name__ == "__main__":
    main()
