#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Aviation Sensemaking QA
Includes action-specific evaluation with ground truth metrics

Judge grades (judge_policy.py):
- paper (default): pinned Claude via Claude Code OR Messages API — reportable
- smoke: same stack, marked as pipeline check
- legacy: OpenAI GPT-5.6 — continuity only

Default provider is Claude Code (`JUDGE_PROVIDER` / `KEO_JUDGE_PROVIDER=claude_cli`) — no API key.
Set JUDGE_PROVIDER=anthropic or --provider anthropic for Messages API.

Usage:
    claude auth login   # once
    python run_evaluation.py --questions-file Q.json --answers-file A.json --output-dir results

    # Messages API instead
    ANTHROPIC_API_KEY=... python run_evaluation.py ... --provider anthropic

    # Legacy GPT
    python run_evaluation.py ... --provider openai --evaluation-model gpt-5.6-luna
"""

import os
import json
import pandas as pd
import argparse
from datetime import datetime
from typing import Optional
from evaluator import SensemakingEvaluator
from judge_policy import resolve_judge_defaults, warn_if_non_paper
from judge_skip import answers_model_from_answers_file
from claude_cli import ClaudeQuotaError

CHECKPOINT_NAME = "global_sensemaking_checkpoint.json"


def _load_global_checkpoint(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        blob = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None
    return blob if isinstance(blob, dict) else None


def _save_global_checkpoint(path: str, blob: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    os.replace(tmp, path)


def _method_prior_evals(ckpt: Optional[dict], method: str) -> list:
    if not ckpt:
        return []
    by = (ckpt.get("by_method") or {}).get(method) or {}
    evs = by.get("individual_evaluations") or []
    return [e for e in evs if isinstance(e, dict) and e.get("question_id") and not e.get("error")]

# Method arms recognised in answers files (legacy + ablation + mechanism controls).
# Every arm generate_ablation_answers.py can write must appear here or it is silently unjudged.
METHOD_FIELDS = (
    ("vanilla", "vanilla_answer"),
    ("textchunkrag", "textchunkrag_answer"),
    ("keo", "keo_answer"),
    ("routed", "routed_answer"),
    ("hybrid", "hybrid_answer"),
    ("community", "community_answer"),
    ("graphrag", "graphrag_answer"),
    # Selection policies (papers/ACL_STRATEGY.md §3 Pivot C): same prompt surface as keo
    ("risk", "risk_answer"),
    # Mechanism controls (papers/ACL_STRATEGY.md §4.1): judged with the same protocol
    ("chunk_tok", "chunk_tok_answer"),
    ("chunk_sum", "chunk_sum_answer"),
    ("chunk_dedup", "chunk_dedup_answer"),
    ("keo_shuf", "keo_shuf_answer"),
    ("keo_norel", "keo_norel_answer"),
)
# Control arms stand in for their family when a run contains only the control.
# chunk_tok is the length-matched chunk reference; prefer it over textchunkrag for headline numbers.
CHUNK_FAMILY = ("textchunkrag", "chunk_tok", "chunk_dedup", "chunk_sum")
GRAPH_FAMILY = (
    "keo", "routed", "hybrid", "community", "graphrag", "risk", "keo_shuf", "keo_norel",
)


def load_method_answers(answers_data):
    """
    Split an answers file into judgeable answers per arm and a count of failed ones.

    Keeps EVERY non-null arm so single-method paper-min files still judge. A failed
    generation stores its exception text in `answer` alongside `error: True`; judging
    that string is worse than useless, because the judge scores
    "Error generating answer: Model X is not supported" at the 1-5 floor and the result
    is indistinguishable from a genuinely poor method in every downstream table.

    Returns (method_answers, failed_counts), both keyed by arm name.
    """
    method_answers = {name: [] for name, _ in METHOD_FIELDS}
    failed_counts = {name: 0 for name, _ in METHOD_FIELDS}

    for answer_result in answers_data:
        qid = answer_result.get("question_id")
        for name, field in METHOD_FIELDS:
            blob = answer_result.get(field)
            if not blob or not isinstance(blob, dict):
                continue
            if blob.get("error"):
                failed_counts[name] += 1
                continue
            text = blob.get("answer")
            if text is None:
                continue
            method_answers[name].append(
                {"question_id": qid, "answer": text, "method": name}
            )

    return method_answers, failed_counts


def run_comprehensive_evaluation(
    questions_file,
    answers_file,
    output_dir,
    evaluation_model=None,
    provider=None,
    API_provider="nebius",
    judge_grade=None,
    final=False,
):
    """Run comprehensive evaluation including action-specific ground truth evaluation"""
    
    print("=" * 70)
    print("COMPREHENSIVE AVIATION SENSEMAKING QA EVALUATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    provider, evaluation_model, grade = resolve_judge_defaults(
        provider=provider,
        model=evaluation_model,
        grade=judge_grade,
        final=final,
    )
    print(f"Judge policy: grade={grade} provider={provider} model={evaluation_model}")
    w = warn_if_non_paper(grade)
    if w:
        print(w)
    
    api_key = None
    if provider == "claude_cli":
        print("✓ Judge via Claude Code (pinned snapshot / Pro-Max — paper-parity protocol)")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Use --provider claude_cli (default) for subscription judging, "
                "or export ANTHROPIC_API_KEY for Messages API."
            )
        print("✓ Anthropic API key verified (Messages API)")
    elif provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        api_key = openai_api_key
        print("✓ OpenAI API key verified (legacy judge path)")
    elif provider == "huggingface":
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        api_key = hf_token
        print("✓ HuggingFace token verified")
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Supported: 'claude_cli' (paper-parity), 'anthropic' (API), "
            "'openai' (legacy), 'huggingface'"
        )
    
    print(f"Initializing evaluator with provider: {provider}")
    if provider == "huggingface":
        print(f"Using HuggingFace API provider: {API_provider}")
    evaluator = SensemakingEvaluator(
        api_key,
        model=evaluation_model,
        provider=provider,
        API_provider=API_provider,
        judge_grade=grade,
    )
    print("✓ Evaluator initialized")
    
    # Load questions and answers
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    # Load questions
    if not os.path.exists(questions_file):
        print(f"ERROR: Questions file not found: {questions_file}")
        return
    
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    print(f"✓ Loaded {len(questions)} questions from {questions_file}")
    
    # Categorize questions
    question_categories = {}
    for q in questions:
        category = q.get('category', 'unknown')
        question_categories[category] = question_categories.get(category, 0) + 1
    
    print("Question categories:")
    for category, count in question_categories.items():
        print(f"  - {category}: {count}")
    
    # Load answers
    if not os.path.exists(answers_file):
        print(f"ERROR: Answers file not found: {answers_file}")
        return
    
    with open(answers_file, 'r') as f:
        answers_data = json.load(f)
    
    method_answers, failed_counts = load_method_answers(answers_data)

    # Back-compat aliases used by remaining three-way code
    vanilla_answers = method_answers["vanilla"]

    def _first_present(family):
        for name in family:
            if method_answers[name]:
                return method_answers[name]
        return []

    textchunkrag_answers = _first_present(CHUNK_FAMILY)
    graphrag_answers = _first_present(GRAPH_FAMILY)
    present = {k: v for k, v in method_answers.items() if v}
    print(
        "✓ Loaded method arms: "
        + ", ".join(f"{k}={len(v)}" for k, v in present.items())
        or "(none)"
    )
    failed = {k: v for k, v in failed_counts.items() if v}
    if failed:
        print(
            "⚠ Skipped failed generations (not judged): "
            + ", ".join(f"{k}={v}" for k, v in failed.items())
        )
    if not present:
        detail = (
            f" Every answer failed generation ({failed}); fix generation before judging."
            if failed else ""
        )
        raise RuntimeError(f"No usable answers in {answers_file}.{detail}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize file variables
    action_eval_file = None
    global_eval_file = None
    standard_eval_file = None
    
    # print("\n" + "=" * 70)
    # print("STEP 2: QUESTION QUALITY EVALUATION")
    # print("=" * 70)
    
    # Get unique question IDs from answers file
    answered_question_ids = set()
    for answer_result in answers_data:
        answered_question_ids.add(answer_result['question_id'])
    
    # Filter questions to only those that have been answered
    answered_questions = [q for q in questions if q.get('id') in answered_question_ids]
    print(f"Found {len(answered_questions)} questions that have been answered out of {len(questions)} total questions")
    
    # # Evaluate question quality for answered questions only
    # question_evaluations = evaluator.evaluate_questions(answered_questions)
    
    # question_eval_file = f"{output_dir}/question_evaluations_{timestamp}.json"
    # evaluator.save_evaluation_results(
    #     {'question_evaluations': question_evaluations}, 
    #     question_eval_file
    # )
    # print(f"✓ Question evaluations saved to {question_eval_file}")
    
    print("\n" + "=" * 70)
    print("STEP 3: ACTION-SPECIFIC EVALUATION WITH GROUND TRUTH")
    print("=" * 70)
    
    # Filter action-specific questions that have been answered
    action_questions = [q for q in questions if q.get('category') == 'action_specific' and q.get('id') in answered_question_ids]
    
    if action_questions:
        print(f"Found {len(action_questions)} answered action-specific questions with ground truth")
        # Score every present arm absolutely (control trees are single-method).
        # Legacy vanilla/textchunk/graphrag pairwise stays when those arms exist.
        action_evaluation = {
            "timestamp": timestamp,
            "by_method": {},
            "comparisons": {},
            "nlp_only": os.getenv("KEO_ACTION_NLP_ONLY", "").strip().lower() in ("1", "true", "yes"),
        }
        answer_model = answers_model_from_answers_file(answers_file)
        if answer_model:
            action_evaluation["answer_model"] = answer_model

        any_ok = False
        for name, answers in present.items():
            # Prefer the primary method field; skip duplicate graphrag alias of keo when identical
            print(f"  → action scoring {name} (n={len(answers)})")
            result = evaluator.evaluate_action_specific_answers(answers, action_questions)
            action_evaluation["by_method"][name] = result
            if result.get("error"):
                print(f"❌ {name} action failed: {result.get('error')}")
            else:
                any_ok = True
                agg = result.get("aggregate_metrics") or {}
                print(
                    f"✓ {name} action n={agg.get('total_evaluated')} "
                    f"rougeL={agg.get('rouge_l_f1_scores_mean', 0):.3f} "
                    f"bleu={agg.get('bleu_scores_mean', 0):.3f} "
                    f"exact={agg.get('exact_matches_rate', 0):.3f}"
                )

        # Optional pairwise among classic three when ≥2 present
        classic = {
            "vanilla": vanilla_answers,
            "textchunkrag": textchunkrag_answers,
            "graphrag": graphrag_answers,
        }
        pairs = [
            ("vanilla", "textchunkrag"),
            ("vanilla", "graphrag"),
            ("textchunkrag", "graphrag"),
        ]
        for a, b in pairs:
            if not classic[a] or not classic[b]:
                continue
            # Prefer already-computed by_method blocks when available
            if a in action_evaluation["by_method"] and b in action_evaluation["by_method"]:
                if action_evaluation["by_method"][a].get("error") or action_evaluation["by_method"][b].get("error"):
                    continue
            print(f"Comparing action methods: {a} vs {b}")
            cmp = evaluator.compare_action_methods_flexible(
                classic[a], classic[b], action_questions, a, b
            )
            key = f"{a}_vs_{b}"
            action_evaluation[key] = cmp
            action_evaluation["comparisons"][key] = cmp
            if cmp.get("error"):
                print(f"❌ {key} failed: {cmp.get('error')}")
            else:
                any_ok = True
                print(f"✓ {key} succeeded")

        if any_ok:
            action_eval_file = f"{output_dir}/action_specific_evaluation_{timestamp}.json"
            evaluator.save_evaluation_results(action_evaluation, action_eval_file)
            print(f"✓ Action-specific evaluation saved to {action_eval_file}")
        else:
            print("No successful action-method scores")
    
    else:
        print("No action-specific questions found")
    
    print("\n" + "=" * 70)
    print("STEP 4: OTHER SENSEMAKING QUESTION EVALUATION")
    print("=" * 70)
    
    # Filter global questions that have been answered
    global_questions = [q for q in questions if q.get('id') in answered_question_ids and q.get('category') != 'action_specific']
    
    if global_questions:
        print(f"Found {len(global_questions)} answered global sensemaking questions")
        print(f"Evaluating present method arms on {len(global_questions)} global questions")

        # Absolute scores for every present arm (works for single-method paper-min files)
        global_evaluation = {"timestamp": timestamp, "by_method": {}}
        # Record the answerer on the eval itself. Skip-on-exists compares this to
        # answers_meta.json; without it a regenerated arm would inherit the old scores.
        answer_model = answers_model_from_answers_file(answers_file)
        if answer_model:
            global_evaluation["answer_model"] = answer_model

        checkpoint_path = os.path.join(output_dir, CHECKPOINT_NAME)
        ckpt = _load_global_checkpoint(checkpoint_path) or {
            "partial": True,
            "complete": False,
            "by_method": {},
        }
        if answer_model:
            ckpt["answer_model"] = answer_model
        # Refuse to resume a checkpoint from a different answerer
        ckpt_model = ckpt.get("answer_model")
        if ckpt_model and answer_model and ckpt_model != answer_model:
            print(
                f"  checkpoint answer_model={ckpt_model} ≠ current {answer_model}; "
                "starting fresh (old checkpoint kept on disk)"
            )
            ckpt = {
                "partial": True,
                "complete": False,
                "by_method": {},
                "answer_model": answer_model,
            }

        scored = 0
        try:
            for name, answers in present.items():
                prior = _method_prior_evals(ckpt, name)
                print(f"  → scoring {name} (n={len(answers)}, resume={len(prior)})")

                def _progress(evals, _name=name, _expected=len(answers)):
                    block = {
                        "global_questions_evaluated": len(evals),
                        "individual_evaluations": evals,
                        "failed_question_ids": [],
                        "incomplete": len(evals) < _expected,
                        "expected_n": _expected,
                    }
                    ckpt.setdefault("by_method", {})[_name] = block
                    ckpt["partial"] = True
                    ckpt["complete"] = False
                    _save_global_checkpoint(checkpoint_path, ckpt)

                result = evaluator.evaluate_global_sensemaking_capability(
                    global_questions,
                    answers,
                    prior_evaluations=prior,
                    on_progress=_progress,
                )
                global_evaluation["by_method"][name] = result
                global_evaluation[f"{name}_global"] = result  # back-compat keys
                ckpt.setdefault("by_method", {})[name] = {
                    "global_questions_evaluated": result.get("global_questions_evaluated"),
                    "individual_evaluations": result.get("individual_evaluations") or [],
                    "failed_question_ids": result.get("failed_question_ids") or [],
                    "incomplete": bool(result.get("incomplete")),
                    "expected_n": result.get("expected_n") or len(answers),
                    "global_metrics": result.get("global_metrics"),
                }
                _save_global_checkpoint(checkpoint_path, ckpt)

                if result.get("error"):
                    print(f"❌ {name} global failed: {result.get('error')}")
                elif result.get("incomplete"):
                    print(
                        f"⚠ {name} incomplete "
                        f"{result.get('global_questions_evaluated')}/{result.get('expected_n')}"
                    )
                else:
                    scored += 1
                    metrics = result.get("global_metrics") or {}
                    overall = sum(
                        metrics.get(m, 0)
                        for m in (
                            "global_perspective",
                            "theme_identification",
                            "synthesis_quality",
                            "strategic_value",
                            "pattern_recognition",
                        )
                    ) / 5.0
                    print(f"✓ {name} overall_avg={overall:.3f}")
        except ClaudeQuotaError as exc:
            _save_global_checkpoint(checkpoint_path, ckpt)
            n_saved = sum(
                len((b or {}).get("individual_evaluations") or [])
                for b in (ckpt.get("by_method") or {}).values()
            )
            print(
                f"\n⚠ Claude quota — saved checkpoint with {n_saved} successful judgments → "
                f"{checkpoint_path}\nRe-run the same command after the limit resets; "
                f"already-scored questions are skipped.\n{exc}"
            )
            raise SystemExit(3) from exc

        # Legacy three-way keys when those arms exist
        if vanilla_answers:
            global_evaluation["vanilla_global"] = global_evaluation["by_method"].get("vanilla")
        if textchunkrag_answers:
            for cand in CHUNK_FAMILY:
                if cand in global_evaluation["by_method"]:
                    global_evaluation["textchunk_global"] = global_evaluation["by_method"][cand]
                    break
        if graphrag_answers:
            # Map primary graph arm into graphrag_global
            for cand in GRAPH_FAMILY:
                if cand in global_evaluation["by_method"]:
                    global_evaluation["graphrag_global"] = global_evaluation["by_method"][cand]
                    break

        if scored:
            global_evaluation["partial"] = False
            global_evaluation["complete"] = True
            global_eval_file = f"{output_dir}/global_sensemaking_evaluation_{timestamp}.json"
            evaluator.save_evaluation_results(global_evaluation, global_eval_file)
            print(f"✓ Global sensemaking evaluation saved to {global_eval_file} ({scored} methods)")
            ckpt["partial"] = False
            ckpt["complete"] = True
            ckpt["final_eval"] = os.path.basename(global_eval_file)
            _save_global_checkpoint(checkpoint_path, ckpt)
        else:
            print(
                "Error in global evaluation: no methods scored completely "
                f"(checkpoint may still have partials at {checkpoint_path})"
            )
            # Incomplete global scoring must not look like success to sweep scripts
            # (previously returned 0 after printing this error).
            raise SystemExit(4)
    
    else:
        print("No global sensemaking questions found")
    print("\n" + "=" * 70)
    print("STEP 5: STANDARD METHOD COMPARISON")
    print("=" * 70)
    
    # Standard comparison for non-action questions that have been answered
    non_action_questions = [q for q in questions 
                           if q.get('category') != 'action_specific' and q.get('type') != 'global' and q.get('id') in answered_question_ids]
    
    if non_action_questions:
        print(f"Comparing methods on answered standard questions (n={len(non_action_questions)})")
        arms = {
            "vanilla": vanilla_answers,
            "textchunkrag": textchunkrag_answers,
            "graphrag": graphrag_answers,
        }
        standard_comparison = {
            "timestamp": timestamp,
            "questions_evaluated": len(non_action_questions),
            "comparisons": {},
        }
        any_ok = False
        for a, b in (("vanilla", "textchunkrag"), ("vanilla", "graphrag"), ("textchunkrag", "graphrag")):
            if not arms[a] or not arms[b]:
                print(f"SKIP standard {a} vs {b} (missing arm)")
                continue
            cmp = evaluator.compare_answer_methods(
                arms[a], arms[b], non_action_questions, a, b
            )
            key = f"{a}_vs_{b}"
            standard_comparison[key] = cmp
            standard_comparison["comparisons"][key] = cmp
            any_ok = True
        if any_ok:
            standard_eval_file = f"{output_dir}/standard_method_comparison_{timestamp}.json"
            evaluator.save_evaluation_results(standard_comparison, standard_eval_file)
            print(f"✓ Standard method comparison saved to {standard_eval_file}")
        else:
            print("No pairwise standard comparisons (need ≥2 method arms)")
        
        print("\nStandard Evaluation Summary:")
        print("=" * 50)
        print(f"Questions Evaluated: {len(non_action_questions)}")
        print("\nPairwise Standard Comparisons:")
        for key, cmp in standard_comparison.get("comparisons", {}).items():
            summary = cmp.get("evaluation_summary", "N/A") if isinstance(cmp, dict) else "N/A"
            print(f"  {key}: {summary}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)

    # Run-level provenance snapshot (paper-parity audit trail)
    try:
        prov_path = f"{output_dir}/run_provenance_{timestamp}.json"
        with open(prov_path, "w") as f:
            json.dump(evaluator.get_model_provenance(), f, indent=2)
        print(f"✓ Run provenance → {prov_path}")
    except Exception as e:
        print(f"Warning: could not write run provenance: {e}")

    # Imputed judge scores compress arm differences, so say so where it will be read
    # rather than only in the provenance file.
    parse_warning = evaluator.warn_if_scores_were_imputed()
    if parse_warning:
        print(parse_warning)
    
    print("Generated Files:")
    #print(f"  - {question_eval_file}")
    if action_eval_file:
        print(f"  - {action_eval_file}")
    if global_eval_file:
        print(f"  - {global_eval_file}")
    if standard_eval_file:
        print(f"  - {standard_eval_file}")
    
    print("\nEvaluation Summary:")
    print(f"  Total Questions: {len(questions)}")
    print(f"  Action-Specific (w/ Ground Truth): {len(action_questions) if 'action_questions' in locals() else 0}")
    print(f"  Global Sensemaking: {len(global_questions) if 'global_questions' in locals() else 0}")
    print(f"  Standard Questions: {len(non_action_questions) if 'non_action_questions' in locals() else 0}")
    print(f"  Vanilla Answers: {len(vanilla_answers)}")
    print(f"  TextChunk RAG Answers: {len(textchunkrag_answers)}")
    print(f"  GraphRAG Answers: {len(graphrag_answers)}")
    print("\nThree-way evaluation completed across all question types!")

def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation for aviation sensemaking QA"
    )
    parser.add_argument(
        '--questions-file',
        required=True,
        help='Path to questions file (JSON format). Required.'
    )
    parser.add_argument(
        '--answers-file',
        required=True,
        help='Path to answers file (JSON format). Required.'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for evaluation results. Required.'
    )
    parser.add_argument(
        '--evaluation-model',
        help='Pinned judge ID (default: claude-haiku-4-5-20251001; --final → claude-sonnet-5)'
    )
    parser.add_argument(
        '--provider',
        default=None,
        choices=["claude_cli", "anthropic", "openai", "huggingface"],
        help='Default: claude_cli (paper-parity / Pro-Max). anthropic=Messages API. openai=legacy.'
    )
    parser.add_argument(
        '--judge-grade',
        default=None,
        choices=["paper", "smoke", "legacy"],
        help='paper=reportable (default for claude_cli/anthropic); smoke=pipeline note; legacy=GPT'
    )
    parser.add_argument(
        '--final',
        action='store_true',
        help='Use final-tier judge model (claude-sonnet-5 / sonnet / gpt-5.6-terra)'
    )
    parser.add_argument(
        '--API-provider',
        default="nebius",
        help='API provider for HuggingFace InferenceClient (only used when provider=huggingface, default: nebius)'
    )
    
    args = parser.parse_args()
    
    try:
        run_comprehensive_evaluation(
            questions_file=args.questions_file,
            answers_file=args.answers_file,
            output_dir=args.output_dir,
            evaluation_model=args.evaluation_model,
            provider=args.provider,
            API_provider=args.API_provider,
            judge_grade=args.judge_grade,
            final=args.final,
        )
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
