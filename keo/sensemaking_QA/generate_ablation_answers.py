#!/usr/bin/env python3
"""
Ablation-aware answer generation for KEO revision experiments.

Methods:
  vanilla      - no retrieval
  textchunk    - text-chunk RAG only
  keo          - graph + community (default paper method); honors --use-mst / --edge-weight
  community    - community summaries only (GraphRAG-style supporting baseline)
  hybrid       - graph + community + chunk retrieval (K2A analysis)
  routed       - task router picks keo / textchunk / hybrid per question

Mechanism controls (papers/ACL_STRATEGY.md §4) — isolate *why* graph context helps:
  chunk_tok    - chunk RAG at KEO's per-question character budget (length confound)
  chunk_sum    - chunk RAG + LLM cross-record summary (aggregation, no structure)
  chunk_dedup  - chunk RAG with near-duplicate chunks removed (redundancy confound)
  keo_shuf     - KEO on an edge-permuted KG (entity coverage, no true structure)
  keo_norel    - KEO with relation labels stripped (connectivity, no relation semantics)

Each control is run as its own arm into its own output dir, then judged against
keo / textchunk with paired tests.

Examples:
  python generate_ablation_answers.py \\
    --question-files output/aviation_sensemaking_questions.json \\
    --output-file evaluation_results/ablations/mst_on_freq_smoke/answers.json \\
    --method keo --use-mst --edge-weight freq --sample-size 3

  # Local vLLM / Ollama ($0 LLM) + local embeddings:
  python generate_ablation_answers.py \\
    --question-files output/aviation_sensemaking_questions.json \\
    --output-file evaluation_results/ablations/local_qwen/answers.json \\
    --method routed --provider local --base-url http://127.0.0.1:8000/v1 \\
    --answer-model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
    --embedding-backend local --edge-weight rarity
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from answer_generator import SensemakingAnswerGenerator
from data_analyzer import AviationDataAnalyzer
from openrouter_client import PaidAPIInterrupt


CORE_METHODS = ("vanilla", "textchunk", "keo", "community", "hybrid", "routed")

# Proposed selection policies: same prompt surface as keo, different edge selection.
# "risk" is the Pivot C lead arm (risk-weighted budgeted coverage + reserved risk floor).
SELECTION_METHODS: Dict[str, Dict] = {
    "risk": {
        "field": "risk_answer",
        "needs_kg": True,
        "selector": "risk",
    },
}

# Mechanism controls: which generator arm to run, which flags to flip, needs KG?
CONTROL_METHODS: Dict[str, Dict] = {
    "chunk_tok": {
        "field": "chunk_tok_answer",
        "needs_kg": True,  # KEO context length defines the matched budget
        "flags": {"match_context_budget": True},
    },
    "chunk_sum": {
        "field": "chunk_sum_answer",
        "needs_kg": False,
        "flags": {},
    },
    "chunk_dedup": {
        "field": "chunk_dedup_answer",
        "needs_kg": False,
        "flags": {"dedup_chunks": True},
    },
    "keo_shuf": {
        "field": "keo_shuf_answer",
        "needs_kg": True,
        "flags": {"shuffle_edges": True},
    },
    "keo_norel": {
        "field": "keo_norel_answer",
        "needs_kg": True,
        "flags": {"strip_relations": True},
    },
}

METHOD_CHOICES = CORE_METHODS + tuple(SELECTION_METHODS) + tuple(CONTROL_METHODS) + ("all",)
ANSWER_FIELDS = (
    "vanilla_answer",
    "textchunkrag_answer",
    "keo_answer",
    "community_answer",
    "hybrid_answer",
    "routed_answer",
    "graphrag_answer",
) + tuple(spec["field"] for spec in SELECTION_METHODS.values()) + tuple(
    spec["field"] for spec in CONTROL_METHODS.values()
)

FIELD_MAP = {
    "vanilla": "vanilla_answer",
    "textchunk": "textchunkrag_answer",
    "keo": "keo_answer",
    "community": "community_answer",
    "hybrid": "hybrid_answer",
    "routed": "routed_answer",
    **{name: spec["field"] for name, spec in SELECTION_METHODS.items()},
    **{name: spec["field"] for name, spec in CONTROL_METHODS.items()},
}



def _atomic_json(path: str, obj) -> None:
    """Write JSON atomically so a crash mid-dump does not truncate a paid checkpoint."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _usable_blob(blob) -> bool:
    return (
        isinstance(blob, dict)
        and not blob.get("error")
        and bool(str(blob.get("answer") or "").strip())
    )


def _method_field(method: str) -> str:
    if method == "all":
        return ""
    return FIELD_MAP.get(method, "")


def _extract_prior_method_answers(records, field: str) -> List[Dict]:
    out: List[Dict] = []
    if not field or not isinstance(records, list):
        return out
    for record in records:
        if not isinstance(record, dict):
            continue
        blob = record.get(field)
        if _usable_blob(blob):
            out.append(blob)
        elif field == "keo_answer":
            alt = record.get("graphrag_answer")
            if _usable_blob(alt):
                out.append(alt)
    return out


def _arm_is_complete(
    records, field: str, expected_n: int, meta: Optional[Dict]
) -> bool:
    """True only when this method has enough usable answers and is not marked partial."""
    if not field or expected_n <= 0:
        return False
    if isinstance(meta, dict) and meta.get("partial") is True:
        return False
    prior = _extract_prior_method_answers(records, field)
    return len(prior) >= expected_n


def _usable_answer_count(answers_path: str) -> int:
    """
    Count usable answer blobs across any method field in an answers.json.

    Kept for offline guards / callers that only need "is this file empty of
    successes?". Complete-vs-partial skip uses `_arm_is_complete` instead.
    """
    records = _load_json(answers_path)
    if not isinstance(records, list):
        return 0
    usable = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        for field in ANSWER_FIELDS:
            if _usable_blob(record.get(field)):
                usable += 1
    return usable


def _meta_answer_model(meta_path: str) -> Optional[str]:
    """Return the answer model recorded alongside an existing arm, if readable."""
    meta = _load_json(meta_path)
    if not isinstance(meta, dict):
        return None
    return meta.get("answer_model")


def _load_questions(question_files: List[str]) -> List[Dict]:
    for path in question_files:
        if os.path.exists(path):
            with open(path, "r") as f:
                questions = json.load(f)
            print(f"Loaded {len(questions)} questions from {path}")
            return questions
    raise FileNotFoundError(f"No question files found among: {question_files}")



def _write_arm_outputs(
    *,
    output_file: str,
    meta_path: str,
    all_results: List[Dict],
    answer_generator: SensemakingAnswerGenerator,
    method: str,
    use_mst: bool,
    edge_weight: str,
    embedding_backend: str,
    embedding_model: Optional[str],
    provider: str,
    API_provider: str,
    base_url: Optional[str],
    answer_model: str,
    kg_path: str,
    partial: bool,
) -> None:
    """Flush answers.json + answers_meta.json (partial=True while an arm is still running)."""
    _atomic_json(output_file, all_results)
    provenance = answer_generator.get_model_provenance()
    meta = {
        "method": method,
        "controls": answer_generator.control_config(),
        "selection": answer_generator.selection_config(),
        "risk_selection_summary": answer_generator.risk_selection_summary(),
        "use_mst": use_mst,
        "edge_weight": edge_weight,
        "embedding_backend": embedding_backend,
        "embedding_model": getattr(answer_generator.embedder, "model", embedding_model),
        "provider": provider,
        "API_provider": API_provider if provider == "huggingface" else None,
        "base_url": base_url if provider in ("local", "openrouter") else None,
        "answer_model": answer_model,
        "answer_model_exact": answer_generator.exact_model_id(),
        "kg_path": kg_path,
        "n_questions": len(all_results),
        "partial": bool(partial),
        "timestamp": datetime.now().isoformat(),
        "model_provenance": provenance,
    }
    _atomic_json(meta_path, meta)
    if not partial:
        rows = []
        for r in all_results:
            row = {
                "question_id": r["question_id"],
                "question": r["question"],
                "category": r["question_category"],
                "type": r["question_type"],
            }
            for key in ANSWER_FIELDS:
                val = r.get(key)
                row[key] = val.get("answer", "") if isinstance(val, dict) else ""
            rows.append(row)
        summary_file = os.path.splitext(output_file)[0] + "_summary.csv"
        pd.DataFrame(rows).to_csv(summary_file, index=False)


def generate_ablation_answers(
    question_files: List[str],
    output_file: str,
    method: str = "keo",
    sample_size: Optional[int] = None,
    answer_model: Optional[str] = None,
    kg_path: str = "../kg/output/knowledge_graph.gml",
    provider: str = "huggingface",
    API_provider: str = "nebius",
    use_mst: bool = True,
    edge_weight: str = "freq",
    embedding_backend: str = "local",
    embedding_model: Optional[str] = None,
    seed: int = 42,
    base_url: Optional[str] = None,
    skip_if_exists: bool = True,
    no_completion_cache: bool = False,
    max_answer_tokens: Optional[int] = 600,
    corpus_csv: Optional[str] = None,
    text_column: Optional[str] = None,
    control_seed: int = 13,
    risk_budget_tokens: int = 1200,
    risk_floor: float = 0.25,
    risk_fill_budget: bool = True,
) -> List[Dict]:
    random.seed(seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("KEO ABLATION ANSWER GENERATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(
        f"method={method} use_mst={use_mst} edge_weight={edge_weight} "
        f"emb={embedding_backend} provider={provider} model={answer_model}"
    )
    control_flags: Dict = dict(CONTROL_METHODS.get(method, {}).get("flags", {}))
    if control_flags:
        print(f"mechanism control: {method} → {control_flags}")

    selector = SELECTION_METHODS.get(method, {}).get("selector", "mst" if use_mst else "subgraph")
    if selector == "risk":
        print(
            f"selection policy: risk-weighted budgeted coverage "
            f"(B={risk_budget_tokens} tokens, risk_floor={risk_floor}, "
            f"fill_budget={risk_fill_budget})"
        )

    if answer_model is None:
        if provider == "openai":
            answer_model = "gpt-5.6-luna"  # legacy closed
        elif provider == "anthropic":
            from anthropic_client import DEFAULT_ANSWER_MODEL
            answer_model = DEFAULT_ANSWER_MODEL
        elif provider == "local":
            answer_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        elif provider == "openrouter":
            from openrouter_client import openrouter_model
            answer_model = openrouter_model()
        else:
            answer_model = "Qwen/Qwen3.8-27B"

    # Early complete-skip (before KG/dataset load) when meta says the arm finished.
    # Partial checkpoints must NOT skip — those resume after load (paid OpenRouter).
    prior_method_answers: List[Dict] = []
    meta_path = os.path.splitext(output_file)[0] + "_meta.json"
    method_field = _method_field(method)
    if skip_if_exists and os.path.exists(output_file) and method != "all":
        meta = _load_json(meta_path) if os.path.exists(meta_path) else None
        records = _load_json(output_file)
        prior_model = meta.get("answer_model") if isinstance(meta, dict) else None
        if prior_model and prior_model != answer_model:
            print(
                f"Found {output_file} generated with {prior_model}, but this run "
                f"requests {answer_model}; regenerating instead of skipping "
                "(judged scores are not comparable across answer models)."
            )
        elif not isinstance(meta, dict):
            print(
                f"Found {output_file} but missing {meta_path}; "
                "will regenerate/resume with provenance restored."
            )
            if isinstance(records, list) and method_field:
                prior_method_answers = _extract_prior_method_answers(records, method_field)
        else:
            expected = int(meta.get("n_questions") or 0)
            if expected and _arm_is_complete(records, method_field, expected, meta):
                print(f"SKIP (complete): {output_file}  — pass --force to regenerate")
                return records if isinstance(records, list) else []
            prior_method_answers = _extract_prior_method_answers(records, method_field)
            if prior_method_answers:
                print(
                    f"RESUME: {output_file} has {len(prior_method_answers)} usable "
                    f"{method} answers "
                    f"(partial={meta.get('partial')}, n_questions={expected or '?'}); "
                    "keeping paid work and filling the rest."
                )
            else:
                print(
                    f"Found {output_file} but no usable {method} answers; regenerating."
                )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
    elif provider == "huggingface":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN not set")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
    elif provider == "local":
        api_key = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
    elif provider == "openrouter":
        from openrouter_client import openrouter_api_key
        api_key = openrouter_api_key()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if corpus_csv:
        csv_path = corpus_csv if os.path.isabs(corpus_csv) else os.path.normpath(
            os.path.join(os.getcwd(), corpus_csv)
        )
        data_paths = {"corpus": csv_path}
    else:
        repo_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
        data_paths = {
            "maintenance_remaining": os.path.join(
                repo_root,
                "OMIn_dataset/data/FAA_data/sampled_for_kg/Maintenance_remaining_for_questions.csv",
            ),
            "aircraft_annotation": os.path.join(
                repo_root,
                "OMIn_dataset/data/MaintNet_data/Aircraft_Annotation_DataFile.csv",
            ),
        }
    analyzer = AviationDataAnalyzer(data_paths)
    analyzer.load_datasets()
    datasets = analyzer.datasets

    cache_root = os.path.join(script_dir, "evaluation_results", ".completion_cache")
    answer_generator = SensemakingAnswerGenerator(
        api_key,
        model=answer_model,
        provider=provider,
        API_provider=API_provider,
        use_mst=use_mst,
        edge_weight=edge_weight,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        emit_evidence=True,
        base_url=base_url,
        completion_cache_dir=os.path.normpath(cache_root),
        use_completion_cache=not no_completion_cache,
        max_answer_tokens=max_answer_tokens,
        text_column=text_column,
        control_seed=control_seed,
        selector=selector,
        risk_budget_tokens=risk_budget_tokens,
        risk_floor=risk_floor,
        risk_fill_budget=risk_fill_budget,
        **control_flags,
    )

    needs_kg = method in ("keo", "community", "hybrid", "routed", "all", "graphrag") or bool(
        CONTROL_METHODS.get(method, {}).get("needs_kg")
    ) or bool(SELECTION_METHODS.get(method, {}).get("needs_kg"))
    kg_loaded = False
    if needs_kg:
        if os.path.exists(kg_path):
            kg_loaded = answer_generator.load_knowledge_graph(kg_path)
        if not kg_loaded:
            raise RuntimeError(f"Failed to load KG at {kg_path} (required for method={method})")

    questions = _load_questions(question_files)
    if sample_size is not None and 0 < sample_size < len(questions):
        questions = random.sample(questions, sample_size)
        print(f"Sampled {len(questions)} questions")

    # "all" covers the core arms only; controls change generator state and are run one per dir
    methods_to_run = list(CORE_METHODS) if method == "all" else [method]
    method_answers: Dict[str, List[Dict]] = {}

    # Align by question id up front so progress checkpoints can flush a valid arm file.
    by_id: Dict[str, Dict] = {}
    for q in questions:
        qid = q.get("id", "")
        by_id[qid] = {
            "question_id": qid,
            "question": q.get("question", ""),
            "question_category": q.get("category", ""),
            "question_type": q.get("type", ""),
            "ground_truth": (
                q.get("ground_truth")
                or q.get("ground_truth_answer")
                or q.get("answer")
            ),
            "original_problem": q.get("original_problem"),
            "ablation": {
                "method": method,
                "controls": answer_generator.control_config(),
                "selection": answer_generator.selection_config(),
                "use_mst": use_mst,
                "edge_weight": edge_weight,
                "embedding_backend": embedding_backend,
                "embedding_model": getattr(answer_generator.embedder, "model", embedding_model),
                "provider": provider,
                "answer_model": answer_model,
                "answer_model_exact": answer_generator.exact_model_id(),
                "kg_path": kg_path,
                "corpus_csv": corpus_csv,
                "text_column": text_column,
            },
            **{field: None for field in ANSWER_FIELDS},
        }

    # Re-bind prior answers against the sampled question set (drop orphans).
    if prior_method_answers and method_field and method != "all":
        qids = set(by_id)
        prior_method_answers = [
            a for a in prior_method_answers
            if isinstance(a, dict) and a.get("question_id") in qids
        ]
        # If early skip used meta.n_questions from a different sample, require full cover now.
        if (
            skip_if_exists
            and len(prior_method_answers) >= len(questions)
            and method_field
        ):
            # Fill by_id and return without spending API.
            for ans in prior_method_answers:
                qid = ans.get("question_id", "")
                if qid in by_id:
                    by_id[qid][method_field] = ans
                    if method == "keo":
                        by_id[qid]["graphrag_answer"] = ans
            all_results = list(by_id.values())
            _write_arm_outputs(
                output_file=output_file,
                meta_path=meta_path,
                all_results=all_results,
                answer_generator=answer_generator,
                method=method,
                use_mst=use_mst,
                edge_weight=edge_weight,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                provider=provider,
                API_provider=API_provider,
                base_url=base_url,
                answer_model=answer_model,
                kg_path=kg_path,
                partial=False,
            )
            answer_generator.cleanup()
            print(f"SKIP (complete after sample align): {output_file}")
            return all_results

    def _apply_method_answers(m: str, answers: List[Dict]) -> None:
        field = FIELD_MAP[m]
        for ans in answers:
            qid = ans.get("question_id", "")
            if qid in by_id:
                by_id[qid][field] = ans
                if m == "keo":
                    by_id[qid]["graphrag_answer"] = ans

    def _checkpoint(m: str, answers_so_far: List[Dict], *, partial: bool) -> None:
        _apply_method_answers(m, answers_so_far)
        _write_arm_outputs(
            output_file=output_file,
            meta_path=meta_path,
            all_results=list(by_id.values()),
            answer_generator=answer_generator,
            method=method,
            use_mst=use_mst,
            edge_weight=edge_weight,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            provider=provider,
            API_provider=API_provider,
            base_url=base_url,
            answer_model=answer_model,
            kg_path=kg_path,
            partial=partial,
        )

    def _gen_kwargs(m: str) -> Dict:
        # Only the active single-method arm resumes; "all" regenerates each sub-arm.
        if method == "all" or m != method:
            return {}
        return {
            "prior_answers": prior_method_answers,
            "on_progress": (lambda ans, _m=m: _checkpoint(_m, ans, partial=True)),
        }

    try:
        for m in methods_to_run:
            kw = _gen_kwargs(m)
            if m == "vanilla":
                method_answers[m] = answer_generator.generate_vanilla_answers(questions, **kw)
            elif m == "textchunk":
                method_answers[m] = answer_generator.generate_textchunkrag_answers(
                    questions, datasets, **kw
                )
            elif m == "keo":
                method_answers[m] = answer_generator.generate_graphrag_answers(
                    questions, datasets,
                    include_graph=True, include_community=True, include_chunks=False,
                    method_label="keo",
                    **kw,
                )
            elif m == "community":
                method_answers[m] = answer_generator.generate_graphrag_answers(
                    questions, datasets,
                    include_graph=False, include_community=True, include_chunks=False,
                    method_label="community",
                    **kw,
                )
            elif m == "hybrid":
                method_answers[m] = answer_generator.generate_graphrag_answers(
                    questions, datasets,
                    include_graph=True, include_community=True, include_chunks=True,
                    method_label="hybrid",
                    **kw,
                )
            elif m == "routed":
                if not kg_loaded and os.path.exists(kg_path):
                    kg_loaded = answer_generator.load_knowledge_graph(kg_path)
                method_answers[m] = answer_generator.generate_routed_answers(
                    questions, datasets, **kw
                )
            elif m in ("chunk_tok", "chunk_dedup"):
                method_answers[m] = answer_generator.generate_textchunkrag_answers(
                    questions, datasets, method_label=m, **kw,
                )
            elif m == "chunk_sum":
                method_answers[m] = answer_generator.generate_chunksum_answers(
                    questions, datasets, method_label=m, **kw,
                )
            elif m == "keo_shuf":
                method_answers[m] = answer_generator.generate_graphrag_answers(
                    questions, datasets,
                    include_graph=True, include_community=True, include_chunks=False,
                    method_label="keo_shuf",
                    **kw,
                )
            elif m == "keo_norel":
                method_answers[m] = answer_generator.generate_graphrag_answers(
                    questions, datasets,
                    include_graph=True, include_community=True, include_chunks=False,
                    method_label="keo_norel",
                    **kw,
                )
            elif m == "risk":
                method_answers[m] = answer_generator.generate_graphrag_answers(
                    questions, datasets,
                    include_graph=True, include_community=True, include_chunks=False,
                    method_label="risk",
                    **kw,
                )
    except PaidAPIInterrupt as exc:
        # Progress callback already flushed partial answers/meta.
        print(f"⚠ Paid API interrupt — checkpoint saved → {output_file}")
        print(f"  {exc}")
        answer_generator.cleanup()
        raise

    for m, answers in method_answers.items():
        _apply_method_answers(m, answers)

    all_results = list(by_id.values())
    _write_arm_outputs(
        output_file=output_file,
        meta_path=meta_path,
        all_results=all_results,
        answer_generator=answer_generator,
        method=method,
        use_mst=use_mst,
        edge_weight=edge_weight,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        provider=provider,
        API_provider=API_provider,
        base_url=base_url,
        answer_model=answer_model,
        kg_path=kg_path,
        partial=False,
    )
    print(
        f"Model provenance: requested={answer_model} "
        f"exact={answer_generator.exact_model_id()} → {meta_path}"
    )
    risk_summary = answer_generator.risk_selection_summary()
    if risk_summary:
        recall = risk_summary["mean_high_severity_recall"]
        print(
            f"Risk selection: {risk_summary['mean_selected_edges']:.1f} edges / "
            f"{risk_summary['mean_tokens_used']:.0f} of {risk_summary['budget_tokens']} tokens "
            f"per question, mean critical recall "
            f"{('n/a' if recall is None else f'{recall:.1%}')} "
            f"over {risk_summary['questions_with_high_severity_candidates']} questions "
            "with high-severity candidates"
        )

    answer_generator.cleanup()
    print(f"Saved answers → {output_file}")
    print(f"Saved meta    → {meta_path}")
    print(f"Saved summary → {os.path.splitext(output_file)[0] + '_summary.csv'}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="KEO ablation answer generation")
    parser.add_argument("--question-files", nargs="+", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--method", default="keo", choices=METHOD_CHOICES)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--kg-path", default="../kg/output/knowledge_graph.gml")
    parser.add_argument(
        "--provider",
        default="huggingface",
        choices=["openai", "anthropic", "huggingface", "local", "openrouter"],
        help="local | openrouter (OPENROUTER_API_KEY; default model google/gemma-4-31b-it) | huggingface | anthropic | openai",
    )
    parser.add_argument("--API-provider", default="nebius")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for --provider local (default http://127.0.0.1:8000/v1)",
    )
    parser.add_argument("--use-mst", dest="use_mst", action="store_true", default=True)
    parser.add_argument("--no-mst", dest="use_mst", action="store_false")
    parser.add_argument(
        "--edge-weight",
        default="freq",
        choices=["freq", "tfidf", "pmi", "query", "rarity"],
    )
    parser.add_argument(
        "--embedding-backend",
        default="local",
        choices=["openai", "local", "hash"],
        help="openai (API), local (sentence-transformers), hash (offline smoke)",
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--control-seed",
        type=int,
        default=13,
        help="Seed for the keo_shuf edge permutation (report ≥3 seeds in the paper)",
    )
    parser.add_argument(
        "--risk-budget-tokens",
        type=int,
        default=1200,
        help="Token budget B for --method risk (default 1200, ≈ KEO's observed context)",
    )
    parser.add_argument(
        "--risk-floor",
        type=float,
        default=0.25,
        help="Fraction of B reserved for high-severity edges in --method risk (0 disables)",
    )
    parser.add_argument(
        "--no-risk-fill-budget",
        dest="risk_fill_budget",
        action="store_false",
        default=True,
        help=(
            "Stop --method risk when coverage saturates instead of filling B. "
            "Shows how little context the objective needs, but leaves the judged "
            "comparison length-confounded."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if --output-file already exists",
    )
    parser.add_argument(
        "--no-completion-cache",
        action="store_true",
        help="Disable on-disk LLM completion cache (default: cache under results/.completion_cache)",
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=600,
        help="Cap completion tokens (default 600; actionable uses min(250, this))",
    )
    parser.add_argument(
        "--corpus-csv",
        default=None,
        help="Optional alternate corpus CSV (overrides OMIn defaults; for second_domain pilot)",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Text column in corpus CSV (default: auto-detect c119/narrative/text)",
    )
    args = parser.parse_args()

    try:

        generate_ablation_answers(
            question_files=args.question_files,
            output_file=args.output_file,
            method=args.method,
            sample_size=args.sample_size,
            answer_model=args.answer_model,
            kg_path=args.kg_path,
            provider=args.provider,
            API_provider=args.API_provider,
            use_mst=args.use_mst,
            edge_weight=args.edge_weight,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            seed=args.seed,
            base_url=args.base_url,
            skip_if_exists=not args.force,
            no_completion_cache=args.no_completion_cache,
            max_answer_tokens=args.max_answer_tokens,
            corpus_csv=args.corpus_csv,
            text_column=args.text_column,
            control_seed=args.control_seed,
            risk_budget_tokens=args.risk_budget_tokens,
            risk_floor=args.risk_floor,
            risk_fill_budget=args.risk_fill_budget,
        )
    except PaidAPIInterrupt:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
