#!/usr/bin/env python3
"""
Guards that keep unusable answers out of the judge (no API calls, no judge calls).

Motivated by a real smoke run: every generation failed with "Model Qwen/Qwen3.8-27B is
not supported by provider nebius", the failure text was stored in the `answer` field,
and the judge scored those strings. Three arms reported `overall_avg=1.000` — the floor
of the 1-5 scale — which is indistinguishable from a genuinely bad method in every
downstream table.

`load_method_answers` must drop `error: True` answers, count them so the caller can
report them, and leave an all-failed arm empty so the caller aborts.

Run: python test_judge_guards_offline.py
"""

from __future__ import annotations

import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_harness import run_suite  # noqa: E402
from run_evaluation import (
    CHUNK_FAMILY,
    GRAPH_FAMILY,
    METHOD_FIELDS,
    load_method_answers,
)

FAILURE_TEXT = "Error generating answer: Model Qwen/Qwen3.8-27B is not supported by provider nebius."


def failed(qid="q1", text=FAILURE_TEXT):
    """An answer blob as answer_generator writes it when generation raises."""
    return {"question_id": qid, "question": "...", "answer": text,
            "method": "vanilla_llm", "error": True}


def good(qid="q1", text="Fuel contamination recurs across several airframes."):
    return {"question_id": qid, "question": "...", "answer": text, "method": "vanilla_llm"}


def test_all_failed_leaves_arm_empty():
    """The regression: an arm whose every answer failed must present as absent."""
    data = [
        {"question_id": "q1", "vanilla_answer": failed("q1")},
        {"question_id": "q2", "vanilla_answer": failed("q2")},
        {"question_id": "q3", "vanilla_answer": failed("q3")},
    ]
    answers, failures = load_method_answers(data)

    assert answers["vanilla"] == [], answers["vanilla"]
    assert failures["vanilla"] == 3, failures["vanilla"]
    present = {k: v for k, v in answers.items() if v}
    assert present == {}, present
    print(f"  3 failures counted, 0 judgeable → caller aborts (present={present})")


def test_failure_text_never_survives():
    """No judgeable answer may contain the generation error string."""
    data = [
        {"question_id": "q1", "vanilla_answer": failed("q1")},
        {"question_id": "q2", "vanilla_answer": good("q2")},
    ]
    answers, failures = load_method_answers(data)

    texts = [a["answer"] for arm in answers.values() for a in arm]
    assert len(texts) == 1, texts
    assert FAILURE_TEXT not in texts[0], texts
    assert failures["vanilla"] == 1, failures
    print(f"  1 of 2 kept; error string excluded ({texts[0][:40]}...)")


def test_mixed_arms_are_independent():
    """One arm failing must not suppress a healthy arm in the same file."""
    data = [
        {"question_id": "q1", "vanilla_answer": failed("q1"), "keo_answer": good("q1")},
        {"question_id": "q2", "vanilla_answer": failed("q2"), "keo_answer": good("q2")},
    ]
    answers, failures = load_method_answers(data)

    assert answers["vanilla"] == [], answers["vanilla"]
    assert len(answers["keo"]) == 2, answers["keo"]
    assert failures["vanilla"] == 2 and failures["keo"] == 0, failures
    print("  vanilla dropped (2 failures), keo kept (2 answers)")


def test_nulls_are_not_counted_as_failures():
    """Absent arms are normal in single-method files and are not generation failures."""
    data = [
        {"question_id": "q1", "vanilla_answer": good("q1"), "keo_answer": None},
        {"question_id": "q2", "vanilla_answer": good("q2"), "keo_answer": None},
    ]
    answers, failures = load_method_answers(data)

    assert len(answers["vanilla"]) == 2, answers["vanilla"]
    assert answers["keo"] == [], answers["keo"]
    assert sum(failures.values()) == 0, failures
    print("  null arms ignored without inflating the failure count")


def test_every_arm_is_guarded():
    """The guard is field-driven, so assert every registered arm goes through it."""
    for name, field in METHOD_FIELDS:
        data = [{"question_id": "q1", field: failed("q1")}]
        answers, failures = load_method_answers(data)
        assert answers[name] == [], f"{name} judged a failed generation"
        assert failures[name] == 1, f"{name} failure not counted"

    names = [name for name, _ in METHOD_FIELDS]
    assert len(names) == len(set(names)), f"duplicate arm names: {names}"
    for family in (CHUNK_FAMILY, GRAPH_FAMILY):
        for arm in family:
            assert arm in names, f"{arm} is in a family but not in METHOD_FIELDS"
    print(f"  all {len(names)} arms guarded; families consistent")


def test_matches_the_real_smoke_output():
    """Replay the actual failed smoke file, if it is still on disk."""
    path = os.path.join("evaluation_results", "ablations", "vanilla", "answers.json")
    if not os.path.exists(path):
        print("  skipped (smoke output not present)")
        return
    with open(path) as f:
        data = json.load(f)
    answers, failures = load_method_answers(data)
    present = {k: len(v) for k, v in answers.items() if v}
    total_failed = sum(failures.values())
    if total_failed:
        assert present == {}, (
            f"the failed smoke run would still be judged as {present}"
        )
        print(f"  replayed real file: {total_failed} failures, nothing judgeable")
    else:
        print(f"  replayed real file: healthy ({present})")


def test_all_error_arm_is_not_reusable():
    """A fully-failed arm must not count as complete for skip_if_exists."""
    import tempfile

    from generate_ablation_answers import _usable_answer_count

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "answers.json")
        with open(path, "w") as f:
            json.dump([{"question_id": "q1", "vanilla_answer": failed("q1")}], f)
        assert _usable_answer_count(path) == 0, "all-error arm looked complete"

        with open(path, "w") as f:
            json.dump([
                {"question_id": "q1", "vanilla_answer": failed("q1")},
                {"question_id": "q2", "vanilla_answer": good("q2")},
            ], f)
        assert _usable_answer_count(path) == 1, _usable_answer_count(path)

        assert _usable_answer_count(os.path.join(tmp, "missing.json")) == 0
    print("  all-error arm → 0 usable; mixed arm → 1; missing file → 0")


def test_model_change_is_detectable():
    """Mac-quantized smoke answers must not be silently reused for a GPU sweep."""
    import tempfile

    from generate_ablation_answers import _meta_answer_model

    with tempfile.TemporaryDirectory() as tmp:
        meta = os.path.join(tmp, "answers_meta.json")
        with open(meta, "w") as f:
            json.dump({"answer_model": "phi4:14b-q4_K_M",
                       "answer_model_exact": "phi4:14b-q4_K_M@abc123"}, f)
        prior = _meta_answer_model(meta)
        assert prior == "phi4:14b-q4_K_M", prior
        assert prior != "microsoft/Phi-4-reasoning", "mismatch would go undetected"

        assert _meta_answer_model(os.path.join(tmp, "absent.json")) is None
    print("  quantized tag distinguished from the bf16 id; missing meta → None")


def _bare_generator(graph=None, retriever=None):
    """
    A generator with only the attributes the graph guards read.

    Built via __new__ so no client, embedder or provenance lookup is constructed — the
    guards under test run before any of that matters.
    """
    from answer_generator import SensemakingAnswerGenerator

    gen = object.__new__(SensemakingAnswerGenerator)
    gen.knowledge_graph = graph
    gen.graph_retriever = retriever
    return gen


def test_missing_graph_raises_instead_of_returning_empty():
    """The community/hybrid null-arm bug: [] became an arm of nulls with no error flag."""
    gen = _bare_generator(graph=None, retriever=None)
    try:
        result = gen.generate_graphrag_answers([{"id": "q1", "question": "?"}], {})
    except RuntimeError as e:
        assert "not loaded" in str(e), e
        print(f"  raises: {str(e)[:70]}...")
        return
    raise AssertionError(f"returned {result!r} instead of raising")


def test_empty_graph_is_distinguished_from_unloaded():
    """nx.Graph truthiness is node count, so a 0-node graph must not read as unloaded."""
    import networkx as nx

    gen = _bare_generator(graph=nx.Graph(), retriever=object())
    assert not gen.knowledge_graph, "precondition: empty graph is falsy"
    try:
        gen.generate_graphrag_answers([{"id": "q1", "question": "?"}], {})
    except RuntimeError as e:
        assert "empty" in str(e).lower(), f"wrong diagnosis for an empty graph: {e}"
        print(f"  raises: {str(e)[:70]}...")
        return
    raise AssertionError("empty graph accepted")


def test_no_placeholder_context_reaches_the_prompt():
    """Graph/community context must never degrade to a placeholder string."""
    gen = _bare_generator(graph=None, retriever=None)

    for label, call in (
        ("_retrieve_graph_context", lambda: gen._retrieve_graph_context("q", 3, 10)),
        ("_get_community_summaries", lambda: gen._get_community_summaries("q")),
    ):
        try:
            out = call()
        except RuntimeError:
            continue
        raise AssertionError(f"{label} returned placeholder text: {out!r}")
    print("  both context paths raise rather than returning placeholder text")


def global_evaluator(side_effect):
    """An evaluator wired so every judged answer runs side_effect(), no API involved."""
    import evaluator as ev_mod

    ev = object.__new__(ev_mod.SensemakingEvaluator)
    ev._parse_health = {}
    ev._state_lock = threading.Lock()
    ev.judge_workers = 1
    ev._evaluate_global_sensemaking_answer = lambda answer, question: side_effect()
    return ev


GLOBAL_Q = [{"id": "q1", "category": "global"}, {"id": "q2", "category": "global"}]
GLOBAL_A = [good("q1"), good("q2")]


def test_fully_failed_arm_does_not_score_zero():
    """An arm where every judgement raised must report an error, not 0.0 ratings."""
    def boom():
        raise ValueError("judge returned nothing")

    ev = global_evaluator(boom)
    result = ev.evaluate_global_sensemaking_capability(GLOBAL_Q, GLOBAL_A)

    assert result.get("error"), f"fully-failed arm looked valid: {result}"
    assert result["global_questions_evaluated"] == 0, result
    assert result["global_metrics"].get("error"), result["global_metrics"]
    # The original bug: a full set of criterion keys at 0.0, which reads as a real rating.
    fabricated = [k for k in result["global_metrics"]
                  if k not in ("total_evaluated", "error")]
    assert not fabricated, f"fabricated ratings for {fabricated}"
    print(f"  error surfaced instead of zeros: {result['error']}")


def test_parse_refusal_is_not_swallowed_per_item():
    """Refusing to impute must abort, not quietly drop the answer and shrink n."""
    import evaluator as ev_mod

    def refuse():
        raise ev_mod.JudgeParseError("Global Perspective unreadable")

    ev = global_evaluator(refuse)
    try:
        ev.evaluate_global_sensemaking_capability(GLOBAL_Q, GLOBAL_A)
    except ev_mod.JudgeParseError:
        print("  JudgeParseError propagates past the per-answer except Exception")
        return
    raise AssertionError("parse refusal was swallowed as a per-item error")


def test_partial_failure_reports_the_reduced_n():
    """Surviving answers are still scored, but n and the failed ids are recorded."""
    calls = {"n": 0}

    def sometimes():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("truncated judge response")
        return {"global_metrics": {"global_perspective": 4, "theme_identification": 4,
                                   "synthesis_quality": 4, "strategic_value": 4,
                                   "pattern_recognition": 4}}

    ev = global_evaluator(sometimes)
    result = ev.evaluate_global_sensemaking_capability(GLOBAL_Q, GLOBAL_A)

    assert not result.get("error"), result
    assert result["global_questions_evaluated"] == 1, result
    assert result["failed_question_ids"] == ["q1"], result
    assert result["global_metrics"]["total_evaluated"] == 1, result["global_metrics"]
    print(f"  1 of 2 scored, failed ids recorded: {result['failed_question_ids']}")


def main():
    tests = [
        ("all failed → arm empty", test_all_failed_leaves_arm_empty),
        ("failure text never judged", test_failure_text_never_survives),
        ("arms independent", test_mixed_arms_are_independent),
        ("nulls are not failures", test_nulls_are_not_counted_as_failures),
        ("every arm guarded", test_every_arm_is_guarded),
        ("real smoke file", test_matches_the_real_smoke_output),
        ("all-error arm not reusable", test_all_error_arm_is_not_reusable),
        ("answer-model change detected", test_model_change_is_detectable),
        ("missing graph raises", test_missing_graph_raises_instead_of_returning_empty),
        ("empty graph diagnosed", test_empty_graph_is_distinguished_from_unloaded),
        ("no placeholder context", test_no_placeholder_context_reaches_the_prompt),
        ("failed arm not scored 0", test_fully_failed_arm_does_not_score_zero),
        ("parse refusal aborts", test_parse_refusal_is_not_swallowed_per_item),
        ("partial failure reports n", test_partial_failure_reports_the_reduced_n),
    ]
    run_suite("judge guard", tests)


if __name__ == "__main__":
    main()
