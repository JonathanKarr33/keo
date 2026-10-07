#!/usr/bin/env python3
"""Offline checks: answer generation resume + paid-API interrupt checkpointing."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KEO_ALLOW_MISSING_NLP", "1")

from answer_generator import SensemakingAnswerGenerator  # noqa: E402
from openrouter_client import PaidAPIInterrupt, looks_like_paid_interrupt  # noqa: E402
from generate_ablation_answers import (  # noqa: E402
    _arm_is_complete,
    _extract_prior_method_answers,
)


def test_paid_interrupt_markers():
    assert looks_like_paid_interrupt("Error code: 402 - Payment Required")
    assert looks_like_paid_interrupt("429 Too Many Requests")
    assert looks_like_paid_interrupt("Insufficient credits for this request")
    assert not looks_like_paid_interrupt("connection reset by peer")
    print("  paid-interrupt markers distinguish billing/rate-limit from noise")


def test_partial_arm_is_not_complete():
    records = [
        {"question_id": "q1", "keo_answer": {"question_id": "q1", "answer": "ok"}},
        {"question_id": "q2", "keo_answer": None},
    ]
    assert not _arm_is_complete(records, "keo_answer", expected_n=2, meta={"partial": False})
    assert not _arm_is_complete(records, "keo_answer", expected_n=2, meta={"partial": True})
    records[1]["keo_answer"] = {"question_id": "q2", "answer": "ok2"}
    assert _arm_is_complete(records, "keo_answer", expected_n=2, meta={"partial": False})
    assert not _arm_is_complete(records, "keo_answer", expected_n=2, meta={"partial": True})
    print("  partial meta / missing answers refuse the complete-skip")


def test_resume_skips_prior_question_ids():
    gen = object.__new__(SensemakingAnswerGenerator)
    gen.use_completion_cache = False
    gen._observed_model_ids = set()
    gen.gen_workers = 1
    gen.edge_weight = "freq"
    gen._tls = __import__("threading").local()
    gen._gen_lock = __import__("threading").RLock()
    gen._progress_lock = __import__("threading").Lock()
    seen = []

    def produce(q):
        seen.append(q["id"])
        return {"question_id": q["id"], "question": q["question"], "answer": f"a-{q['id']}", "method": "vanilla_llm"}

    prior = [{"question_id": "q1", "question": "Q1", "answer": "kept", "method": "vanilla_llm"}]
    out = SensemakingAnswerGenerator._iter_generate(
        gen,
        [{"id": "q1", "question": "Q1"}, {"id": "q2", "question": "Q2"}],
        desc="test",
        produce=produce,
        method_label="vanilla_llm",
        prior_answers=prior,
    )
    assert seen == ["q2"], f"resume regenerated {seen}"
    assert [a["question_id"] for a in out] == ["q1", "q2"]
    assert out[0]["answer"] == "kept"
    print("  resume kept q1 and only generated q2")


def test_progress_checkpoint_before_paid_interrupt():
    gen = object.__new__(SensemakingAnswerGenerator)
    gen.use_completion_cache = False
    gen._observed_model_ids = set()
    gen.gen_workers = 1
    gen.edge_weight = "freq"
    gen._tls = __import__("threading").local()
    gen._gen_lock = __import__("threading").RLock()
    gen._progress_lock = __import__("threading").Lock()
    snaps = []

    def produce(q):
        if q["id"] == "q2":
            raise PaidAPIInterrupt("limit")
        return {"question_id": q["id"], "question": q["question"], "answer": "ok", "method": "t"}

    try:
        SensemakingAnswerGenerator._iter_generate(
            gen,
            [{"id": "q1", "question": "Q1"}, {"id": "q2", "question": "Q2"}],
            desc="test",
            produce=produce,
            method_label="t",
            on_progress=lambda s: snaps.append(list(s)),
        )
    except PaidAPIInterrupt:
        pass
    else:
        raise AssertionError("expected PaidAPIInterrupt")
    assert snaps, "on_progress never ran"
    assert any(len(s) >= 1 and s[0]["question_id"] == "q1" for s in snaps)
    print(f"  checkpoint saw {len(snaps[-1])} answer(s) before paid interrupt")


def test_parallel_workers_preserve_all_ids():
    gen = object.__new__(SensemakingAnswerGenerator)
    gen.use_completion_cache = False
    gen._observed_model_ids = set()
    gen.gen_workers = 4
    gen.edge_weight = "freq"
    gen._tls = __import__("threading").local()
    gen._gen_lock = __import__("threading").RLock()
    gen._progress_lock = __import__("threading").Lock()

    def produce(q):
        return {"question_id": q["id"], "question": q["question"], "answer": q["id"], "method": "t"}

    qs = [{"id": f"q{i}", "question": f"Q{i}"} for i in range(20)]
    out = SensemakingAnswerGenerator._iter_generate(
        gen, qs, desc="par", produce=produce, method_label="t"
    )
    assert [a["question_id"] for a in out] == [f"q{i}" for i in range(20)]
    print("  4 workers returned all 20 ids in question order")


def test_extract_prior_ignores_errors():
    records = [
        {"keo_answer": {"question_id": "q1", "answer": "ok"}},
        {"keo_answer": {"question_id": "q2", "answer": "Error", "error": True}},
        {"keo_answer": {"question_id": "q3", "answer": ""}},
    ]
    prior = _extract_prior_method_answers(records, "keo_answer")
    assert [a["question_id"] for a in prior] == ["q1"]
    print("  prior extract drops errors and blank answers")


if __name__ == "__main__":
    from offline_harness import run_suite

    run_suite(
        "generation resume",
        [
            ("paid interrupt markers", test_paid_interrupt_markers),
            ("partial arm not complete", test_partial_arm_is_not_complete),
            ("resume skips prior ids", test_resume_skips_prior_question_ids),
            ("checkpoint before paid interrupt", test_progress_checkpoint_before_paid_interrupt),
            ("parallel workers preserve ids", test_parallel_workers_preserve_all_ids),
            ("extract prior ignores errors", test_extract_prior_ignores_errors),
        ],
    )
