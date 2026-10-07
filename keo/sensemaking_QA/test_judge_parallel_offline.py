#!/usr/bin/env python3
"""Offline checks on concurrent judging.

Parallelism here is only safe if two things hold: scores stay attached to the question that
produced them, and the shared imputation counters do not lose increments. Both failures are silent
— the run still prints a clean table, with scores on the wrong rows or an imputation rate that
understates how much the judge could not be read.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KEO_ALLOW_MISSING_NLP", "1")

import evaluator as ev_mod  # noqa: E402
from evaluator import SensemakingEvaluator, JudgeParseError, judge_workers  # noqa: E402
from offline_harness import run_suite  # noqa: E402


def bare_evaluator(workers: int) -> SensemakingEvaluator:
    """An evaluator with only the state these checks touch, skipping client setup."""
    e = object.__new__(SensemakingEvaluator)
    e._parse_health = {}
    e._observed_model_ids = set()
    e._state_lock = threading.Lock()
    e.judge_workers = workers
    return e


def test_worker_count_is_bounded():
    for raw, expected in (("1", 1), ("8", 8), ("0", 1), ("-4", 1), ("999", 16), ("abc", 1), ("", 1)):
        os.environ["KEO_JUDGE_WORKERS"] = raw
        got = judge_workers()
        assert got == expected, f"KEO_JUDGE_WORKERS={raw!r} gave {got}, expected {expected}"
    os.environ.pop("KEO_JUDGE_WORKERS", None)
    default = judge_workers()
    assert 1 <= default <= 16, f"default {default} outside the supported range"
    assert default > 1, "default should parallelise; set KEO_JUDGE_WORKERS=1 for a sequential run"
    print(f"  worker count clamps to 1..16, defaults to {default}")


def test_results_keep_input_order():
    """A slow first item must not end up with the last item's score."""
    for workers in (1, 4, 8):
        e = bare_evaluator(workers)
        items = list(range(20))

        def score(i):
            # Invert the durations so completion order is the reverse of input order.
            time.sleep((20 - i) * 0.002)
            return f"score-{i}"

        out = e._map_judge_calls(items, score, "t")
        assert out == [f"score-{i}" for i in items], (
            f"workers={workers}: results came back in completion order, so scores would attach "
            f"to the wrong questions: {out[:4]}"
        )
    print("  order preserved at 1, 4 and 8 workers even when later items finish first")


def test_every_item_is_scored_exactly_once():
    for workers in (1, 6):
        e = bare_evaluator(workers)
        seen = []
        lock = threading.Lock()

        def score(i):
            with lock:
                seen.append(i)
            return i

        out = e._map_judge_calls(list(range(50)), score, "t")
        assert sorted(seen) == list(range(50)), f"workers={workers}: items skipped or repeated"
        assert out == list(range(50)), f"workers={workers}: wrong values returned"
    print("  all 50 items scored exactly once, sequentially and concurrently")


def test_parse_health_counts_survive_concurrency():
    """The imputation rate decides whether a null result is trustworthy, so it must not drop."""
    e = bare_evaluator(8)
    report = {"requested": ["a", "b"], "parsed": ["a", "b"], "no_number": [], "out_of_range": []}

    def record(_):
        e._record_parse_health("global", report, [])
        return 1

    e._map_judge_calls(list(range(200)), record, "t")
    health = e._parse_health["global"]
    assert health["responses"] == 200, f"lost increments: {health['responses']} of 200"
    assert health["criteria_expected"] == 400, f"expected 400, got {health['criteria_expected']}"
    assert health["criteria_parsed"] == 400, f"parsed 400, got {health['criteria_parsed']}"
    print("  200 concurrent recordings kept every increment")


def test_refusal_still_aborts_from_a_worker_thread():
    """Imputation refusal must stop the run even when raised off the main thread."""
    e = bare_evaluator(4)
    bad = {"requested": ["a"], "parsed": [], "no_number": ["a"], "out_of_range": []}

    def record(i):
        if i == 7:
            e._record_parse_health("global", bad, ["a"])
        return i

    try:
        e._map_judge_calls(list(range(20)), record, "t")
    except JudgeParseError:
        pass
    else:
        raise AssertionError(
            "a refused imputation inside a worker was swallowed, so the run would continue "
            "with a silently reduced n"
        )
    print("  JudgeParseError propagates out of the pool and aborts")


def test_override_still_permits_imputation():
    """With the override set, concurrency must not turn a tolerated gap into an abort."""
    e = bare_evaluator(4)
    bad = {"requested": ["a"], "parsed": [], "no_number": ["a"], "out_of_range": []}
    original = ev_mod.ALLOW_JUDGE_IMPUTATION
    ev_mod.ALLOW_JUDGE_IMPUTATION = True
    try:
        e._map_judge_calls(list(range(10)), lambda i: e._record_parse_health("global", bad, ["a"]), "t")
        assert e._parse_health["global"]["imputed"]["a"] == 10, "imputations were not all counted"
    finally:
        ev_mod.ALLOW_JUDGE_IMPUTATION = original
    print("  override path counts imputations under concurrency instead of aborting")


def test_question_rubric_tolerates_gaps_concurrently():
    e = bare_evaluator(4)
    bad = {"requested": ["a"], "parsed": [], "no_number": ["a"], "out_of_range": []}
    e._map_judge_calls(list(range(10)), lambda i: e._record_parse_health("question", bad, ["a"]), "t")
    assert e._parse_health["question"]["imputed"]["a"] == 10
    print("  question-quality rubric still tolerates gaps off the main thread")


def test_quota_exhaustion_is_named_not_mistaken_for_a_format_problem():
    """A usage-limit notice must abort with a quota error, not a rubric-parse error."""
    from claude_cli import _looks_like_quota_exhausted, _looks_throttled

    exhausted = [
        "You've reached your usage limit. Your limit will reset at 9pm.",
        "Claude usage limit reached",
        "quota exceeded for this organization",
        "Upgrade to continue using Claude",
    ]
    for msg in exhausted:
        assert _looks_like_quota_exhausted(msg), f"not recognised as a hard limit: {msg!r}"

    throttled = ["429 Too Many Requests", "rate limit exceeded", "Overloaded"]
    for msg in throttled:
        assert _looks_throttled(msg), f"not recognised as throttling: {msg!r}"

    # A real rubric reply must not trip either check, or every run would abort.
    good = ("Global Perspective: 4 - broad\nTheme Identification: 3 - partial\n"
            "Synthesis Quality: 4\nStrategic Value: 3\nPattern Recognition: 4")
    assert not _looks_like_quota_exhausted(good), "a valid rubric reply read as a quota failure"
    assert not _looks_throttled(good), "a valid rubric reply read as throttling"
    print("  limit notices are recognised and a valid rubric reply is not")


def test_quota_error_is_fatal_and_not_swallowed():
    """Per-item handlers must not absorb a quota stop, which would shrink n silently."""
    from claude_cli import ClaudeQuotaError

    e = bare_evaluator(4)

    def per_item_handler():
        """Mirrors the real call site: broad except, delegating the fatal decision."""
        try:
            raise ClaudeQuotaError("usage limit reached")
        except Exception as exc:  # noqa: BLE001 - deliberately broad, as in the judging loops
            e._reraise_fatal_judge_errors(exc)
            return "swallowed"

    try:
        outcome = per_item_handler()
    except ClaudeQuotaError:
        outcome = "aborted"
    if outcome != "aborted":
        raise AssertionError(
            "a quota stop was swallowed as a per-item error, so the run would continue "
            "judging nothing and report a reduced n as if it were the real sample"
        )
    print("  a quota stop aborts the run instead of quietly dropping answers")


def test_progress_checkpoint_keeps_successes_before_quota():
    """Successful scores are handed to on_progress before ClaudeQuotaError aborts."""
    from claude_cli import ClaudeQuotaError

    GLOBAL_Q = [
        {"id": "q1", "question": "Q1", "type": "global"},
        {"id": "q2", "question": "Q2", "type": "global"},
        {"id": "q3", "question": "Q3", "type": "global"},
    ]
    GLOBAL_A = [
        {"question_id": "q1", "answer": "A1"},
        {"question_id": "q2", "answer": "A2"},
        {"question_id": "q3", "answer": "A3"},
    ]

    e = bare_evaluator(1)
    e.model = "test-judge"
    e._last_call_meta = {}
    seen = []

    def fake_eval(answer, question):
        qid = question.get("id")
        if qid == "q2":
            raise ClaudeQuotaError("limit")
        return {
            "question_id": qid,
            "global_metrics": {
                "global_perspective": 4,
                "theme_identification": 4,
                "synthesis_quality": 4,
                "strategic_value": 4,
                "pattern_recognition": 4,
            },
        }

    e._evaluate_global_sensemaking_answer = fake_eval  # type: ignore[method-assign]

    try:
        e.evaluate_global_sensemaking_capability(
            GLOBAL_Q, GLOBAL_A, on_progress=lambda evs: seen.append(list(evs))
        )
        raise AssertionError("expected ClaudeQuotaError")
    except ClaudeQuotaError:
        pass

    assert seen, "on_progress never ran before quota abort"
    assert {e["question_id"] for e in seen[-1]} == {"q1"}, seen[-1]
    print(f"  checkpoint saw {len(seen[-1])} success(es) before quota abort")


def test_resume_skips_prior_question_ids():
    """prior_evaluations are kept and those question ids are not re-judged."""
    GLOBAL_Q = [
        {"id": "q1", "question": "Q1", "type": "global"},
        {"id": "q2", "question": "Q2", "type": "global"},
    ]
    GLOBAL_A = [
        {"question_id": "q1", "answer": "A1"},
        {"question_id": "q2", "answer": "A2"},
    ]
    prior = [{
        "question_id": "q1",
        "global_metrics": {
            "global_perspective": 3,
            "theme_identification": 3,
            "synthesis_quality": 3,
            "strategic_value": 3,
            "pattern_recognition": 3,
        },
    }]

    e = bare_evaluator(1)
    e.model = "test-judge"
    called = []

    def fake_eval(answer, question):
        called.append(question.get("id"))
        return {
            "question_id": question.get("id"),
            "global_metrics": {
                "global_perspective": 5,
                "theme_identification": 5,
                "synthesis_quality": 5,
                "strategic_value": 5,
                "pattern_recognition": 5,
            },
        }

    e._evaluate_global_sensemaking_answer = fake_eval  # type: ignore[method-assign]
    e._calculate_global_metrics = lambda evals: {  # type: ignore[method-assign]
        "total_evaluated": len(evals),
        "global_perspective": 4.0,
        "theme_identification": 4.0,
        "synthesis_quality": 4.0,
        "strategic_value": 4.0,
        "pattern_recognition": 4.0,
    }

    result = e.evaluate_global_sensemaking_capability(
        GLOBAL_Q, GLOBAL_A, prior_evaluations=prior
    )
    assert called == ["q2"], called
    assert result["global_questions_evaluated"] == 2, result
    assert not result.get("incomplete"), result
    ids = {e["question_id"] for e in result["individual_evaluations"]}
    assert ids == {"q1", "q2"}, ids
    print("  resume kept q1 and only judged q2")


def main():
    run_suite("judge parallel", [
        ("worker count bounded", test_worker_count_is_bounded),
        ("input order preserved", test_results_keep_input_order),
        ("each item scored once", test_every_item_is_scored_exactly_once),
        ("parse health counts survive", test_parse_health_counts_survive_concurrency),
        ("refusal aborts from worker", test_refusal_still_aborts_from_a_worker_thread),
        ("override permits imputation", test_override_still_permits_imputation),
        ("question rubric tolerates gaps", test_question_rubric_tolerates_gaps_concurrently),
        ("quota exhaustion is named", test_quota_exhaustion_is_named_not_mistaken_for_a_format_problem),
        ("quota error is fatal", test_quota_error_is_fatal_and_not_swallowed),
        ("progress checkpoint before quota", test_progress_checkpoint_keeps_successes_before_quota),
        ("resume skips prior ids", test_resume_skips_prior_question_ids),
    ])


if __name__ == "__main__":
    main()
