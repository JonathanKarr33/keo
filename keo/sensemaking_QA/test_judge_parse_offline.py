#!/usr/bin/env python3
"""
Tests for judge score parsing (pure functions, no API calls).

Why this matters more than it looks: every unreadable criterion is replaced by 3.0, the
midpoint of the 1-5 scale. Midpoint imputation pulls all arms toward each other, so it
*shrinks* between-arm differences. The mechanism controls are interpreted through null
results ("shuffling edges changes nothing"), which means parse failures manufacture
exactly the finding the paper wants to read. The parser therefore has to be robust, and
whatever it cannot read has to be counted.

Run: KEO_ALLOW_MISSING_NLP=1 python test_judge_parse_offline.py
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import (
    JUDGE_SCALE_MAX,
    JUDGE_SCALE_MIN,
    parse_criterion_scores,
)
from offline_harness import run_suite  # noqa: E402

GLOBAL_CRITERIA = ['Global Perspective', 'Theme Identification', 'Synthesis Quality',
                   'Strategic Value', 'Pattern Recognition']


def test_plain_format_still_parses():
    """The format the real judge emitted must be unchanged by the robustness work."""
    text = (
        "Global Perspective: 1 - Error response provides zero dataset analysis.\n"
        "Theme Identification: 2 - Minimal thematic content.\n"
        "Synthesis Quality: 3 - Some integration across sources.\n"
        "Strategic Value: 4 - Actionable in places.\n"
        "Pattern Recognition: 5 - Strong relational insight.\n"
    )
    scores, report = parse_criterion_scores(text, GLOBAL_CRITERIA)

    assert scores == {
        'Global Perspective': 1.0,
        'Theme Identification': 2.0,
        'Synthesis Quality': 3.0,
        'Strategic Value': 4.0,
        'Pattern Recognition': 5.0,
    }, scores
    assert report["missing"] == [], report
    print("  5/5 parsed from the real judge's format")


def test_markdown_no_longer_breaks_everything():
    """The old case-sensitive startswith missed '**Criterion**: n' — silently."""
    text = (
        "- **Global Perspective**: 4 — broad coverage\n"
        "* `Theme Identification`: 3\n"
        "  **Synthesis Quality:** 5\n"
    )
    scores, report = parse_criterion_scores(text, GLOBAL_CRITERIA)

    assert scores['Global Perspective'] == 4.0, scores
    assert scores['Theme Identification'] == 3.0, scores
    assert scores['Synthesis Quality'] == 5.0, scores
    assert set(report["missing"]) == {'Strategic Value', 'Pattern Recognition'}, report
    print(f"  markdown parsed: {len(report['parsed'])}/5, {len(report['missing'])} absent")


def test_out_of_scale_numbers_are_rejected():
    """A year or a count must not be averaged as a rating."""
    text = (
        "Global Perspective: 2024 records were reviewed\n"
        "Theme Identification: 0 - nothing found\n"
        "Synthesis Quality: 4\n"
    )
    scores, report = parse_criterion_scores(text, GLOBAL_CRITERIA)

    assert 'Global Perspective' not in scores, scores
    assert 'Theme Identification' not in scores, "0 is below the 1-5 scale"
    assert scores['Synthesis Quality'] == 4.0, scores
    rejected = {item["criterion"] for item in report["out_of_range"]}
    assert rejected == {'Global Perspective', 'Theme Identification'}, report
    print(f"  rejected out-of-scale: {sorted(rejected)}")


def test_missing_number_is_reported_not_invented():
    """A mentioned criterion with no digits is recorded, not silently scored."""
    text = "Global Perspective: excellent throughout\nSynthesis Quality: 3\n"
    scores, report = parse_criterion_scores(text, GLOBAL_CRITERIA)

    assert 'Global Perspective' not in scores, scores
    assert report["no_number"] == ['Global Perspective'], report
    print("  'excellent' is not a score; recorded as no_number")


def test_unrelated_prose_does_not_capture_a_number():
    """A criterion word opening a sentence must not pick up a passing number."""
    cases = [
        "Clarity of the reasoning was discussed across 12 separate records.\n",
        "Strategic Value in 2 areas was weak.\n",
    ]
    for text in cases:
        criterion = text.split()[0] if text.startswith("Clarity") else "Strategic Value"
        scores, report = parse_criterion_scores(text, [criterion])
        assert scores == {}, f"captured a number from prose {text!r}: {scores}"
        assert report["missing"] == [criterion], report
    print(f"  {len(cases)} prose lines yielded no score (no delimiter, no number taken)")


def test_later_mention_wins():
    """Judges restate scores in summaries; the trailing value was what the old code kept."""
    text = "Synthesis Quality: 2 - initial read\nSynthesis Quality: 4 - on reflection\n"
    scores, _ = parse_criterion_scores(text, ['Synthesis Quality'])

    assert scores['Synthesis Quality'] == 4.0, scores
    print("  last mention wins, matching prior behaviour")


def test_scale_bounds_are_inclusive():
    """Both endpoints of the rubric are valid scores."""
    text = f"Strategic Value: {JUDGE_SCALE_MIN:.0f}\nPattern Recognition: {JUDGE_SCALE_MAX:.0f}\n"
    scores, report = parse_criterion_scores(text, GLOBAL_CRITERIA)

    assert scores['Strategic Value'] == JUDGE_SCALE_MIN, scores
    assert scores['Pattern Recognition'] == JUDGE_SCALE_MAX, scores
    assert report["out_of_range"] == [], report
    print(f"  {JUDGE_SCALE_MIN:.0f} and {JUDGE_SCALE_MAX:.0f} both accepted")


def test_total_parse_failure_is_visible():
    """The catastrophic case: nothing parses, and the report says so plainly."""
    text = "The answer was quite good overall. I would rate it favourably.\n"
    scores, report = parse_criterion_scores(text, GLOBAL_CRITERIA)

    assert scores == {}, scores
    assert report["missing"] == GLOBAL_CRITERIA, report
    assert len(report["missing"]) == 5, report
    print("  all 5 criteria reported missing instead of becoming 3.0 invisibly")


@contextmanager
def imputation_allowed():
    """Reload evaluator with KEO_ALLOW_JUDGE_IMPUTATION=1, then restore the default."""
    import importlib

    import evaluator

    os.environ["KEO_ALLOW_JUDGE_IMPUTATION"] = "1"
    try:
        yield importlib.reload(evaluator)
    finally:
        os.environ.pop("KEO_ALLOW_JUDGE_IMPUTATION", None)
        importlib.reload(evaluator)


def test_unreadable_comparison_rubric_stops_the_run():
    """The default: an unreadable arm-comparison criterion is an error, not a 3.0."""
    from evaluator import SensemakingEvaluator

    ev = object.__new__(SensemakingEvaluator)
    ev._parse_health = {}
    ev._state_lock = threading.Lock()
    ev.judge_workers = 1
    try:
        ev._parse_global_scores("Global Perspective: excellent\n")
    except RuntimeError as e:
        assert "KEO_ALLOW_JUDGE_IMPUTATION" in str(e), e
        print("  default refuses to substitute a midpoint on the global rubric")
        return
    raise AssertionError("imputed silently by default")


def test_question_quality_is_not_an_arm_comparison():
    """Question scoring rates the benchmark, not methods, so it tolerates a gap."""
    from evaluator import SensemakingEvaluator

    ev = object.__new__(SensemakingEvaluator)
    ev._parse_health = {}
    ev._state_lock = threading.Lock()
    ev.judge_workers = 1
    scores = ev._parse_question_scores("Complexity: unreadable\n")

    assert scores, "question scoring should still return a usable dict"
    health = ev.parse_health_summary()["by_rubric"]["question"]
    assert health["imputation_rate"] > 0, health
    assert health["imputed_by_criterion"], health
    print(f"  tolerated and counted: {health['imputation_rate']:.0%} imputed, "
          f"{sorted(health['imputed_by_criterion'])}")


def test_imputation_rate_is_accumulated():
    """With the override on, parse_health_summary exposes the rate and trips its alarm."""
    with imputation_allowed() as evaluator:
        ev = object.__new__(evaluator.SensemakingEvaluator)
        ev._parse_health = {}
        ev._state_lock = threading.Lock()
        ev.judge_workers = 1

        good = ("Global Perspective: 4\nTheme Identification: 4\nSynthesis Quality: 4\n"
                "Strategic Value: 4\nPattern Recognition: 4\n")
        for _ in range(9):
            ev._parse_global_scores(good)
        ev._parse_global_scores("Global Perspective: unreadable\n")

        summary = ev.parse_health_summary()
        rubric = summary["by_rubric"]["global"]
        assert rubric["responses"] == 10, rubric
        assert rubric["criteria_expected"] == 50, rubric
        assert rubric["criteria_parsed"] == 45, rubric
        assert abs(summary["imputation_rate"] - 0.10) < 1e-9, summary
        assert summary["alarm_threshold"] == evaluator.JUDGE_IMPUTATION_ALARM
        assert summary["allow_imputation_override"] is True, summary
        # Sitting exactly on the threshold does not trip it; the alarm is strictly-greater.
        assert not summary["exceeds_alarm"], summary
        assert ev.warn_if_scores_were_imputed() is None

        ev._parse_global_scores("Global Perspective: unreadable\n")
        assert ev.parse_health_summary()["exceeds_alarm"], "past threshold, alarm silent"
        assert ev.warn_if_scores_were_imputed(), "alarm should produce a warning"
        print(f"  {summary['imputation_rate']:.1%} over {rubric['responses']} responses "
              "is at threshold and silent; one more trips the warning")


def test_override_is_recorded_in_provenance():
    """A run that imputed must not look identical to one that refused."""
    from evaluator import SensemakingEvaluator

    strict = object.__new__(SensemakingEvaluator)
    strict._parse_health = {}
    strict._state_lock = threading.Lock()
    strict.judge_workers = 1
    assert strict.parse_health_summary()["refuses_imputation"] is True

    with imputation_allowed() as evaluator:
        loose = object.__new__(evaluator.SensemakingEvaluator)
        loose._parse_health = {}
        loose._state_lock = threading.Lock()
        loose.judge_workers = 1
        assert loose.parse_health_summary()["refuses_imputation"] is False
    print("  refuses_imputation distinguishes the two run types in provenance")


def main():
    tests = [
        ("plain format parses", test_plain_format_still_parses),
        ("markdown parses", test_markdown_no_longer_breaks_everything),
        ("out-of-scale rejected", test_out_of_scale_numbers_are_rejected),
        ("no-number reported", test_missing_number_is_reported_not_invented),
        ("prose not captured", test_unrelated_prose_does_not_capture_a_number),
        ("later mention wins", test_later_mention_wins),
        ("scale bounds inclusive", test_scale_bounds_are_inclusive),
        ("total failure visible", test_total_parse_failure_is_visible),
        ("comparison rubric stops run", test_unreadable_comparison_rubric_stops_the_run),
        ("question rubric tolerated", test_question_quality_is_not_an_arm_comparison),
        ("imputation rate tracked", test_imputation_rate_is_accumulated),
        ("override in provenance", test_override_is_recorded_in_provenance),
    ]
    run_suite("judge parse", tests)


if __name__ == "__main__":
    main()
