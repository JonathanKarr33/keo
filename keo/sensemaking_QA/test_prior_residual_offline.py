#!/usr/bin/env python3
"""Offline checks for prior-residual utility and the leave-one-out router."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prior_residual import (  # noqa: E402
    citation_rate,
    copy_utility,
    entity_yield,
    loo_route,
    paired_signflip,
    partial_spearman,
    prior_echo,
    residual_utility,
    seed_prior_contamination,
)


def test_echo_ignores_question_words():
    question = "What causes leakage during maintenance?"
    prior = "Leakage during maintenance is caused by worn seals and corrosion."
    answer = "Leakage during maintenance is caused by worn seals and corrosion."
    echo = prior_echo(answer, prior, question)
    assert echo is not None and echo > 0.8, echo
    moved = prior_echo(
        "The dataset shows fuel-cap siphoning after gravity flow was blocked.",
        prior,
        question,
    )
    assert moved is not None and moved < echo, (moved, echo)
    print("  echo is high when the answer restates the closed book")


def test_contamination_and_citation_drop_question_only_entities():
    question = "What causes leakage?"
    prior = "Worn seals and corrosion cause repeated failures."
    # "leakage" is only in the question, so this seed is uninformative, not a hit.
    pi = seed_prior_contamination(["LEAKAGE"], prior, question)
    assert pi is None, pi
    pi_hit = seed_prior_contamination(["WORN SEALS", "SPAR CRACK"], prior, question)
    assert pi_hit is not None and 0.3 < pi_hit < 0.8, pi_hit
    cited = citation_rate(
        ["SPAR CRACK", "WORN SEALS"],
        "The records point to a spar crack.",
        question,
    )
    assert cited is not None and abs(cited - 0.5) < 1e-9, cited
    print("  question-only entities are dropped; prior hits and citations are partial")


def test_utility_is_zero_when_the_prior_already_says_it():
    prior = "Fuel contamination causes worn seals and leakage."
    question = "What causes fuel contamination leakage?"
    idf = {"fuel": 1.2, "contamination": 1.5, "worn": 1.4, "seals": 1.3, "spar": 2.0, "crack": 2.2}
    inside = residual_utility(
        "FUEL CONTAMINATION", "causes", "WORN SEALS", question, prior, idf, severity=2.0,
    )
    outside = residual_utility(
        "SPAR CRACK", "causes", "FUEL CONTAMINATION", question, prior, idf, severity=2.0,
    )
    assert inside == 0.0, inside
    assert outside > inside, (outside, inside)
    empty = residual_utility("a", "b", "c", question, prior, idf, severity=3.0)
    assert empty == 0.0, empty
    print("  utility is zero inside the prior and zero on empty edges")


def test_loo_router_recovers_a_real_threshold():
    # High feature → chunk is better. A router must beat always-graph.
    feature = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    keo = [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
    chunk = [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    out = loo_route(feature, keo, chunk)
    assert out["policy_mean"] > out["always_keo"] + 0.5, out
    assert out["policy_mean"] > out["always_chunk"] + 0.5, out
    assert out["gap_closed"] is not None and out["gap_closed"] > 0.5, out
    print(f"  LOO closes {out['gap_closed']:.2f} of the oracle gap when the threshold is real")


def test_loo_abstains_when_the_feature_is_noise():
    feature = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    keo = [2.0] * 8
    chunk = [3.0] * 8  # chunk dominates everywhere; no threshold helps
    out = loo_route(feature, keo, chunk)
    assert abs(out["policy_mean"] - 3.0) < 1e-9, out
    assert out["folds_abstained"] == 8, out
    print("  LOO abstains to the better constant arm when the feature carries nothing")


def test_copy_utility_prefers_short_severe_names():
    severe = copy_utility("ENGINE FIRE")
    mild_long = copy_utility("ROUTINE ANNUAL INSPECTION OF THE CABIN")
    assert severe > mild_long, (severe, mild_long)
    assert copy_utility("x") == 0.0
    print("  copy utility ranks a short severe name above a long mild one")


def test_entity_yield_is_the_retrieval_surplus():
    def ans(qid, text, entities):
        return {
            "question_id": qid,
            "question": "What failed?",
            "answer": text,
            "evidence": {"seed_nodes": [{"id": e, "sim": 0.4} for e in entities], "triples": []},
        }

    answers = {
        "q1": ans("q1", "The report describes an engine fire after takeoff.", ["ENGINE FIRE"]),
        "q2": ans("q2", "The report describes a spar crack on the wing.", ["SPAR CRACK"]),
        "q3": ans("q3", "The report describes fuel contamination in the tank.", ["FUEL CONTAMINATION"]),
        "q4": ans("q4", "The report describes a gear collapse on landing.", ["GEAR COLLAPSE"]),
        "q5": ans("q5", "The report describes oil starvation in cruise.", ["OIL STARVATION"]),
        "q6": ans("q6", "The report describes a rotor strike on shutdown.", ["ROTOR STRIKE"]),
        "q7": ans("q7", "The report describes brake failure during rollout.", ["BRAKE FAILURE"]),
        "q8": ans("q8", "The report describes a propeller strike on the runway.", ["PROPELLER STRIKE"]),
    }
    scores = {
        q: {"overall": 3.0, "global_perspective": 3, "theme_identification": 3,
            "synthesis_quality": 3, "strategic_value": 3, "pattern_recognition": 3}
        for q in answers
    }
    out = entity_yield(answers, scores, list(answers), seed=0)
    assert out["lift"]["mean"] > 0.5, out["lift"]
    assert out["mean_citation_own"] > out["mean_citation_cross"]
    print(f"  entity yield separates own names from a derangement (Y={out['lift']['mean']:.2f})")


def test_partial_spearman_removes_a_shared_confound():
    z = [i / 10.0 for i in range(40)]
    x = [(-1) ** i * (i % 7) for i in range(40)]
    y = [zi + 0.01 * ((-1) ** i) for i, zi in enumerate(z)]
    cleared = abs(partial_spearman(x, y, z))
    assert cleared < 0.35, cleared
    kept = partial_spearman(list(range(40)), [i + 0.01 for i in range(40)], x)
    assert kept > 0.95, kept
    print("  partial Spearman drops a pure confound and keeps a real association")


def test_signflip_refuses_a_tiny_sample():
    try:
        paired_signflip([0.1, 0.2])
        raise AssertionError("should have refused")
    except ValueError as e:
        assert "8" in str(e)
    print("  sign-flip refuses a sample under 8")


if __name__ == "__main__":
    test_echo_ignores_question_words()
    test_contamination_and_citation_drop_question_only_entities()
    test_utility_is_zero_when_the_prior_already_says_it()
    test_loo_router_recovers_a_real_threshold()
    test_loo_abstains_when_the_feature_is_noise()
    test_copy_utility_prefers_short_severe_names()
    test_entity_yield_is_the_retrieval_surplus()
    test_partial_spearman_removes_a_shared_confound()
    test_signflip_refuses_a_tiny_sample()
    print("prior residual offline: 9 checks passed")
