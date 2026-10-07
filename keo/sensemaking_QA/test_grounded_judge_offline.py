#!/usr/bin/env python3
"""Offline checks for the grounded judge. No API calls.

Run: python test_grounded_judge_offline.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounded_judge import (  # noqa: E402
    JUDGED_AXES,
    NOT_IN_SOURCE,
    RUBRIC_ID,
    STYLE_NAMES,
    SUPPORTED,
    GroundedJudge,
    GroundedJudgeParseError,
    answer_payload,
    apply_claim_labels,
    build_prompt,
    reconcile_position_swap,
    score_payload,
    source_statements,
)
from judge_styles import HOLISTIC_NAMES, JudgeStyleError, require_style  # noqa: E402
from offline_harness import run_suite  # noqa: E402


EVIDENCE = {
    "triples": [
        {"id": "abc", "head": "FUEL CAP", "relation": "causes", "tail": "WATER IN FUEL"},
        {"id": "def", "head": "OIL DIPSTICK", "relation": "related_to", "tail": "OIL LEAK"},
    ],
    "chunks": [{"id": "chk", "sim": 0.2, "preview": "loose gas cap noted on preflight"}],
}


def _good_payload():
    return {
        "claims": [
            {"text": "A loose fuel cap lets water into the fuel.", "label": "supported", "source_ids": ["S1"], "because": "S1 states it."},
            {"text": "The dipstick was left out.", "label": "contradicted", "source_ids": ["S2"], "because": "S2 does not say left out; it links dipstick to a leak."},
            {"text": "Pilots should train on water recognition.", "label": "not_in_source", "source_ids": [], "because": "Absent from the card."},
            {"text": "Secure the fuel cap before flight.", "label": "recommendation", "source_ids": ["S1"], "because": "Advice."},
        ],
        "relevance": {"score": 4, "because": "The leakage question is answered."},
        "coverage": {"score": 3, "because": "Two of the bearing statements are used."},
        "specificity": {"score": 4, "because": "The cap action is tied to S1."},
    }


def test_source_card_uses_stored_text_only():
    statements = source_statements(EVIDENCE)
    assert [item["id"] for item in statements] == ["S1", "S2", "S3"], statements
    assert statements[0]["text"] == "FUEL CAP — causes — WATER IN FUEL"
    assert statements[2]["truncated"] is True
    assert source_statements({"chunks_retrieved": 12, "context_used": 1165}) == []
    assert source_statements(None) == []
    assert source_statements({"triples": []}) == []


def test_prompt_hides_the_system_name():
    prompt = build_prompt("Why do leaks repeat?", "Water enters through the cap.", source_statements(EVIDENCE))
    assert "keo" not in prompt.lower()
    assert "phi-4" not in prompt.lower()
    assert "S1" in prompt and "FUEL CAP" in prompt
    assert "longer answer is not" in prompt.lower() or "Length and the number" in prompt


def test_attribution_is_the_support_rate_not_a_model_average():
    applied = apply_claim_labels(_good_payload()["claims"], ["S1", "S2", "S3"])
    # 1 supported, 1 contradicted, 1 not_in_source; the recommendation is excluded.
    assert applied["n_factual"] == 3, applied
    assert applied["counts"][SUPPORTED] == 1
    assert abs(applied["support_rate"] - (1 / 3)) < 1e-9
    assert abs(applied["attribution"] - (1 + 4 * (1 / 3))) < 1e-9
    scored = score_payload(_good_payload(), source_statements(EVIDENCE))
    assert set(scored["scores"]) == {"attribution", *JUDGED_AXES}
    assert "overall" not in scored["scores"]
    assert scored["scores"]["relevance"] == 4


def test_citation_of_an_unknown_id_does_not_count_as_support():
    claims = [
        {"text": "Seals fail from age.", "label": "supported", "source_ids": ["S99"], "because": "invented id"},
        {"text": "The card says the opposite of a leak.", "label": "contradicted", "source_ids": [], "because": "no span"},
    ]
    applied = apply_claim_labels(claims, ["S1"])
    assert all(claim["label"] == NOT_IN_SOURCE for claim in applied["claims"])
    assert applied["attribution"] == 1.0


def test_recommendations_alone_leave_attribution_unset():
    claims = [{"text": "Inspect the cap.", "label": "advice", "source_ids": ["S1"], "because": "imperative"}]
    applied = apply_claim_labels(claims, ["S1"])
    assert applied["attribution"] is None
    assert applied["n_factual"] == 0


def test_scores_outside_the_scale_are_refused():
    payload = _good_payload()
    payload["coverage"] = {"score": 0, "because": "below the scale"}
    try:
        score_payload(payload, source_statements(EVIDENCE))
    except GroundedJudgeParseError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("a zero score was accepted")


def test_fenced_json_and_missing_object():
    judge = GroundedJudge(complete=lambda messages: "```json\n" + json.dumps(_good_payload()) + "\n```")
    result = judge.judge_one(
        "Why do leaks repeat?",
        "Water enters through the fuel cap.",
        EVIDENCE,
        question_id="q1",
    )
    assert result["status"] == "scored"
    assert result["rubric"] == RUBRIC_ID
    assert result["n_source_statements"] == 3

    def fail_if_called(_messages):
        raise AssertionError("the model was called without a source card")

    skipped = GroundedJudge(fail_if_called).judge_one("Q", "A", {"context_used": 10}, question_id="q2")
    assert skipped["status"] == "no_source"
    assert skipped["scores"] is None


def test_position_swap_keeps_only_a_consistent_side():
    agreed = reconcile_position_swap(
        {"attribution": "A", "relevance": "B", "coverage": "tie", "specificity": "A"},
        {"attribution": "B", "relevance": "A", "coverage": "tie", "specificity": "A"},
    )
    # Swapped order's B is the original A, so attribution agrees on A.
    # Specificity is A in both presentation orders, which is position bias.
    assert agreed["attribution"] == "A", agreed
    assert agreed["relevance"] == "B"
    assert agreed["coverage"] == "tie"
    assert agreed["specificity"] == "tie"


def test_style_names_and_gpt_lock():
    assert STYLE_NAMES == ("Grounded", "Source-card")
    assert HOLISTIC_NAMES == ("Holistic", "Sensemaking")
    assert RUBRIC_ID == "grounded"
    try:
        require_style("openai", "grounded")
    except JudgeStyleError as exc:
        assert "Holistic" in str(exc)
    else:
        raise AssertionError("GPT was allowed onto the grounded judge")
    require_style("claude_cli", "grounded")
    require_style("openai", "holistic")


def test_duplicate_graphrag_field_uses_the_primary_answer():
    row = {
        "question_id": "q",
        "keo_answer": {"answer": "Water.", "method": "keo", "evidence": EVIDENCE},
        "graphrag_answer": {"answer": "Water.", "method": "graphrag", "evidence": EVIDENCE},
    }
    picked = answer_payload(row)
    assert picked["answer"] == "Water."
    assert "method" not in build_prompt("Q", picked["answer"], source_statements(EVIDENCE)).lower()


def test_answer_field_selection_does_not_leak_into_the_prompt():
    row = {
        "question_id": "q",
        "question": "Why do leaks repeat?",
        "keo_answer": {"answer": "Water.", "method": "keo", "evidence": EVIDENCE},
        "chunk_tok_answer": {"answer": "Seals.", "method": "chunk_tok"},
    }
    assert answer_payload(row) is None
    picked = answer_payload(row, "keo_answer")
    prompt = build_prompt(row["question"], picked["answer"], source_statements(picked["evidence"]))
    assert "method" not in prompt.lower()
    assert "chunk_tok" not in prompt


def checks():
    return [
        ("source card", test_source_card_uses_stored_text_only),
        ("blind prompt", test_prompt_hides_the_system_name),
        ("attribution rate", test_attribution_is_the_support_rate_not_a_model_average),
        ("unknown citation", test_citation_of_an_unknown_id_does_not_count_as_support),
        ("recommendations", test_recommendations_alone_leave_attribution_unset),
        ("scale refusal", test_scores_outside_the_scale_are_refused),
        ("no source skips the model", test_fenced_json_and_missing_object),
        ("position swap", test_position_swap_keeps_only_a_consistent_side),
        ("field selection", test_answer_field_selection_does_not_leak_into_the_prompt),
        ("style names", test_style_names_and_gpt_lock),
        ("duplicate graphrag field", test_duplicate_graphrag_field_uses_the_primary_answer),
    ]


if __name__ == "__main__":
    run_suite("grounded judge", checks())
