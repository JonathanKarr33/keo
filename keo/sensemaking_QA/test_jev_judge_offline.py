#!/usr/bin/env python3
"""Offline checks for the Jev score mapping. No API calls.

Run: python test_jev_judge_offline.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jev_client import JevError, rubric_score  # noqa: E402
from jev_judge import (  # noqa: E402
    GROUNDED_AXES,
    HOLISTIC_AXES,
    questions_for,
    state_for,
)
from offline_harness import run_suite  # noqa: E402


def test_zero_index_maps_onto_one_to_five():
    assert rubric_score(0) == 1.0
    assert abs(rubric_score(1.99) - 2.99) < 1e-9
    assert rubric_score(4) == 5.0
    try:
        rubric_score(5)
    except JevError:
        pass
    else:
        raise AssertionError("an out-of-range Jev index was accepted")


def test_questions_are_ordered_scores_and_the_system_name_is_absent():
    holistic = questions_for(HOLISTIC_AXES)
    grounded = questions_for(GROUNDED_AXES)
    assert set(holistic) == set(HOLISTIC_AXES)
    assert all(item["type"] == "score" and len(item["criteria"]) == 5 for item in holistic.values())
    assert all(item["type"] == "score" and len(item["criteria"]) == 5 for item in grounded.values())
    blob = str(holistic) + str(grounded)
    assert "keo" not in blob.lower()
    assert "claude" not in blob.lower()


def test_holistic_state_omits_the_source_card():
    bare = state_for("Why do leaks repeat?", "Water enters through the cap.")
    assert "source_card" not in bare
    card = state_for("Why do leaks repeat?", "Water.", "S1 FUEL CAP — causes — WATER")
    assert "FUEL CAP" in card["source_card"]
    assert "method" not in str(card).lower()


def checks():
    return [
        ("score index", test_zero_index_maps_onto_one_to_five),
        ("question form", test_questions_are_ordered_scores_and_the_system_name_is_absent),
        ("holistic state", test_holistic_state_omits_the_source_card),
    ]


if __name__ == "__main__":
    run_suite("jev judge", checks())
