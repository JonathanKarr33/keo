"""Offline checks for the annotator sheet builder. No model calls."""

import json

from export_annotator_sheets import build_answer_sheet, build_severity_sheet
from offline_harness import run_suite


def test_answer_sheet_is_blind_and_balanced():
    blind, key = build_answer_sheet(n=40, seed=17)
    assert blind["n"] == 40
    assert len(key["items"]) == 40
    blob = json.dumps(blind)
    assert "haiku" not in blob.lower()
    assert "sonnet" not in blob.lower()
    assert "left_system" not in blind["items"][0]
    assert "keo_answer" not in blind["items"][0]
    left_keo = sum(1 for row in key["items"] if row["left_system"] == "keo")
    assert 10 < left_keo < 30
    assert all(row["annotator_a"]["overall"] is None for row in blind["items"])


def test_severity_sheet_hides_the_prior():
    blind, key = build_severity_sheet(n=20, seed=17)
    assert blind["n"] == 20
    assert "prior_class" not in json.dumps(blind)
    classes = {row["prior_class"] for row in key["items"]}
    assert "none" in classes
    assert len(classes) > 1


def main():
    run_suite("annotator sheet", [
        ("answer sheet blind and balanced", test_answer_sheet_is_blind_and_balanced),
        ("severity sheet hides prior", test_severity_sheet_hides_the_prior),
    ])


if __name__ == "__main__":
    main()
