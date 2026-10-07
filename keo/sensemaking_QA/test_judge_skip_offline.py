#!/usr/bin/env python3
"""The judge skip must not treat a leftover eval as done after the answerer changes."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_skip import should_skip_judge  # noqa: E402
from offline_harness import run_suite  # noqa: E402


def _arm(tmpdir, *, answers_model, eval_model=None, write_eval=True, sidecar=False):
    os.makedirs(tmpdir, exist_ok=True)
    json.dump({"answer_model": answers_model}, open(os.path.join(tmpdir, "answers_meta.json"), "w"))
    if write_eval:
        blob = {"by_method": {"keo": {}}}
        if eval_model is not None:
            blob["answer_model"] = eval_model
        path = os.path.join(tmpdir, "global_sensemaking_evaluation_20260101.json")
        json.dump(blob, open(path, "w"))
        if sidecar:
            json.dump({}, open(path.replace(".json", "_judge_meta.json"), "w"))
    return tmpdir


def test_skip_when_models_match():
    with tempfile.TemporaryDirectory() as d:
        _arm(d, answers_model="gemma", eval_model="gemma")
        skip, reason = should_skip_judge(d, requested_model="gemma")
        assert skip, reason
    print("  matching answerer is skipped")


def test_refuse_when_answers_were_regenerated():
    with tempfile.TemporaryDirectory() as d:
        _arm(d, answers_model="qwen", eval_model="gemma")
        skip, reason = should_skip_judge(d, requested_model="qwen")
        assert not skip
        assert "gemma" in reason and "qwen" in reason
    print("  leftover gemma eval is not skipped after qwen answers")


def test_refuse_eval_without_recorded_model():
    with tempfile.TemporaryDirectory() as d:
        _arm(d, answers_model="gemma", eval_model=None)
        skip, reason = should_skip_judge(d)
        assert not skip
        assert "does not record" in reason
    print("  an eval that never recorded the answerer is not treated as done")


def test_sidecar_alone_is_not_judged():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(d, exist_ok=True)
        json.dump({"answer_model": "gemma"}, open(os.path.join(d, "answers_meta.json"), "w"))
        json.dump({}, open(os.path.join(d, "global_sensemaking_evaluation_20260101_judge_meta.json"), "w"))
        skip, reason = should_skip_judge(d)
        assert not skip
        assert "no evaluation file" in reason
    print("  a sidecar-only directory is not skipped")


def test_requested_model_mismatch():
    with tempfile.TemporaryDirectory() as d:
        _arm(d, answers_model="gemma", eval_model="gemma")
        skip, reason = should_skip_judge(d, requested_model="qwen")
        assert not skip
        assert "qwen" in reason
    print("  skip refuses when the sweep asked for a different model")


def main():
    run_suite("judge skip", [
        ("models match", test_skip_when_models_match),
        ("regenerated answers", test_refuse_when_answers_were_regenerated),
        ("unrecorded eval", test_refuse_eval_without_recorded_model),
        ("sidecar only", test_sidecar_alone_is_not_judged),
        ("requested mismatch", test_requested_model_mismatch),
    ])


if __name__ == "__main__":
    main()
