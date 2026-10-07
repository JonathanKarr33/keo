#!/usr/bin/env python3
"""Offline checks on local-model generation.

A reasoning model served over an OpenAI-compatible endpoint returns its chain of thought in a
separate `reasoning` field. When the completion budget runs out mid-thought, `content` comes back
empty or absent, and coercing that to "" writes a blank answer carrying no error — the arm then
looks fully generated and the judge scores silence. These checks pin that behaviour down.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KEO_ALLOW_MISSING_NLP", "1")

from answer_generator import SensemakingAnswerGenerator  # noqa: E402
from offline_harness import run_suite  # noqa: E402


def make_generator(model: str, reply_content, reasoning: str = "") -> SensemakingAnswerGenerator:
    """A generator wired to a stub endpoint, skipping the real __init__ to stay offline."""
    gen = object.__new__(SensemakingAnswerGenerator)
    gen.provider = "local"
    gen.model = model
    gen.max_answer_tokens = 600
    gen.use_completion_cache = False
    gen._observed_model_ids = set()
    gen.base_url = "http://127.0.0.1:8000/v1"

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content=reply_content, reasoning=reasoning)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")], model=model
        )

    gen.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    gen.captured = captured
    return gen


MESSAGES = [{"role": "user", "content": "What recurring themes appear?"}]


def test_thinking_is_disabled_for_reasoning_models():
    for model in ("mlx-community/gemma-4-E4B-it-qat-4bit", "Qwen/Qwen3-8B"):
        extra = make_generator(model, "ok")._local_extra_body()
        kwargs = extra.get("chat_template_kwargs", {})
        assert kwargs.get("enable_thinking") is False, (
            f"{model} is a reasoning model; without enable_thinking=False it spends the "
            "completion budget thinking and returns no answer"
        )
    plain = make_generator("mlx-community/Phi-4-mini-instruct-4bit", "ok")._local_extra_body()
    assert plain == {}, f"non-reasoning model should not be sent host kwargs, got {plain}"
    print("  gemma-4 and qwen disable thinking; phi is left alone")


def test_empty_completion_raises_instead_of_blank_answer():
    for label, content in (("absent content", None), ("blank content", "   ")):
        gen = make_generator("mlx-community/gemma-4-E4B-it-qat-4bit", content, reasoning="x" * 2200)
        try:
            gen._create_chat_completion(MESSAGES)
        except RuntimeError as e:
            assert "reasoning" in str(e), f"{label}: error should name the cause, got {e}"
        else:
            raise AssertionError(
                f"{label}: returned an empty answer instead of raising, so the arm would "
                "record a blank answer with no error"
            )
    print("  an empty completion stops the item rather than writing a blank answer")


def test_error_names_the_remedy():
    gen = make_generator("mlx-community/gemma-4-E4B-it-qat-4bit", None, reasoning="x" * 2200)
    try:
        gen._create_chat_completion(MESSAGES)
    except RuntimeError as e:
        msg = str(e)
        assert "2200" in msg, "error should report how much reasoning was returned"
        assert "max-answer-tokens" in msg or "_local_extra_body" in msg, (
            f"error should say what to change, got {msg}"
        )
    print("  the error reports the reasoning length and how to fix it")


def test_no_reasoning_field_still_raises():
    """A plain endpoint returning nothing is also a failure, just with a different message."""
    gen = make_generator("mlx-community/Phi-4-mini-instruct-4bit", "")
    try:
        gen._create_chat_completion(MESSAGES)
    except RuntimeError as e:
        assert "no content" in str(e), f"expected a no-content message, got {e}"
    else:
        raise AssertionError("an empty reply returned silently")
    print("  an empty reply without reasoning still stops the item")


def test_terminators_are_passed_and_stripped():
    gen = make_generator("mlx-community/gemma-4-E4B-it-qat-4bit", "Ready<|end|>")
    text = gen._create_chat_completion(MESSAGES)
    assert gen.captured.get("stop"), "stop sequences were not sent, so terminators leak"
    assert text == "Ready", f"terminator survived into the answer: {text!r}"
    print("  chat terminators are both suppressed and stripped")


def test_answer_token_cap_is_applied():
    gen = make_generator("mlx-community/gemma-4-E4B-it-qat-4bit", "ok")
    gen._create_chat_completion(MESSAGES, max_tokens=5000)
    assert gen.captured["max_tokens"] == 600, (
        f"max_answer_tokens should cap the request, sent {gen.captured['max_tokens']}"
    )
    print("  max_answer_tokens caps a larger per-call request")


def main():
    run_suite("generation", [
        ("thinking disabled for reasoning models", test_thinking_is_disabled_for_reasoning_models),
        ("empty completion raises", test_empty_completion_raises_instead_of_blank_answer),
        ("error names the remedy", test_error_names_the_remedy),
        ("empty reply without reasoning raises", test_no_reasoning_field_still_raises),
        ("terminators passed and stripped", test_terminators_are_passed_and_stripped),
        ("answer token cap applied", test_answer_token_cap_is_applied),
    ])


if __name__ == "__main__":
    main()
