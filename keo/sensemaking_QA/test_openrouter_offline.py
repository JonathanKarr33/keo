"""Offline checks for the OpenRouter client. No network."""

import os

os.environ.pop("OPENROUTER_API_KEY", None)

from openrouter_client import (  # noqa: E402
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    PaidAPIInterrupt,
    looks_like_paid_interrupt,
    openrouter_api_key,
    openrouter_model,
)


def test_missing_key_raises():
    try:
        openrouter_api_key()
    except ValueError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("missing key should raise")


def test_explicit_empty_key_raises():
    try:
        openrouter_api_key("   ")
    except ValueError:
        return
    raise AssertionError("blank key should raise")


def test_defaults_do_not_call_network():
    assert DEFAULT_OPENROUTER_MODEL == "google/gemma-4-31b-it"
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert openrouter_model() == DEFAULT_OPENROUTER_MODEL
    assert openrouter_api_key("test-key") == "test-key"


def test_paid_interrupt_is_distinct_error():
    assert issubclass(PaidAPIInterrupt, RuntimeError)
    assert looks_like_paid_interrupt("402 Payment Required")
    assert not looks_like_paid_interrupt("model not found")


if __name__ == "__main__":
    test_missing_key_raises()
    test_explicit_empty_key_raises()
    test_defaults_do_not_call_network()
    test_paid_interrupt_is_distinct_error()
    print("OK: 4 openrouter checks passed")
