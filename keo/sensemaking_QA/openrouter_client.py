"""OpenRouter chat client for a larger answerer.

The API key is not stored in the repo. Set OPENROUTER_API_KEY before any call.
Nothing in this module contacts the network at import time; make_openrouter_client
only constructs an OpenAI-compatible client pointed at OpenRouter.
"""

from __future__ import annotations

import os
from typing import Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Larger Gemma than the local E4B answerer. Override with OPENROUTER_MODEL.
# Launcher default is the same; keep these aligned so a bare provider=openrouter
# call cannot silently fall back to Gemma-3 27B.
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it"

# Billing / hard rate-limit stop. Callers checkpoint and exit; soft per-item errors
# must not absorb this (same role as ClaudeQuotaError for the judge).
_HARD_LIMIT_MARKERS = (
    "insufficient credits",
    "payment required",
    "402",
    "credit limit",
    "credits exhausted",
    "usage limit",
    "quota exceeded",
    "quota exhausted",
    "billing",
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "capacity",
    "overloaded",
)


class PaidAPIInterrupt(RuntimeError):
    """OpenRouter (or other paid) hard stop: checkpoint answers and abort the arm."""


def looks_like_paid_interrupt(*chunks: str) -> bool:
    blob = " ".join(c for c in chunks if c).lower()
    return any(m in blob for m in _HARD_LIMIT_MARKERS)


def openrouter_api_key(explicit: Optional[str] = None) -> str:
    key = (explicit or os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Export it before provider='openrouter'. "
            "This client does not call OpenRouter without a key."
        )
    return key


def openrouter_model(explicit: Optional[str] = None) -> str:
    return (explicit or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL).strip()


def make_openrouter_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Build an OpenAI-compatible client. Does not send a request."""
    from openai import OpenAI

    return OpenAI(
        api_key=openrouter_api_key(api_key),
        base_url=(base_url or os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/"),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "keo"),
        },
    )
