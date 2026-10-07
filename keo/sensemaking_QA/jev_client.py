"""Jev on OpenRouter (Decisions API). Not a chat model.

typesafe/jev-1.13 answers typed questions and returns probabilities.
Score is a probability-weighted index: 0 is the first criterion. This client
maps that onto the 1-5 rubric by adding 1. Output tokens are free; input is
billed to OPENROUTER_API_KEY. Claude is not called.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from openrouter_client import looks_like_paid_interrupt, openrouter_api_key

JEV_MODEL = "typesafe/jev-1.13"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


class JevError(RuntimeError):
    pass


def load_repo_env() -> None:
    """Load repo-root .env without overriding variables already set. Never prints values."""
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def rubric_score(raw: Any, n_levels: int = 5) -> float:
    """Map Jev's 0-based score index onto 1..n_levels."""
    try:
        value = float(raw) + 1.0
    except (TypeError, ValueError) as exc:
        raise JevError(f"Jev score was not a number: {raw!r}") from exc
    if value < 1.0 - 1e-6 or value > n_levels + 1e-6:
        raise JevError(f"Jev score {raw} is outside 0..{n_levels - 1}")
    return value


def decide(
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    model: str = JEV_MODEL,
    timeout_s: float = 60.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """POST one Decisions request. Raises JevError. Paid interrupts propagate."""
    load_repo_env()
    key = openrouter_api_key()
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
    request = urllib.request.Request(
        DECISIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "keo"),
        },
        method="POST",
    )
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or "answers" not in payload:
                raise JevError(f"Decisions response had no answers: {str(payload)[:300]}")
            return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if looks_like_paid_interrupt(str(exc.code), detail):
                raise JevError(f"OpenRouter stopped the Jev run: {detail[:400]}") from exc
            last_error = JevError(f"HTTP {exc.code}: {detail[:400]}")
        except urllib.error.URLError as exc:
            last_error = JevError(f"Jev request failed: {exc.reason}")
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error
