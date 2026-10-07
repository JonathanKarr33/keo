"""
Shared Anthropic Messages API helpers for KEO — paper-grade API twin.

Default judging uses Claude Code (`provider=claude_cli`) with the same pinned IDs.
Use this module when you need Messages API (`ANTHROPIC_API_KEY`).

Defaults (Sep 2026):
  - Ablations: claude-haiku-4-5-20251001  (~$1 / $5 per 1M)
  - Final:     claude-sonnet-5            (~$2 / $10 per 1M)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_FINAL_JUDGE_MODEL = "claude-sonnet-5"
DEFAULT_ANSWER_MODEL = "claude-haiku-4-5-20251001"  # optional closed answerer only
DEFAULT_JUDGE_TEMPERATURE = 0.0

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False


def make_anthropic_client(api_key: Optional[str] = None) -> Any:
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package required: pip install anthropic")
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=key)


def split_openai_style_messages(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Convert OpenAI chat messages → (system, anthropic_messages)."""
    system_parts: List[str] = []
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            out.append({"role": role, "content": content})
        else:
            out.append({"role": "user", "content": f"[{role}] {content}"})
    system = "\n\n".join(system_parts) if system_parts else None
    if not out:
        out = [{"role": "user", "content": "(empty)"}]
    if out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(begin)"})
    return system, out


def _model_allows_temperature(model: str) -> bool:
    mid = str(model).lower()
    if mid.startswith("claude-sonnet-5") or mid.startswith("claude-opus-5"):
        return False
    if mid.startswith("claude-fable"):
        return False
    return True


def _is_transient_api_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    markers = ("429", "rate limit", "overloaded", "timeout", "503", "529", "capacity")
    return any(m in msg for m in markers)


def anthropic_chat(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1000,
    temperature: Optional[float] = DEFAULT_JUDGE_TEMPERATURE,
    disable_thinking: bool = True,
    max_retries: int = 3,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Create a Messages API completion for judging.
    Returns (text, observed_model_id, call_meta).
    """
    # Expand haiku/sonnet aliases to pinned snapshots
    try:
        from judge_policy import pin_judge_model

        model = pin_judge_model(model)
    except Exception:
        pass

    system, anth_messages = split_openai_style_messages(messages)
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anth_messages,
    }
    if system:
        kwargs["system"] = system

    temp = DEFAULT_JUDGE_TEMPERATURE if temperature is None else temperature
    if _model_allows_temperature(model):
        kwargs["temperature"] = temp

    if disable_thinking:
        mid = str(model).lower()
        if mid.startswith("claude-sonnet-5") or mid.startswith("claude-opus-5") or mid.startswith("claude-fable"):
            kwargs["thinking"] = {"type": "disabled"}

    call_meta: Dict[str, Any] = {
        "requested_model": model,
        "temperature": kwargs.get("temperature"),
        "max_tokens": max_tokens,
        "thinking": kwargs.get("thinking"),
    }

    retries = max(0, int(os.getenv("ANTHROPIC_RETRIES", str(max_retries))))
    response = None
    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(**kwargs)
            call_meta["attempt"] = attempt
            break
        except Exception as e:
            last_err = e
            # Drop thinking once if rejected
            if "thinking" in kwargs and "thinking" in str(e).lower():
                kwargs.pop("thinking", None)
                call_meta["thinking_omitted"] = str(e)[:200]
                continue
            if attempt < retries and _is_transient_api_error(e):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    if response is None:
        assert last_err is not None
        raise last_err

    observed = getattr(response, "model", None) or model
    call_meta["observed_model"] = str(observed)
    usage = getattr(response, "usage", None)
    if usage is not None:
        call_meta["usage"] = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
    stop = getattr(response, "stop_reason", None)
    if stop:
        call_meta["stop_reason"] = stop

    parts = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif isinstance(block, dict) and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts).strip(), str(observed) if observed else model, call_meta


def resolve_anthropic_model(model_id: str) -> Dict[str, Any]:
    try:
        from judge_policy import pin_judge_model

        resolved = pin_judge_model(model_id)
    except Exception:
        resolved = model_id
    return {
        "requested_id": model_id,
        "resolved_id": resolved,
        "exact_id": resolved,
        "provider": "anthropic",
        "billing": "anthropic_messages_api",
        "note": "Pinned Anthropic snapshot — Messages API twin of Claude Code judging",
    }
