"""
Claude Code CLI wrapper — subscription-backed completions (Pro/Max).

For LLM-as-judge we match Messages API rigor as closely as the CLI allows:
  - pinned snapshot IDs (claude-haiku-4-5-20251001), not drifting aliases
  - --tools "" (pure completion, no agent tools)
  - --effort low (stable / cheap judging)
  - --output-format json (extract result + observed model)
  - retries + call metadata in provenance

Billing: Claude Pro/Max via `claude auth login` (not ANTHROPIC_API_KEY).
Agent drafts still use tools=default (see run_claude_agent.sh).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class ClaudeNotFoundError(RuntimeError):
    pass


class ClaudeAuthError(RuntimeError):
    pass


class ClaudeQuotaError(RuntimeError):
    """Subscription usage limit reached, after retries.

    Distinct from ClaudeAuthError because the remedy is different — wait for the window to reset or
    lower KEO_JUDGE_WORKERS, not re-authenticate. Without this the limit message became the judge's
    "answer", failed to parse, and the run aborted complaining that the rubric format had drifted,
    which sends you debugging the prompt instead of checking your quota.
    """


@dataclass(frozen=True)
class ClaudeResult:
    stdout: str
    stderr: str
    exit_code: int
    model: Optional[str] = None
    result_text: Optional[str] = None
    raw_json: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# Short names → pinned snapshots (same IDs as anthropic_client paper defaults)
_ALIAS_TO_PINNED = {
    "haiku": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "claude-sonnet-4-6": "claude-sonnet-5",
    "opus": "claude-opus-5",
}

DEFAULT_CLI_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_CLI_FINAL_MODEL = "claude-sonnet-5"

_auth_cache: Optional[Tuple[float, str]] = None
_AUTH_TTL_S = 300.0


def _detect_claude_cmd() -> str:
    cmd = os.environ.get("CLAUDE_CMD", "").strip()
    if cmd:
        return cmd
    found = shutil.which("claude")
    if found:
        return found
    raise ClaudeNotFoundError(
        "Could not find `claude` CLI. Install Claude Code so `claude` is on PATH, "
        "or set CLAUDE_CMD to the full path of the executable."
    )


def pin_cli_model(model: Optional[str]) -> str:
    """
    Resolve to a pinned snapshot ID for reproducible judging.
    Aliases expand to dated/full IDs; already-pinned IDs pass through.
    """
    if not model:
        return DEFAULT_CLI_JUDGE_MODEL
    key = str(model).strip()
    low = key.lower()
    if low in _ALIAS_TO_PINNED:
        return _ALIAS_TO_PINNED[low]
    return key


# Back-compat name used elsewhere
def normalize_cli_model(model: Optional[str]) -> str:
    return pin_cli_model(model)


def auth_status_text(*, force: bool = False) -> str:
    global _auth_cache
    now = time.time()
    if not force and _auth_cache and (now - _auth_cache[0]) < _AUTH_TTL_S:
        return _auth_cache[1]
    claude_cmd = _detect_claude_cmd()
    proc = subprocess.run(
        [claude_cmd, "auth", "status", "--text"],
        text=True,
        capture_output=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    status = "\n".join([s for s in [out, err] if s]).strip()
    _auth_cache = (now, status)
    return status


def ensure_logged_in() -> None:
    status = auth_status_text().lower()
    if "not logged in" in status:
        raise ClaudeAuthError(
            "Not logged in to Claude Code. Run `claude auth login` once "
            "(uses Pro/Max subscription — no ANTHROPIC_API_KEY)."
        )


def messages_to_prompt(messages: List[Dict[str, str]]) -> Tuple[Optional[str], str]:
    """OpenAI-style messages → (system_prompt, user_blob)."""
    system_parts: List[str] = []
    body: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            body.append(f"Assistant:\n{content}")
        else:
            body.append(f"User:\n{content}" if body else content)
    system = "\n\n".join(system_parts) if system_parts else None
    user = "\n\n".join(body) if body else "(empty)"
    return system, user


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_cli_json_stdout(stdout: str, fallback_model: str) -> Tuple[str, str, Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse Claude Code --output-format json envelope.
    Returns (result_text, observed_model, raw_obj, meta).
    """
    meta: Dict[str, Any] = {"output_format": "json"}
    raw = (stdout or "").strip()
    if not raw:
        return "", fallback_model, None, meta
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Sometimes CLI prints text before JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return _strip_fences(raw), fallback_model, None, {**meta, "parse": "raw_text"}
        else:
            return _strip_fences(raw), fallback_model, None, {**meta, "parse": "raw_text"}

    # Common Claude Code JSON shapes
    result = (
        obj.get("result")
        or obj.get("text")
        or obj.get("content")
        or obj.get("message")
    )
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        result = "\n".join(parts)
    if result is None:
        result = raw

    observed = obj.get("model")
    if not observed and isinstance(obj.get("data"), dict):
        observed = obj["data"].get("model")
    if not observed and isinstance(obj.get("message"), dict):
        observed = obj["message"].get("model")
    observed = observed or fallback_model

    for k in ("session_id", "duration_ms", "duration_api_ms", "num_turns", "total_cost_usd", "usage"):
        if k in obj:
            meta[k] = obj[k]

    return _strip_fences(str(result)), str(observed), obj if isinstance(obj, dict) else None, meta


def _looks_like_auth_failure(*chunks: str) -> bool:
    blob = "\n".join(c or "" for c in chunks).lower()
    markers = (
        "authentication_error",
        "oauth access token has expired",
        "not logged in",
        "failed to authenticate",
        "please run /login",
        "claude auth login",
    )
    return any(m in blob for m in markers)


# Hard stops: the window has to reset, so retrying in-process cannot help.
_QUOTA_EXHAUSTED_MARKERS = (
    "usage limit reached",
    "you've reached your usage limit",
    "youve reached your usage limit",
    "quota exceeded",
    "out of credits",
    "insufficient credit",
    "upgrade to continue",
    "limit will reset",
    "limit resets",
)

# Transient: worth a backoff before giving up, e.g. bursts from parallel workers.
_THROTTLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "overloaded",
    "server_error",
)


def _looks_like_quota_exhausted(*chunks: str) -> bool:
    blob = "\n".join(c or "" for c in chunks).lower()
    return any(m in blob for m in _QUOTA_EXHAUSTED_MARKERS)


def _looks_throttled(*chunks: str) -> bool:
    blob = "\n".join(c or "" for c in chunks).lower()
    return any(m in blob for m in _THROTTLE_MARKERS)


def run_claude(
    prompt: str,
    *,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout_s: Optional[float] = None,
    tools: Optional[str] = None,
    effort: Optional[str] = None,
    permission_mode: str = "dontAsk",
    no_session_persistence: bool = True,
    output_format: str = "json",
    max_retries: int = 2,
) -> ClaudeResult:
    """
    Run local `claude` CLI non-interactively (`-p`), prompt on stdin.

    Judge mode: tools=\"\", effort=low, output_format=json, pinned model ID.
    """
    claude_cmd = _detect_claude_cmd()
    ensure_logged_in()
    cli_model = pin_cli_model(model)

    argv = [
        claude_cmd,
        "-p",
        "Follow the instructions in stdin exactly. Output only the requested content.",
        "--model",
        cli_model,
        "--permission-mode",
        permission_mode,
        "--output-format",
        output_format,
    ]
    if no_session_persistence:
        argv.append("--no-session-persistence")
    if system_prompt:
        argv.extend(["--system-prompt", system_prompt])
    if tools is not None:
        argv.extend(["--tools", tools])
    if effort:
        argv.extend(["--effort", effort])

    timeout = timeout_s
    if timeout is None:
        env_t = os.getenv("CLAUDE_CLI_TIMEOUT_S", "").strip()
        timeout = float(env_t) if env_t else 600.0

    retries = max(0, int(os.getenv("CLAUDE_CLI_RETRIES", str(max_retries))))
    last: Optional[ClaudeResult] = None
    for attempt in range(retries + 1):
        proc = subprocess.run(
            argv,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        result_text = stdout
        observed = cli_model
        raw_json = None
        meta: Dict[str, Any] = {
            "attempt": attempt,
            "requested_model": cli_model,
            "claude_code_version": claude_code_version_text(),
            "billing": "claude_pro_max_subscription",
            "tools": tools,
            "effort": effort,
            "permission_mode": permission_mode,
        }
        if output_format == "json":
            result_text, observed, raw_json, jmeta = _parse_cli_json_stdout(stdout, cli_model)
            meta.update(jmeta)
        else:
            result_text = _strip_fences(stdout)

        if _looks_like_auth_failure(result_text, stderr, stdout) or (
            isinstance(raw_json, dict) and raw_json.get("is_error") and _looks_like_auth_failure(str(raw_json))
        ):
            raise ClaudeAuthError(
                "Claude Code auth failed (OAuth expired or invalid). "
                "Run `claude auth login`, then retry."
            )

        # A limit notice is not an answer. Left unchecked it becomes the judge's "response", fails
        # the rubric parse, and aborts the run blaming the prompt format.
        if _looks_like_quota_exhausted(result_text, stderr, stdout):
            raise ClaudeQuotaError(
                "Claude subscription usage limit reached, so no further judging is possible. "
                "Successful judgments from this run are written to "
                "global_sensemaking_checkpoint.json in the arm directory; re-run the same "
                "command after the limit resets and those question ids are skipped. "
                "If this arrived sooner than expected, lower KEO_JUDGE_WORKERS "
                f"(parallel judging spends quota faster).\nCLI said: "
                f"{(result_text or stderr or stdout or '').strip()[:400]}"
            )

        last = ClaudeResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            model=observed,
            result_text=result_text,
            raw_json=raw_json,
            meta=meta,
        )
        if proc.returncode == 0 and (result_text or "").strip():
            return last
        if attempt < retries:
            # Throttling from a burst of parallel workers deserves a longer wait than a generic
            # failure, since retrying immediately just adds to the burst.
            backoff = 1.5 * (attempt + 1)
            if _looks_throttled(result_text, stderr, stdout):
                backoff = max(backoff, 5.0 * (attempt + 1))
            time.sleep(backoff)
            continue
        if _looks_throttled(result_text, stderr, stdout):
            raise ClaudeQuotaError(
                "Claude Code kept reporting rate limits after "
                f"{retries + 1} attempts, so judging cannot proceed reliably. Lower "
                "KEO_JUDGE_WORKERS and re-run; already-judged arms are skipped.\nCLI said: "
                f"{(result_text or stderr or stdout or '').strip()[:400]}"
            )
    assert last is not None
    return last


def claude_cli_chat(
    *,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    effort: Optional[str] = None,
    tools: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Paper-parity judge completion via Claude Code.
    Returns (text, observed_model_id, call_meta).
    """
    system, user = messages_to_prompt(messages)
    effort = effort or os.getenv("CLAUDE_CLI_EFFORT", "low")
    pinned = pin_cli_model(model)

    judge_system = (
        (system or "You are a careful LLM-as-judge.")
        + "\n\nDo not use tools. Follow the requested output format exactly. "
        "No preamble, markdown fences, or closing chatter. "
        "Be deterministic and concise."
    )

    res = run_claude(
        user,
        model=pinned,
        system_prompt=judge_system,
        timeout_s=timeout_s,
        tools=tools,
        effort=effort,
        permission_mode="dontAsk",
        no_session_persistence=True,
        output_format="json",
    )
    if res.exit_code != 0:
        err = (res.stderr or res.stdout or "").strip()[:800]
        raise RuntimeError(
            f"Claude Code CLI failed (exit={res.exit_code}, model={res.model}): {err}"
        )
    text = (res.result_text or res.stdout or "").strip()
    if not text:
        raise RuntimeError(f"Claude Code CLI returned empty judge text (model={res.model})")
    meta = dict(res.meta or {})
    meta["observed_model"] = res.model or pinned
    meta["requested_model"] = pinned
    return text, str(res.model or pinned), meta


def claude_code_version_text() -> Optional[str]:
    try:
        claude_cmd = _detect_claude_cmd()
    except ClaudeNotFoundError:
        return None
    proc = subprocess.run(
        [claude_cmd, "--version"], text=True, capture_output=True, check=False
    )
    out = (proc.stdout or "").strip()
    return out or None


def is_pinned_snapshot(model_id: str) -> bool:
    """True if model looks like a dated/full Anthropic ID (not a short alias)."""
    m = (model_id or "").strip().lower()
    if m in _ALIAS_TO_PINNED or m in ("haiku", "sonnet", "opus"):
        return False
    return m.startswith("claude-")


def resolve_claude_cli_model(model_id: str) -> Dict[str, Any]:
    pinned = pin_cli_model(model_id)
    return {
        "requested_id": model_id,
        "resolved_id": pinned,
        "exact_id": pinned,
        "provider": "claude_cli",
        "billing": "claude_pro_max_subscription",
        "claude_code_version": claude_code_version_text(),
        "protocol": {
            "tools": "",
            "effort": "low",
            "output_format": "json",
            "permission_mode": "dontAsk",
            "pinned_snapshot": True,
        },
        "note": (
            "Claude Code with pinned snapshot + JSON envelope — "
            "subscription billing; protocol aligned with Messages API judging"
        ),
    }
