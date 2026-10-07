"""
KEO judge policy — paper-parity Claude judging.

Both of these are **reportable (paper)** when using pinned snapshot IDs +
deterministic protocol:
  - provider=anthropic  (Messages API, ANTHROPIC_API_KEY)
  - provider=claude_cli (Claude Code Pro/Max, claude auth login)

Smoke grade keeps short aliases / lax settings for pipeline checks only.
Legacy = OpenAI GPT-5.6 continuity.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# Shared pinned IDs (API and Claude Code)
PAPER_JUDGE_MODEL = "claude-haiku-4-5-20251001"
PAPER_FINAL_MODEL = "claude-sonnet-5"
PAPER_TEMPERATURE = 0.0

_DEFAULT_PAPER_PROVIDER = (
    os.getenv("JUDGE_PROVIDER")
    or os.getenv("KEO_JUDGE_PROVIDER")
    or "claude_cli"
).strip().lower()
PAPER_PROVIDER = (
    "anthropic" if _DEFAULT_PAPER_PROVIDER in ("anthropic", "api") else "claude_cli"
)

SMOKE_PROVIDER = "claude_cli"
SMOKE_JUDGE_MODEL = "claude-haiku-4-5-20251001"  # still pinned; grade marks non-reportable intent
SMOKE_FINAL_MODEL = "claude-sonnet-5"

LEGACY_PROVIDER = "openai"
LEGACY_JUDGE_MODEL = "gpt-5.6-luna"
LEGACY_FINAL_MODEL = "gpt-5.6-terra"

GRADE_PAPER = "paper"
GRADE_SMOKE = "smoke"
GRADE_LEGACY = "legacy"


def normalize_provider(provider: Optional[str]) -> str:
    p = (provider or "").strip().lower().replace("-", "_")
    if p in ("claude_code", "claude"):
        return "claude_cli"
    if p in ("api",):
        return "anthropic"
    return p or PAPER_PROVIDER


def infer_grade(provider: str, explicit: Optional[str] = None) -> str:
    if explicit in (GRADE_PAPER, GRADE_SMOKE, GRADE_LEGACY):
        return explicit
    p = normalize_provider(provider)
    if p == "openai":
        return GRADE_LEGACY
    # anthropic and claude_cli both default to paper when grade omitted
    return GRADE_PAPER


def pin_judge_model(model: Optional[str]) -> str:
    """Expand haiku/sonnet aliases to pinned snapshots (shared by CLI + API)."""
    from claude_cli import pin_cli_model

    return pin_cli_model(model)


def resolve_judge_defaults(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    grade: Optional[str] = None,
    final: bool = False,
) -> Tuple[str, str, str]:
    """
    Returns (provider, model, grade).
    Default: paper-grade Claude via Claude Code (pinned Haiku), no API key.
    Set JUDGE_PROVIDER=anthropic or KEO_JUDGE_PROVIDER=anthropic for Messages API.
    """
    g = grade
    p = normalize_provider(provider) if provider else None

    if g is None and p is None:
        g = GRADE_PAPER
        p = PAPER_PROVIDER
    elif g is None:
        g = infer_grade(p or PAPER_PROVIDER)
    elif p is None:
        if g == GRADE_SMOKE:
            p = SMOKE_PROVIDER
        elif g == GRADE_LEGACY:
            p = LEGACY_PROVIDER
        else:
            p = PAPER_PROVIDER

    assert p is not None and g is not None

    if model:
        if p in ("claude_cli", "anthropic"):
            model = pin_judge_model(model)
        return p, model, g

    if g == GRADE_LEGACY:
        return p, (LEGACY_FINAL_MODEL if final else LEGACY_JUDGE_MODEL), g
    return p, (PAPER_FINAL_MODEL if final else PAPER_JUDGE_MODEL), g


def judge_meta(
    *,
    provider: str,
    model: str,
    grade: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    g = infer_grade(provider, grade)
    p = normalize_provider(provider)
    # Paper is reportable for both anthropic and claude_cli
    reportable = g == GRADE_PAPER and p in ("anthropic", "claude_cli")
    if g == GRADE_PAPER and p == "claude_cli":
        policy = (
            "Claude Code + pinned snapshot + JSON envelope (Pro/Max) — "
            "paper-parity with Messages API protocol; subscription billing"
        )
    elif g == GRADE_PAPER and p == "anthropic":
        policy = "Messages API + pinned snapshot — paper-grade (API credits)"
    elif g == GRADE_SMOKE:
        policy = "Smoke intent — same CLI stack; treat as pipeline check if desired"
    else:
        policy = "OpenAI legacy judge — continuity only"

    out: Dict[str, Any] = {
        "judge_grade": g,
        "reportable_for_paper": reportable,
        "provider": p,
        "requested_model": model,
        "policy": policy,
        "parity": {
            "pinned_model": True,
            "tools_disabled": p == "claude_cli",
            "temperature_0": p == "anthropic",
            "effort_low": p == "claude_cli",
            "json_envelope": p == "claude_cli",
        },
    }
    if extra:
        out.update(extra)
    return out


def warn_if_non_paper(grade: str) -> Optional[str]:
    if grade == GRADE_PAPER:
        return None
    if grade == GRADE_SMOKE:
        return (
            "NOTE: judge_grade=smoke. Protocol still uses pinned Claude IDs via Claude Code; "
            "mark paper if you intend these scores for tables (--judge-grade paper)."
        )
    return (
        "WARNING: judge_grade=legacy (openai). GPT runs only the Holistic judge "
        "(Sensemaking rubric). The Grounded judge (Source-card rubric) is Claude-only."
    )
