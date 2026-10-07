"""Two judge styles. Each style has two names. They are not pooled.

Holistic / Sensemaking
    Absolute 1-5 impression scores on Global Perspective, Theme Identification,
    Synthesis Quality, Strategic Value, and Pattern Recognition. The judge sees
    the question and the answer, not the retrieved context. Every published
    matrix uses this style. The legacy GPT judge runs only this style.

Grounded / Source-card
    Atomic claims checked against the context stored with the answer, plus
    anchored relevance, coverage, and specificity. Claude only.
"""

from __future__ import annotations

HOLISTIC_ID = "holistic"
HOLISTIC_NAMES = ("Holistic", "Sensemaking")

GROUNDED_ID = "grounded"
GROUNDED_NAMES = ("Grounded", "Source-card")

_GPT_PROVIDERS = frozenset({"openai", "legacy"})


class JudgeStyleError(ValueError):
    """A model was pointed at a judge style it is not allowed to run."""


def require_style(provider: str, style_id: str) -> None:
    """GPT stays on Holistic / Sensemaking. Grounded / Source-card is Claude."""
    if style_id != GROUNDED_ID:
        return
    if (provider or "").strip().lower() in _GPT_PROVIDERS:
        raise JudgeStyleError(
            "GPT runs only the Holistic judge (Sensemaking rubric). "
            "The Grounded judge (Source-card rubric) is not available on that model."
        )
