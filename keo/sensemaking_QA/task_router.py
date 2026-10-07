"""
Task router: pick retrieval policy from question type / cues.

Policies:
  keo       — graph + community (global sensemaking)
  textchunk — chunk RAG (procedural / knowledge-to-action)
  hybrid    — graph + community + chunks (ambiguous)
"""

from __future__ import annotations

import re
from typing import Dict, Literal, Optional

Policy = Literal["keo", "textchunk", "hybrid"]

ACTION_CUES = re.compile(
    r"\b(what action|how (do|to|should)|replace|inspect|repair|troubleshoot|"
    r"when:|procedure|step|fix|drain|tighten|torque)\b",
    re.I,
)
SENSE_CUES = re.compile(
    r"\b(pattern|trend|across|recurring|system-?level|sensemaking|overall|"
    r"compare|distribution|seasonal|aircraft types|to what extent)\b",
    re.I,
)


def route_question(question_data: Dict, prefer: Optional[Policy] = None) -> Policy:
    """
    Route a question to a retrieval policy.

    Priority:
      1. explicit prefer override
      2. annotated type/category from benchmark
      3. lexical cues
      4. default hybrid
    """
    if prefer in ("keo", "textchunk", "hybrid"):
        return prefer

    qtype = str(question_data.get("type", "")).lower()
    category = str(question_data.get("category", "")).lower()
    text = str(question_data.get("question", ""))

    if qtype in {"actionable", "action", "knowledge-to-action", "k2a"}:
        return "textchunk"
    if "action" in category or "knowledge-to-action" in category:
        return "textchunk"
    if qtype in {"global", "sensemaking"} or "sense" in category or "comprehensive" in category:
        return "keo"
    if ACTION_CUES.search(text) and not SENSE_CUES.search(text):
        return "textchunk"
    if SENSE_CUES.search(text) and not ACTION_CUES.search(text):
        return "keo"
    if ACTION_CUES.search(text) and SENSE_CUES.search(text):
        return "hybrid"
    return "hybrid"


def explain_route(question_data: Dict, policy: Policy) -> str:
    return (
        f"policy={policy}; type={question_data.get('type')}; "
        f"category={question_data.get('category')}"
    )
