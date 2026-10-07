#!/usr/bin/env python3
"""Source-bounded judge. Separate from evaluator.py.

The five-criterion global judge (Global Perspective, Theme Identification,
Synthesis Quality, Strategic Value, Pattern Recognition) is unchanged and
remains the source of the published scores. This module writes a different
file and a different score. Do not average the two, and do not pool them
across judges.

The rubric follows the current grounded-evaluation practice:

- Rashkin et al. 2023, Attributable to Identified Sources. A claim counts as
  supported only when the judge points at a span in the source card that was
  stored with the answer. Parametric knowledge is not a source.
- Min et al. 2023, FActScore. The answer is split into atomic claims. The
  attribution score is the fraction of factual claims labeled supported,
  computed here from those labels rather than taken from the model's average.
- Es et al. 2024, RAGAS. Faithfulness, whether the answer addresses the
  question, and whether it uses the retrieved context are three scores.
- Liu et al. 2023, G-Eval. The model reasons inside a fixed form, then emits
  that form. Temperature is 0 where the provider allows it.
- Kim et al. 2024, Prometheus 2. Each level from 1 to 5 has a definition.
- Zheng et al. 2023, Judging LLM-as-a-Judge. The system name is not shown.
  Length is not a criterion. A pairwise comparison, if used, is run in both
  orders and kept only when both orders agree.
- Edge et al. 2024, GraphRAG. That paper's comprehensiveness / diversity /
  empowerment / directness criteria are the family the old judge belongs to.
  None of those names are used here, so a later table cannot mix the scales.

Chunk arms that stored only a character count, and not the retrieved text,
are marked no_source and are not scored. Attribution is undefined without
the context the generator saw.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from judge_styles import GROUNDED_ID, GROUNDED_NAMES, require_style

RUBRIC_ID = GROUNDED_ID
STYLE_NAMES = GROUNDED_NAMES
NOT_COMPARABLE_TO = "holistic / Sensemaking (evaluator._evaluate_global_sensemaking_answer)"

# Haiku matrices already on disk. Sonnet trees are copies of the same answers.
# Chunk arms are skipped: they stored a character count, not the retrieved text.
CURRENT_ROOTS = (
    "evaluation_results/ablations/_gemma4e4b",
    "evaluation_results/ablations/phi4",
    "evaluation_results/ablations/openrouter_gemma4_31b",
    "evaluation_results/ablations/labeled_graph",
)

# Factual labels enter the support rate. Recommendations do not: they are
# advice, and specificity is where generic advice is penalised.
SUPPORTED = "supported"
CONTRADICTED = "contradicted"
NOT_IN_SOURCE = "not_in_source"
RECOMMENDATION = "recommendation"
FACTUAL_LABELS = (SUPPORTED, CONTRADICTED, NOT_IN_SOURCE)

JUDGED_AXES = ("relevance", "coverage", "specificity")
SCORE_MIN = 1
SCORE_MAX = 5

_LABEL_ALIASES = {
    "supported": SUPPORTED,
    "support": SUPPORTED,
    "entailed": SUPPORTED,
    "contradicted": CONTRADICTED,
    "contradiction": CONTRADICTED,
    "refuted": CONTRADICTED,
    "not_in_source": NOT_IN_SOURCE,
    "not in source": NOT_IN_SOURCE,
    "not_in_context": NOT_IN_SOURCE,
    "unsupported": NOT_IN_SOURCE,
    "absent": NOT_IN_SOURCE,
    "recommendation": RECOMMENDATION,
    "advice": RECOMMENDATION,
    "generic": RECOMMENDATION,
}

SYSTEM_PROMPT = """You are a source-bounded evaluator of aviation maintenance answers.

The source card is the only evidence. Training data, general aviation knowledge, and plausibility are not evidence. A statement can be true of aircraft in general and still be not_in_source when the card does not state it.

A longer answer is not a better answer. Do not reward added themes, lists, or caveats that the card does not contain.

You are not told which system wrote the answer. Do not guess.

Reply with one JSON object and no other text."""


class GroundedJudgeParseError(RuntimeError):
    """The judge replied, but the form could not be read. Nothing is imputed."""


def source_statements(evidence: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Turn stored evidence into the card the judge is allowed to cite.

    Graph arms store triples (head, relation, tail). Some hybrid records also
    store chunk previews. A character count without text is not a statement.
    """
    if not isinstance(evidence, Mapping):
        return []
    statements: List[Dict[str, Any]] = []
    triples = evidence.get("triples") or []
    if isinstance(triples, list):
        for triple in triples:
            if not isinstance(triple, Mapping):
                continue
            head = str(triple.get("head") or "").strip()
            tail = str(triple.get("tail") or "").strip()
            if not head and not tail:
                continue
            relation = str(triple.get("relation") or "related_to").strip() or "related_to"
            statements.append({
                "text": f"{head} — {relation} — {tail}",
                "truncated": False,
                "kind": "triple",
            })
    chunks = evidence.get("chunks") or []
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, str):
                text = chunk.strip()
                truncated = False
            elif isinstance(chunk, Mapping):
                text = str(chunk.get("text") or chunk.get("preview") or "").strip()
                truncated = "preview" in chunk and "text" not in chunk
            else:
                continue
            if not text:
                continue
            statements.append({
                "text": text,
                "truncated": truncated,
                "kind": "chunk",
            })
    for index, statement in enumerate(statements, start=1):
        statement["id"] = f"S{index}"
    return statements


def render_source_card(statements: Sequence[Mapping[str, Any]]) -> str:
    if not statements:
        return ""
    lines = [
        "Source card. This is the entire context stored with the answer.",
        "A line marked truncated is incomplete: cite it only for a claim that the visible text states.",
    ]
    for statement in statements:
        mark = " truncated" if statement.get("truncated") else ""
        lines.append(f"[{statement['id']}]{mark} {statement['text']}")
    return "\n".join(lines)


def _anchors() -> str:
    return """
Relevance — does the answer address the question that was asked? Ignore whether the claims are in the source.
1 The answer does not address the question.
2 The answer shares the topic and responds to a different question.
3 A major part of the question is unanswered.
4 The question is answered, with a minor omission.
5 The question that was asked is answered.

Coverage — of the source-card statements that bear on the question, how many does the answer use? Facts that are not on the card do not raise this score.
1 None of the relevant statements are used.
2 One relevant statement is used and others of equal bearing are ignored.
3 Some relevant statements are used and others of equal bearing are omitted.
4 Most relevant statements are used.
5 The relevant statements are used, and the answer does not pad with themes the card does not contain.

Specificity — could a maintainer act on the causes or steps, and is each one tied to a source-card statement? Length and the number of bullet points are not specificity.
1 The advice would fit any maintenance question.
2 The advice is domain-plausible and tied to no source-card statement.
3 One concrete cause or action appears, weakly tied to the card.
4 Concrete causes or actions appear, and each is tied to a source-card statement.
5 A maintainer could carry out the stated causes or actions, and each is tied to a source-card statement.
""".strip()


def build_prompt(question: str, answer: str, statements: Sequence[Mapping[str, Any]]) -> str:
    """User message. The method name, model name, and arm id are not included."""
    card = render_source_card(statements)
    return f"""Question:
{question.strip()}

Answer:
{answer.strip()}

{card}

Split the answer into atomic claims. Label every claim with one of:
- supported — the cited source ids state it
- contradicted — the cited source ids state the opposite
- not_in_source — the card does not state it, whether or not it is generally true
- recommendation — advice or an imperative, not a factual claim

A supported or contradicted claim must cite one or more source ids from the card. Do not cite an id that is not on the card.

Then score relevance, coverage, and specificity with the definitions below. Use an integer from 1 to 5.

{_anchors()}

Return this JSON object and nothing else:
{{
  "claims": [
    {{"text": "...", "label": "supported|contradicted|not_in_source|recommendation", "source_ids": ["S1"], "because": "one sentence"}}
  ],
  "relevance": {{"score": 1, "because": "one sentence"}},
  "coverage": {{"score": 1, "because": "one sentence"}},
  "specificity": {{"score": 1, "because": "one sentence"}}
}}"""


def build_pairwise_prompt(
    question: str,
    answer_a: str,
    statements_a: Sequence[Mapping[str, Any]],
    answer_b: str,
    statements_b: Sequence[Mapping[str, Any]],
) -> str:
    """One presentation order. Call twice with the sides swapped (Zheng et al. 2023)."""
    return f"""Question:
{question.strip()}

Answer A:
{answer_a.strip()}

Source card A:
{render_source_card(statements_a)}

Answer B:
{answer_b.strip()}

Source card B:
{render_source_card(statements_b)}

Compare the answers on attribution, relevance, coverage, and specificity.
Attribution means a larger share of factual claims is stated by that answer's own source card.
Use the same relevance, coverage, and specificity definitions as a single-answer review.
A longer answer is not the winner. Each side may use only its own source card.

For each axis return "A", "B", or "tie".

Return this JSON object and nothing else:
{{
  "attribution": "A",
  "relevance": "A",
  "coverage": "tie",
  "specificity": "B",
  "because": "one sentence per axis"
}}"""


def reconcile_position_swap(
    order_ab: Mapping[str, str],
    order_ba: Mapping[str, str],
    axes: Sequence[str] = ("attribution",) + JUDGED_AXES,
) -> Dict[str, str]:
    """Keep a side only when both presentation orders prefer it.

    order_ba uses A/B in the swapped presentation, so its "A" is the original B.
    Disagreement is a tie. That is the position-bias check in Zheng et al. 2023.
    """
    decided: Dict[str, str] = {}
    for axis in axes:
        forward = _pref_token(order_ab.get(axis))
        backward = _pref_token(order_ba.get(axis))
        if forward is None or backward is None:
            raise GroundedJudgeParseError(f"{axis} preference missing from a swapped order")
        backward_as_original = {"A": "B", "B": "A", "tie": "tie"}[backward]
        if forward == backward_as_original:
            decided[axis] = forward
        else:
            decided[axis] = "tie"
    return decided


def _pref_token(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if token in ("a", "b", "tie"):
        return token.upper() if token != "tie" else "tie"
    return None


def _extract_json_object(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    blob = fenced.group(1) if fenced else text
    start = blob.find("{")
    if start < 0:
        raise GroundedJudgeParseError("judge reply contained no JSON object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(blob)):
        char = blob[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(blob[start:index + 1])
                except json.JSONDecodeError as exc:
                    raise GroundedJudgeParseError(f"judge JSON did not parse: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise GroundedJudgeParseError("judge JSON was not an object")
                return parsed
    raise GroundedJudgeParseError("judge JSON object was not closed")


def _normalize_label(raw: Any) -> str:
    if not isinstance(raw, str):
        raise GroundedJudgeParseError(f"claim label was not a string: {raw!r}")
    key = raw.strip().lower().replace("-", "_")
    if key not in _LABEL_ALIASES:
        raise GroundedJudgeParseError(f"unknown claim label: {raw!r}")
    return _LABEL_ALIASES[key]


def _axis_score(payload: Mapping[str, Any], axis: str) -> Optional[float]:
    block = payload.get(axis)
    if not isinstance(block, Mapping) or "score" not in block:
        raise GroundedJudgeParseError(f"{axis} score missing")
    try:
        value = float(block["score"])
    except (TypeError, ValueError) as exc:
        raise GroundedJudgeParseError(f"{axis} score was not a number") from exc
    if not SCORE_MIN <= value <= SCORE_MAX:
        raise GroundedJudgeParseError(f"{axis} score {value} is outside {SCORE_MIN}-{SCORE_MAX}")
    return value


def apply_claim_labels(
    claims: Sequence[Mapping[str, Any]],
    valid_ids: Iterable[str],
) -> Dict[str, Any]:
    """Compute attribution from labels. A citation of an id not on the card does not count.

    support rate = supported / factual claims.
    attribution = 1 + 4 * support rate, so every factual claim supported is 5 and
    none supported is 1. Recommendations are excluded from that rate.
    An empty factual set leaves attribution null rather than a midpoint.
    """
    allowed = set(valid_ids)
    normalized: List[Dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise GroundedJudgeParseError("a claim was not an object")
        text = str(claim.get("text") or "").strip()
        if not text:
            raise GroundedJudgeParseError("a claim had empty text")
        label = _normalize_label(claim.get("label"))
        raw_ids = claim.get("source_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list):
            raise GroundedJudgeParseError("source_ids was not a list")
        source_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        cited = [item for item in source_ids if item in allowed]
        if label in (SUPPORTED, CONTRADICTED) and not cited:
            label = NOT_IN_SOURCE
        normalized.append({
            "text": text,
            "label": label,
            "source_ids": cited,
            "because": str(claim.get("because") or "").strip(),
        })

    counts = {label: 0 for label in (SUPPORTED, CONTRADICTED, NOT_IN_SOURCE, RECOMMENDATION)}
    for claim in normalized:
        counts[claim["label"]] += 1
    factual = counts[SUPPORTED] + counts[CONTRADICTED] + counts[NOT_IN_SOURCE]
    if factual == 0:
        attribution = None
        support_rate = None
    else:
        support_rate = counts[SUPPORTED] / factual
        attribution = 1.0 + 4.0 * support_rate
    return {
        "claims": normalized,
        "counts": counts,
        "n_factual": factual,
        "support_rate": support_rate,
        "attribution": attribution,
    }


def score_payload(
    payload: Mapping[str, Any],
    statements: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise GroundedJudgeParseError("claims must be a non-empty list")
    applied = apply_claim_labels(claims, [str(item["id"]) for item in statements])
    scores: Dict[str, Optional[float]] = {"attribution": applied["attribution"]}
    for axis in JUDGED_AXES:
        scores[axis] = _axis_score(payload, axis)
    return {
        "scores": scores,
        "claims": applied["claims"],
        "counts": applied["counts"],
        "n_factual": applied["n_factual"],
        "support_rate": applied["support_rate"],
    }


def no_source_result(question_id: str, question: str) -> Dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "status": "no_source",
        "rubric": RUBRIC_ID,
        "style_names": list(STYLE_NAMES),
        "scores": None,
        "reason": (
            "The retrieved context was not stored with this answer. "
            "Attribution is undefined, so this item is not scored."
        ),
    }


class GroundedJudge:
    """Calls `complete(messages) -> str`. Does not touch the old evaluator rubric."""

    def __init__(self, complete: Callable[[List[Dict[str, str]]], str]):
        self.complete = complete

    @classmethod
    def from_evaluator(cls, evaluator: Any) -> "GroundedJudge":
        require_style(getattr(evaluator, "provider", ""), GROUNDED_ID)

        def complete(messages: List[Dict[str, str]]) -> str:
            return evaluator._create_chat_completion(
                messages=messages,
                max_tokens=1600,
                temperature=0.0,
            )
        return cls(complete)

    def judge_one(
        self,
        question: str,
        answer: str,
        evidence: Optional[Mapping[str, Any]],
        question_id: str = "",
    ) -> Dict[str, Any]:
        statements = source_statements(evidence)
        if not statements:
            return no_source_result(question_id, question)
        if not str(answer or "").strip():
            return {
            "question_id": question_id,
            "question": question,
            "status": "empty_answer",
            "rubric": RUBRIC_ID,
            "style_names": list(STYLE_NAMES),
                "scores": None,
                "reason": "The answer text is empty.",
            }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, answer, statements)},
        ]
        raw = self.complete(messages)
        parsed = score_payload(_extract_json_object(raw), statements)
        return {
            "question_id": question_id,
            "question": question,
            "status": "scored",
            "rubric": RUBRIC_ID,
            "style_names": list(STYLE_NAMES),
            "n_source_statements": len(statements),
            "scores": parsed["scores"],
            "support_rate": parsed["support_rate"],
            "counts": parsed["counts"],
            "claims": parsed["claims"],
            "judge_raw": raw,
        }

    def judge_pair(
        self,
        question: str,
        answer_a: str,
        evidence_a: Optional[Mapping[str, Any]],
        answer_b: str,
        evidence_b: Optional[Mapping[str, Any]],
        question_id: str = "",
    ) -> Dict[str, Any]:
        statements_a = source_statements(evidence_a)
        statements_b = source_statements(evidence_b)
        if not statements_a or not statements_b:
            return {
                "question_id": question_id,
                "status": "no_source",
                "rubric": RUBRIC_ID,
                "style_names": list(STYLE_NAMES),
                "preferences": None,
                "reason": "Both answers need a stored source card before a pairwise comparison.",
            }
        order_ab = _pairwise_call(self.complete, question, answer_a, statements_a, answer_b, statements_b)
        order_ba = _pairwise_call(self.complete, question, answer_b, statements_b, answer_a, statements_a)
        return {
            "question_id": question_id,
            "status": "scored",
            "rubric": RUBRIC_ID,
            "style_names": list(STYLE_NAMES),
            "protocol": "position_swap",
            "preferences": reconcile_position_swap(order_ab, order_ba),
            "order_ab": order_ab,
            "order_ba": order_ba,
        }


def _pairwise_call(complete, question, answer_a, statements_a, answer_b, statements_b) -> Dict[str, str]:
    raw = complete([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_pairwise_prompt(
            question, answer_a, statements_a, answer_b, statements_b
        )},
    ])
    payload = _extract_json_object(raw)
    decided = {}
    for axis in ("attribution",) + JUDGED_AXES:
        token = _pref_token(payload.get(axis))
        if token is None:
            raise GroundedJudgeParseError(f"pairwise {axis} was not A, B, or tie")
        decided[axis] = token
    return decided


def filled_answer_fields(row: Mapping[str, Any]) -> List[str]:
    return [
        key for key, value in row.items()
        if key.endswith("_answer") and isinstance(value, Mapping) and value.get("answer")
    ]


def answer_payload(row: Mapping[str, Any], field: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    """Pick the answer object inside a generation row. Never returns the method name."""
    if field:
        payload = row.get(field)
        return payload if isinstance(payload, Mapping) else None
    if isinstance(row.get("answer"), str) and "evidence" in row:
        return row
    filled = filled_answer_fields(row)
    if len(filled) == 1:
        return row[filled[0]]
    # mst rows store the same text under keo_answer and graphrag_answer.
    primary = [key for key in filled if key != "graphrag_answer"]
    if len(primary) == 1:
        return row[primary[0]]
    return None


def is_sensemaking_row(row: Mapping[str, Any]) -> bool:
    """Same cut as the holistic judge: global items, not knowledge-to-action."""
    if row.get("question_type") == "actionable":
        return False
    if row.get("question_category") == "action_specific":
        return False
    return True


def load_answer_rows(path: str) -> List[Dict[str, Any]]:
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("answers"), list):
        return payload["answers"]
    raise ValueError(f"{path} is not an answers list")


def _write_document(path: str, document: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _document_shell(answers_path: str, field: Optional[str], model: str) -> Dict[str, Any]:
    return {
        "rubric": RUBRIC_ID,
        "style_names": list(STYLE_NAMES),
        "not_comparable_to": NOT_COMPARABLE_TO,
        "answers_file": answers_path,
        "field": field,
        "judge_model": model,
        "items": [],
    }


def _recount(document: Dict[str, Any]) -> Dict[str, Any]:
    items = document.get("items") or []
    document["n_items"] = len(items)
    document["n_scored"] = sum(1 for item in items if item.get("status") == "scored")
    document["n_no_source"] = sum(1 for item in items if item.get("status") == "no_source")
    document["n_ambiguous"] = sum(1 for item in items if item.get("status") == "ambiguous_record")
    document["n_parse_error"] = sum(1 for item in items if item.get("status") == "parse_error")
    document["timestamp"] = datetime.now(timezone.utc).isoformat()
    return document


def _load_resume(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as handle:
            prior = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    kept = {}
    for item in prior.get("items") or []:
        if item.get("status") in ("scored", "no_source", "empty_answer") and item.get("question_id"):
            kept[item["question_id"]] = item
    return kept


def judge_file(
    answers_path: str,
    output_path: str,
    evaluator: Any,
    field: Optional[str] = None,
    limit: Optional[int] = None,
    sensemaking_only: bool = False,
) -> Dict[str, Any]:
    """Score one answers file. Eight-way (or KEO_JUDGE_WORKERS) concurrency comes from the evaluator.

    Already scored question ids in output_path are kept. A parse failure is stored
    and is not filled with a midpoint. Holistic evaluation files are not written.
    """
    rows = load_answer_rows(answers_path)
    if sensemaking_only:
        rows = [row for row in rows if is_sensemaking_row(row)]
    if limit is not None:
        rows = rows[:limit]
    judge = GroundedJudge.from_evaluator(evaluator)
    resumed = _load_resume(output_path)
    document = _document_shell(answers_path, field, getattr(evaluator, "model", ""))
    items: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    todo: List[tuple] = []
    for index, row in enumerate(rows):
        question_id = str(row.get("question_id") or "")
        if question_id and question_id in resumed:
            items[index] = resumed[question_id]
            continue
        payload = answer_payload(row, field)
        if payload is None:
            items[index] = {
                "question_id": question_id,
                "status": "ambiguous_record",
                "rubric": RUBRIC_ID,
                "style_names": list(STYLE_NAMES),
                "scores": None,
                "reason": "Pass --field when a row contains more than one answer.",
            }
            continue
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else None
        question = str(payload.get("question") or row.get("question") or "")
        answer = str(payload.get("answer") or "")
        if not source_statements(evidence) or not answer.strip():
            items[index] = judge.judge_one(question, answer, evidence, question_id)
            continue
        todo.append((index, question_id, question, answer, evidence))

    lock = threading.Lock()

    def flush() -> None:
        document["items"] = [item for item in items if item is not None]
        _write_document(output_path, _recount(document))

    def score_job(job):
        index, question_id, question, answer, evidence = job
        try:
            result = judge.judge_one(question, answer, evidence, question_id)
        except GroundedJudgeParseError as exc:
            result = {
                "question_id": question_id,
                "question": question,
                "status": "parse_error",
                "rubric": RUBRIC_ID,
                "style_names": list(STYLE_NAMES),
                "scores": None,
                "reason": str(exc),
            }
        with lock:
            items[index] = result
            flush()
        return result

    if todo:
        evaluator._map_judge_calls(todo, score_job, f"Grounded/Source-card {os.path.basename(os.path.dirname(answers_path))}")
    flush()
    return document


def iter_current_arms(root: str = ".") -> List[Dict[str, str]]:
    """Graph arms under the current Haiku answer trees that stored a source card."""
    found: List[Dict[str, str]] = []
    for relative in CURRENT_ROOTS:
        base = os.path.join(root, relative)
        if not os.path.isdir(base):
            continue
        for arm in sorted(os.listdir(base)):
            answers = os.path.join(base, arm, "answers.json")
            if not os.path.isfile(answers):
                continue
            rows = [row for row in load_answer_rows(answers) if is_sensemaking_row(row)]
            if not rows:
                continue
            payload = answer_payload(rows[0])
            if payload is None:
                continue
            evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else None
            if not source_statements(evidence):
                continue
            field = None
            filled = filled_answer_fields(rows[0])
            primary = [key for key in filled if key != "graphrag_answer"]
            if len(primary) == 1:
                field = primary[0]
            elif len(filled) == 1:
                field = filled[0]
            found.append({
                "answers": answers,
                "output": os.path.join(base, arm, "grounded_sourcecard.json"),
                "field": field or "",
                "arm": f"{os.path.basename(base)}/{arm}",
            })
    return found


def judge_current(evaluator: Any, root: str = ".", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    documents = []
    for arm in iter_current_arms(root):
        print(f"=== {arm['arm']} field={arm['field']} ===")
        documents.append(judge_file(
            arm["answers"],
            arm["output"],
            evaluator,
            field=arm["field"] or None,
            limit=limit,
            sensemaking_only=True,
        ))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score answers with the Grounded judge (Source-card rubric)."
    )
    parser.add_argument("--answers-file", default=None)
    parser.add_argument("--output", default=None, help="New JSON file. Does not replace holistic evaluation files.")
    parser.add_argument("--field", default=None, help="Answer field, e.g. keo_answer.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--current",
        action="store_true",
        help="Haiku-style sweep of the current graph answers (sensemaking items only).",
    )
    parser.add_argument("--workers", type=int, default=None, help="Concurrent judge calls. Default is KEO_JUDGE_WORKERS.")
    args = parser.parse_args()
    if args.workers is not None:
        os.environ["KEO_JUDGE_WORKERS"] = str(args.workers)

    from evaluator import SensemakingEvaluator
    from judge_policy import resolve_judge_defaults

    provider, model, grade = resolve_judge_defaults(provider=args.provider, model=args.model, final=False)
    require_style(provider, GROUNDED_ID)
    evaluator = SensemakingEvaluator(provider=provider, model=model, judge_grade=grade)
    if args.current:
        documents = judge_current(evaluator, limit=args.limit)
        scored = sum(doc["n_scored"] for doc in documents)
        print(f"style={STYLE_NAMES[0]}/{STYLE_NAMES[1]} arms={len(documents)} scored={scored} workers={evaluator.judge_workers}")
        return
    if not args.answers_file or not args.output:
        raise SystemExit("Pass --current, or both --answers-file and --output.")
    document = judge_file(
        args.answers_file,
        args.output,
        evaluator,
        field=args.field,
        limit=args.limit,
    )
    print(
        f"style={STYLE_NAMES[0]}/{STYLE_NAMES[1]} scored={document['n_scored']} "
        f"no_source={document['n_no_source']} ambiguous={document['n_ambiguous']} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
