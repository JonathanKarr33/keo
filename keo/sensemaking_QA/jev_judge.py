#!/usr/bin/env python3
"""Score both KEO rubrics with Jev on OpenRouter. Claude is not called.

Holistic / Sensemaking: five 1-5 scores. The source card is not in the state.
Grounded / Source-card: attribution, relevance, coverage, specificity. The
source card is the only evidence. Jev returns a probability-weighted score,
not prose and not an extracted claim list, so attribution here is that score
rather than a hand-counted support rate.

Writes jev_holistic.json and jev_sourcecard.json. Does not touch
global_sensemaking_evaluation_*.json or grounded_sourcecard.json.
Do not pool these numbers with Haiku, Sonnet, or GPT.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from grounded_judge import (
    CURRENT_ROOTS,
    answer_payload,
    filled_answer_fields,
    is_sensemaking_row,
    load_answer_rows,
    render_source_card,
    source_statements,
)
from jev_client import JEV_MODEL, JevError, decide, rubric_score

HOLISTIC_ID = "jev_holistic"
GROUNDED_ID = "jev_grounded"

_LEVELS = {
    "global_perspective": (
        "Does the answer show dataset-wide patterns? A longer answer is not a higher score.",
        [
            "No dataset-wide pattern is described.",
            "One local fact is treated as a pattern.",
            "Some patterns are named, and they stay narrow.",
            "Dataset-wide patterns are described, with a minor gap.",
            "The answer is organized around dataset-wide patterns.",
        ],
    ),
    "theme_identification": (
        "Are the major themes clearly identified? A longer list is not a higher score.",
        [
            "No theme is identified.",
            "A topic word appears without a theme.",
            "One theme is clear and others the question needs are missing.",
            "The major themes are identified, with a minor omission.",
            "The major themes are identified and distinguished from each other.",
        ],
    ),
    "synthesis_quality": (
        "How well are separate points drawn together? Length is not synthesis.",
        [
            "The points are unrelated.",
            "Points are listed and not connected.",
            "Some points are connected and others sit side by side.",
            "The points are drawn into one account, with a minor gap.",
            "Separate points are drawn into one account.",
        ],
    ),
    "strategic_value": (
        "Does the answer help a high-level maintenance decision? Generic advice is not a high score.",
        [
            "Nothing a decision could use.",
            "Advice that would fit any maintenance question.",
            "One usable point, and the rest is generic.",
            "The answer supports a decision, with a minor gap.",
            "A maintainer could use the answer in a high-level decision.",
        ],
    ),
    "pattern_recognition": (
        "Are relationships among the themes identified? Naming more themes is not a higher score.",
        [
            "No relationship is identified.",
            "Two items are mentioned and not related.",
            "One relationship is identified.",
            "The main relationships are identified, with a minor gap.",
            "The relationships among the themes are identified.",
        ],
    ),
    "attribution": (
        "Of the factual claims in the answer, how many does the source card state? "
        "Training data and general aviation knowledge are not evidence. "
        "A claim that is true in general but absent from the card is not supported.",
        [
            "Almost none of the factual claims are stated by the source card.",
            "A small minority of the factual claims are stated by the source card.",
            "About half of the factual claims are stated by the source card.",
            "Most factual claims are stated by the source card.",
            "The factual claims are stated by the source card.",
        ],
    ),
    "relevance": (
        "Does the answer address the question that was asked? Ignore whether the claims are in the source.",
        [
            "The answer does not address the question.",
            "The answer shares the topic and responds to a different question.",
            "A major part of the question is unanswered.",
            "The question is answered, with a minor omission.",
            "The question that was asked is answered.",
        ],
    ),
    "coverage": (
        "Of the source-card statements that bear on the question, how many does the answer use? "
        "Facts that are not on the card do not raise this score.",
        [
            "None of the relevant statements are used.",
            "One relevant statement is used and others of equal bearing are ignored.",
            "Some relevant statements are used and others of equal bearing are omitted.",
            "Most relevant statements are used.",
            "The relevant statements are used, without themes the card does not contain.",
        ],
    ),
    "specificity": (
        "Could a maintainer act on the causes or steps, and is each one tied to a source-card statement? "
        "Length is not specificity.",
        [
            "The advice would fit any maintenance question.",
            "The advice is domain-plausible and tied to no source-card statement.",
            "One concrete cause or action appears, weakly tied to the card.",
            "Concrete causes or actions appear, and each is tied to a source-card statement.",
            "A maintainer could carry out the stated causes or actions, and each is tied to the card.",
        ],
    ),
}

HOLISTIC_AXES = (
    "global_perspective",
    "theme_identification",
    "synthesis_quality",
    "strategic_value",
    "pattern_recognition",
)
GROUNDED_AXES = ("attribution", "relevance", "coverage", "specificity")


def questions_for(axes: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    payload = {}
    for axis in axes:
        instructions, criteria = _LEVELS[axis]
        payload[axis] = {
            "type": "score",
            "instructions": instructions,
            "criteria": list(criteria),
        }
    return payload


def state_for(question: str, answer: str, source_card: str = "") -> Dict[str, str]:
    state = {"question": question.strip(), "answer": answer.strip()}
    if source_card:
        state["source_card"] = source_card
    return state


def scores_from_answers(answers: Mapping[str, Any], axes: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {}
    for axis in axes:
        block = answers.get(axis)
        if not isinstance(block, Mapping) or block.get("type") != "score":
            raise JevError(f"{axis} was not a score answer")
        n_levels = len(_LEVELS[axis][1])
        parsed[axis] = {
            "score": rubric_score(block.get("score"), n_levels),
            "confidence": block.get("confidence"),
            "probabilities": block.get("probabilities"),
        }
    return parsed


def _write(path: str, document: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _resume(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as handle:
            prior = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        item["question_id"]: item
        for item in prior.get("items") or []
        if item.get("status") == "scored" and item.get("question_id")
    }


def _primary_field(row: Mapping[str, Any]) -> Optional[str]:
    filled = filled_answer_fields(row)
    primary = [key for key in filled if key != "graphrag_answer"]
    if len(primary) == 1:
        return primary[0]
    if len(filled) == 1:
        return filled[0]
    return None


def judge_arm(
    answers_path: str,
    output_path: str,
    *,
    style: str,
    field: Optional[str],
    workers: int,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    if style == "holistic":
        rubric, axes = HOLISTIC_ID, HOLISTIC_AXES
        names = ["Holistic", "Sensemaking"]
    elif style == "grounded":
        rubric, axes = GROUNDED_ID, GROUNDED_AXES
        names = ["Grounded", "Source-card"]
    else:
        raise ValueError(style)
    rows = [row for row in load_answer_rows(answers_path) if is_sensemaking_row(row)]
    if limit is not None:
        rows = rows[:limit]
    questions = questions_for(axes)
    resumed = _resume(output_path)
    items: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    todo = []
    for index, row in enumerate(rows):
        question_id = str(row.get("question_id") or "")
        if question_id in resumed:
            items[index] = resumed[question_id]
            continue
        payload = answer_payload(row, field)
        if payload is None:
            items[index] = {
                "question_id": question_id,
                "status": "ambiguous_record",
                "rubric": rubric,
                "scores": None,
            }
            continue
        question = str(payload.get("question") or row.get("question") or "")
        answer = str(payload.get("answer") or "")
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else None
        statements = source_statements(evidence)
        if style == "grounded" and not statements:
            items[index] = {
                "question_id": question_id,
                "status": "no_source",
                "rubric": rubric,
                "style_names": names,
                "scores": None,
                "reason": "Retrieved context was not stored with this answer.",
            }
            continue
        if not answer.strip():
            items[index] = {
                "question_id": question_id,
                "status": "empty_answer",
                "rubric": rubric,
                "scores": None,
            }
            continue
        card = render_source_card(statements) if style == "grounded" else ""
        todo.append((index, question_id, state_for(question, answer, card)))

    document: Dict[str, Any] = {
        "rubric": rubric,
        "style_names": names,
        "judge_model": JEV_MODEL,
        "provider": "openrouter",
        "not_comparable_to": "Claude holistic files and grounded_sourcecard.json",
        "answers_file": answers_path,
        "field": field,
        "items": [],
    }
    lock = threading.Lock()

    def flush() -> None:
        present = [item for item in items if item is not None]
        document["items"] = present
        document["n_items"] = len(present)
        document["n_scored"] = sum(1 for item in present if item.get("status") == "scored")
        document["n_no_source"] = sum(1 for item in present if item.get("status") == "no_source")
        document["timestamp"] = datetime.now(timezone.utc).isoformat()
        _write(output_path, document)

    def score_job(job):
        index, question_id, state = job
        try:
            payload = decide(state, questions)
            parsed = scores_from_answers(payload.get("answers") or {}, axes)
            point = {axis: parsed[axis]["score"] for axis in axes}
            result = {
                "question_id": question_id,
                "status": "scored",
                "rubric": rubric,
                "style_names": names,
                "scores": point,
                "detail": parsed,
                "model": payload.get("model"),
                "usage": payload.get("usage"),
            }
            if style == "holistic":
                result["overall"] = sum(point.values()) / len(point)
        except JevError as exc:
            if "OpenRouter stopped" in str(exc):
                raise
            result = {
                "question_id": question_id,
                "status": "error",
                "rubric": rubric,
                "scores": None,
                "reason": str(exc),
            }
        with lock:
            items[index] = result
            flush()
        return result

    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        pool_size = max(1, min(workers, len(todo)))
        label = f"Jev {style} {os.path.basename(os.path.dirname(answers_path))}"
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            futures = [pool.submit(score_job, job) for job in todo]
            for future in tqdm(as_completed(futures), total=len(futures), desc=label):
                future.result()
    flush()
    return document


def iter_arms(root: str = ".") -> List[Dict[str, str]]:
    found = []
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
            field = _primary_field(rows[0])
            if not field:
                continue
            found.append({
                "arm": f"{os.path.basename(base)}/{arm}",
                "answers": answers,
                "field": field,
                "dir": os.path.join(base, arm),
            })
    return found


def judge_current(root: str, styles: Sequence[str], workers: int, limit: Optional[int]) -> None:
    for arm in iter_arms(root):
        rows = [row for row in load_answer_rows(arm["answers"]) if is_sensemaking_row(row)]
        payload = answer_payload(rows[0], arm["field"]) if rows else None
        evidence = payload.get("evidence") if isinstance(payload, Mapping) else None
        has_source = bool(source_statements(evidence))
        for style in styles:
            if style == "grounded" and not has_source:
                print(f"skip {arm['arm']} grounded (no source card)")
                continue
            name = "jev_holistic.json" if style == "holistic" else "jev_sourcecard.json"
            output = os.path.join(arm["dir"], name)
            print(f"=== {arm['arm']} {style} field={arm['field']} ===")
            document = judge_arm(
                arm["answers"],
                output,
                style=style,
                field=arm["field"],
                workers=workers,
                limit=limit,
            )
            print(f"  scored={document['n_scored']} -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Holistic and Grounded rubrics with Jev.")
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--style", choices=("both", "holistic", "grounded"), default="both")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.current:
        raise SystemExit("Pass --current.")
    styles = ("holistic", "grounded") if args.style == "both" else (args.style,)
    judge_current(".", styles, args.workers, args.limit)


if __name__ == "__main__":
    main()
