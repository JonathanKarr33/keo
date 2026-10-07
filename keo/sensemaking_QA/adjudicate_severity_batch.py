#!/usr/bin/env python3
"""Fill human_class on a severity adjudication batch via Claude Code (LLM proxy).

This is NOT expert adjudication. It is an independent labeler from the lexicon
prior (different model, no prior_class in the prompt) so we can score critical
recall against something other than the retrieval prior. Paper must say
``llm_proxy``, not ``expert``.

Resumable: skips items that already have human_class. Writes in place (or --out).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from claude_cli import DEFAULT_CLI_JUDGE_MODEL, claude_cli_chat
from severity_lexicon import SEVERITY_CLASSES

VALID = set(SEVERITY_CLASSES)

SYSTEM = (
    "You are an independent aviation safety labeler. Assign one airworthiness "
    "failure-condition class to each maintenance fact (triple), for the worst "
    "plausible outcome of that fact alone.\n"
    "Classes (pick exactly one):\n"
    "- catastrophic: loss of aircraft / multiple fatalities plausible\n"
    "- hazardous: large reduction in safety margin, serious injury plausible\n"
    "- major: significant reduction in margins / crew workload\n"
    "- minor: slight reduction in margins, routine crew response\n"
    "- none: no safety effect\n"
    "Do not use tools. Reply with ONLY a JSON array of objects "
    '{"i": <int>, "human_class": "<class>"}. No markdown fences, no prose.'
)


def _parse_labels(text: str, expected: List[int]) -> Dict[int, str]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    # Prefer a JSON array; fall back to first [...] span.
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON array in response: {raw[:240]!r}")
        obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, list):
        raise ValueError(f"expected JSON array, got {type(obj)}")
    out: Dict[int, str] = {}
    for row in obj:
        if not isinstance(row, dict):
            continue
        i = row.get("i")
        cls = str(row.get("human_class") or "").strip().lower()
        if i is None or cls not in VALID:
            continue
        out[int(i)] = cls
    missing = [i for i in expected if i not in out]
    if missing:
        raise ValueError(f"missing labels for indices {missing[:8]}… ({len(missing)} total)")
    return out


def _batch_prompt(items: List[Dict[str, Any]], indices: List[int]) -> str:
    lines = [
        "Label each triple. Index i matches the number below. "
        "Ignore any lexicon prior you might invent; judge the fact only.",
        "",
    ]
    for i in indices:
        it = items[i]
        lines.append(f"{i}. {it['head']} | {it['relation']} | {it['tail']}")
    return "\n".join(lines)


def _checkpoint(batch: Dict[str, Any], path: Optional[str], annotator_id: str, model: str) -> None:
    items = batch["items"]
    batch["adjudication"] = {
        "kind": "llm_proxy_not_expert",
        "annotator_id": annotator_id,
        "model_requested": model,
        "n_labeled": sum(1 for it in items if it.get("human_class")),
        "n_pending": sum(1 for it in items if not it.get("human_class")),
        "note": (
            "Labels come from Claude Code with no prior_class in the prompt. "
            "Do not cite as expert adjudication; use for non-circular recall "
            "until humans replace human_class."
        ),
    }
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(batch, f, indent=2)
    os.replace(tmp, path)


def adjudicate(
    batch: Dict[str, Any],
    *,
    model: str,
    batch_size: int,
    annotator_id: str,
    sleep_s: float,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    items = batch["items"]
    pending = [i for i, it in enumerate(items) if not it.get("human_class")]
    print(
        f"items={len(items)}  pending={len(pending)}  batch_size={batch_size}  model={model}",
        flush=True,
    )
    n_calls = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        # Re-check: may have been filled mid-run via resume.
        chunk = [i for i in chunk if not items[i].get("human_class")]
        if not chunk:
            continue
        user = _batch_prompt(items, chunk)
        print(f"  calling Claude on indices {chunk[0]}..{chunk[-1]} ({len(chunk)}) …", flush=True)
        t0 = time.time()
        text, observed, meta = claude_cli_chat(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            model=model,
            timeout_s=120,
        )
        labels = _parse_labels(text, chunk)
        for i, cls in labels.items():
            items[i]["human_class"] = cls
            items[i]["annotator_id"] = annotator_id
            items[i]["notes"] = items[i].get("notes") or "llm_proxy_adjudication"
        n_calls += 1
        print(
            f"  call {n_calls}: labeled {len(chunk)} in {time.time()-t0:.1f}s "
            f"(observed={observed}, keys={sorted(set(labels.values()))})",
            flush=True,
        )
        _checkpoint(batch, checkpoint_path, annotator_id, model)
        if sleep_s > 0:
            time.sleep(sleep_s)
    _checkpoint(batch, checkpoint_path, annotator_id, model)
    return batch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch",
        default="human_validation/ratings/severity_batch001.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: overwrite --batch in place (after writing a .bak)",
    )
    p.add_argument("--model", default=DEFAULT_CLI_JUDGE_MODEL)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument(
        "--annotator-id",
        default=None,
        help="Default: llm_proxy:<model>",
    )
    p.add_argument("--limit", type=int, default=None, help="Only label first N pending (smoke)")
    args = p.parse_args()

    batch = json.load(open(args.batch))
    if args.limit is not None:
        # Temporarily clear labels beyond limit so we only fill a prefix of pending.
        pending = [i for i, it in enumerate(batch["items"]) if not it.get("human_class")]
        keep = set(pending[: args.limit])
        # Nothing to do for skip; adjudicate already skips labeled.

    annotator = args.annotator_id or f"llm_proxy:{args.model}"
    out_path = args.out or args.batch

    # Backup once before first write.
    bak = args.batch + ".pre_llm_proxy.bak"
    if not os.path.exists(bak):
        with open(bak, "w") as f:
            json.dump(batch, f, indent=2)
        print(f"backup → {bak}")

    # Optional smoke: only first --limit pending indices get labeled; others stay null.
    if args.limit is not None:
        pending = [i for i, it in enumerate(batch["items"]) if not it.get("human_class")]
        freeze = set(pending[args.limit :])
        # Mark frozen with a sentinel we strip after? Simpler: temporarily set a skip flag.
        for i in freeze:
            batch["items"][i]["_skip"] = True
        # Patch adjudicate pending filter by pre-setting a fake class then clearing — messy.
        # Instead rewrite pending list via temporary human_class placeholder.
        for i in freeze:
            batch["items"][i]["human_class"] = "__skip__"

    try:
        batch = adjudicate(
            batch,
            model=args.model,
            batch_size=args.batch_size,
            annotator_id=annotator,
            sleep_s=args.sleep,
            checkpoint_path=out_path,
        )
    finally:
        if args.limit is not None:
            for it in batch["items"]:
                if it.get("human_class") == "__skip__":
                    it["human_class"] = None
                it.pop("_skip", None)

    labeled = sum(1 for it in batch["items"] if it.get("human_class"))
    print(f"✓ {out_path}  labeled={labeled}/{len(batch['items'])}", flush=True)


if __name__ == "__main__":
    main()
