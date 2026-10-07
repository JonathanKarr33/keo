#!/usr/bin/env python3
"""Fill human_validation batch ratings via Claude Code (llm_proxy, not expert).

Rates questions (1--5 rubrics) and triples (plausible / relation_fit) with no
GPU. Writes a separate ``*_llm_proxy.json`` so the unlabeled expert file stays
clean. Paper must say llm_proxy, not human gold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_QA = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _QA not in sys.path:
    sys.path.insert(0, _QA)

from claude_cli import DEFAULT_CLI_JUDGE_MODEL, claude_cli_chat

Q_FIELDS = ("clarity", "difficulty", "domain_relevance", "answerability_without_leak")
TRIPLE_FIELDS = ("plausible", "relation_fit")
VALID_YN = {"yes", "no", "unsure"}


Q_SYSTEM = (
    "You rate aviation-maintenance QA benchmark items for a research paper. "
    "Score each question on four integer axes from 1 (worst) to 5 (best):\n"
    "- clarity: wording is unambiguous\n"
    "- difficulty: hard for a domain expert with the corpus (5=hard)\n"
    "- domain_relevance: about aviation maintenance / safety\n"
    "- answerability_without_leak: answerable from public maintenance knowledge "
    "without needing a leaked gold document\n"
    "Reply ONLY with a JSON array of objects "
    '{"i": <int>, "clarity": <1-5>, "difficulty": <1-5>, '
    '"domain_relevance": <1-5>, "answerability_without_leak": <1-5>}. '
    "No markdown fences, no prose."
)

T_SYSTEM = (
    "You rate knowledge-graph triples extracted from aviation maintenance text. "
    "For each triple assign:\n"
    "- plausible: yes | no | unsure  (is the fact itself believable in aviation maintenance?)\n"
    "- relation_fit: yes | no | unsure  (does the relation label fit head/tail?)\n"
    "Reply ONLY with a JSON array of objects "
    '{"i": <int>, "plausible": "yes|no|unsure", "relation_fit": "yes|no|unsure"}. '
    "No markdown fences, no prose."
)


def _parse_json_array(text: str) -> List[dict]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON array: {raw[:240]!r}")
        obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, list):
        raise ValueError(f"expected list, got {type(obj)}")
    return [x for x in obj if isinstance(x, dict)]


def _checkpoint(batch: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(batch, f, indent=2)
    os.replace(tmp, path)


def rate_questions(
    batch: dict,
    *,
    model: str,
    batch_size: int,
    sleep_s: float,
    out_path: str,
) -> None:
    items = batch["questions"]
    pending = [i for i, it in enumerate(items) if it.get("clarity") is None]
    print(f"questions pending={len(pending)}/{len(items)}", flush=True)
    for start in range(0, len(pending), batch_size):
        chunk = [i for i in pending[start : start + batch_size] if items[i].get("clarity") is None]
        if not chunk:
            continue
        lines = [
            "Rate each question. Index i matches the number below.",
            "",
        ]
        for i in chunk:
            lines.append(f"{i}. {items[i].get('question','')}")
        user = "\n".join(lines)
        print(f"  Q call indices {chunk[0]}..{chunk[-1]} …", flush=True)
        t0 = time.time()
        text, observed, _ = claude_cli_chat(
            messages=[
                {"role": "system", "content": Q_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=model,
            timeout_s=180,
        )
        parsed = {int(r["i"]): r for r in _parse_json_array(text) if "i" in r}
        missing = [i for i in chunk if i not in parsed]
        if missing:
            raise ValueError(f"missing question ratings for {missing[:8]}")
        for i in chunk:
            r = parsed[i]
            for f in Q_FIELDS:
                v = int(r[f])
                if not 1 <= v <= 5:
                    raise ValueError(f"bad {f}={v} for i={i}")
                items[i][f] = v
            items[i]["notes"] = items[i].get("notes") or "llm_proxy"
        print(f"  Q ok in {time.time()-t0:.1f}s observed={observed}", flush=True)
        _checkpoint(batch, out_path)
        if sleep_s:
            time.sleep(sleep_s)


def rate_triples(
    batch: dict,
    *,
    model: str,
    batch_size: int,
    sleep_s: float,
    out_path: str,
) -> None:
    items = batch["triples"]
    pending = [i for i, it in enumerate(items) if it.get("plausible") is None]
    print(f"triples pending={len(pending)}/{len(items)}", flush=True)
    for start in range(0, len(pending), batch_size):
        chunk = [i for i in pending[start : start + batch_size] if items[i].get("plausible") is None]
        if not chunk:
            continue
        lines = [
            "Rate each triple. Index i matches the number below.",
            "",
        ]
        for i in chunk:
            it = items[i]
            lines.append(f"{i}. {it['head']} | {it['relation']} | {it['tail']}")
        user = "\n".join(lines)
        print(f"  T call indices {chunk[0]}..{chunk[-1]} …", flush=True)
        t0 = time.time()
        text, observed, _ = claude_cli_chat(
            messages=[
                {"role": "system", "content": T_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=model,
            timeout_s=180,
        )
        parsed = {int(r["i"]): r for r in _parse_json_array(text) if "i" in r}
        missing = [i for i in chunk if i not in parsed]
        if missing:
            raise ValueError(f"missing triple ratings for {missing[:8]}")
        for i in chunk:
            r = parsed[i]
            for f in TRIPLE_FIELDS:
                v = str(r[f]).strip().lower()
                if v not in VALID_YN:
                    raise ValueError(f"bad {f}={v} for i={i}")
                items[i][f] = v
            items[i]["notes"] = items[i].get("notes") or "llm_proxy"
        print(f"  T ok in {time.time()-t0:.1f}s observed={observed}", flush=True)
        _checkpoint(batch, out_path)
        if sleep_s:
            time.sleep(sleep_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch",
        default="human_validation/ratings/batch001_to_rate.json",
    )
    p.add_argument(
        "--out",
        default="human_validation/ratings/batch001_llm_proxy.json",
    )
    p.add_argument("--model", default=DEFAULT_CLI_JUDGE_MODEL)
    p.add_argument("--q-batch-size", type=int, default=8)
    p.add_argument("--t-batch-size", type=int, default=20)
    p.add_argument("--sleep", type=float, default=0.3)
    args = p.parse_args()

    if os.path.isfile(args.out):
        batch = json.load(open(args.out))
        print(f"resume from {args.out}", flush=True)
    else:
        batch = json.load(open(args.batch))
        batch["annotator"] = f"llm_proxy:{args.model}"
        batch["adjudication"] = {
            "kind": "llm_proxy_not_expert",
            "model": args.model,
            "note": "Not human gold; replace with expert ratings before claiming validation.",
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    rate_questions(
        batch,
        model=args.model,
        batch_size=args.q_batch_size,
        sleep_s=args.sleep,
        out_path=args.out,
    )
    rate_triples(
        batch,
        model=args.model,
        batch_size=args.t_batch_size,
        sleep_s=args.sleep,
        out_path=args.out,
    )
    _checkpoint(batch, args.out)
    nq = sum(1 for q in batch["questions"] if q.get("clarity") is not None)
    nt = sum(1 for t in batch["triples"] if t.get("plausible") is not None)
    print(f"✓ {args.out}  questions={nq}/{len(batch['questions'])}  triples={nt}/{len(batch['triples'])}")


if __name__ == "__main__":
    main()
