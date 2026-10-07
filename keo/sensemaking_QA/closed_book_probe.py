#!/usr/bin/env python3
"""Closed-book contamination probe — prompt pack + optional live local LLM pass.

ACL_STRATEGY §4.2: ask the answerer with *no* retrieved context, then measure
how much of the answer is literally recoverable from the OMIn/FAA corpus via
n-gram overlap (same metric as contamination_probe.py).

Default mode builds prompts only (no LLM). ``--live`` calls an OpenAI-compatible
local server (Phi-4 / mlx-lm), is resumable, and refuses to invent a
contamination verdict from prompts alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

from contamination_probe import load_corpus_text, ngrams, tokens
from openai import OpenAI

SYSTEM = (
    "You are answering from parametric memory only. Do not retrieve, browse, or "
    "invent citations to documents you have not been shown in this message. If you "
    "do not know, say you do not know."
)

_IDK = re.compile(
    r"\b(i do not know|i don't know|do not know|don't know|cannot answer|"
    r"can't answer|no information|insufficient (knowledge|information)|"
    r"unable to (answer|determine))\b",
    re.I,
)


def closed_book_prompt(question: str) -> List[Dict[str, str]]:
    """Chat messages for a closed-book ask — no corpus context in the user turn."""
    q = (question or "").strip()
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Answer the following aviation-maintenance sensemaking question "
                "using only what you already know. Do not claim to have read a "
                "specific FAA or MaintNet record.\n\n"
                f"Question: {q}"
            ),
        },
    ]


def answer_corpus_overlap(answer: str, corpus_text: str, n: int = 5) -> Dict:
    """Fraction of answer n-grams that also appear in the retrieval corpus."""
    a_ng = ngrams(tokens(answer), n)
    c_ng = ngrams(tokens(corpus_text), n)
    if not a_ng:
        return {"n_grams": 0, "hits_in_corpus": 0, "overlap": 0.0, "ngram_n": n}
    hits = sum(1 for g in a_ng if g in c_ng)
    return {
        "n_grams": len(a_ng),
        "hits_in_corpus": hits,
        "overlap": hits / len(a_ng),
        "ngram_n": n,
    }


def looks_like_idk(answer: str) -> bool:
    text = (answer or "").strip()
    if len(text) < 40 and _IDK.search(text):
        return True
    return bool(_IDK.search(text[:240]))


def build_probe_pack(questions: List[dict]) -> dict:
    rows = []
    for q in questions:
        rows.append(
            {
                "question_id": q.get("id"),
                "question": q.get("question"),
                "messages": closed_book_prompt(q.get("question") or ""),
            }
        )
    return {
        "n_questions": len(rows),
        "system": SYSTEM,
        "note": (
            "Prompts only — no model was called. Run with --live when the "
            "answerer is free; score answers with answer_corpus_overlap."
        ),
        "items": rows,
    }


def _chat(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"enable_thinking": False},
    )
    choice = resp.choices[0].message
    text = (choice.content or "").strip()
    # strip common chat terminators leaked by some servers
    for term in ("<|end|>", "<|eot_id|>", "</s>"):
        if text.endswith(term):
            text = text[: -len(term)].strip()
    if not text:
        raise RuntimeError("empty closed-book completion")
    return text


def run_live(
    questions: List[dict],
    *,
    corpus_text: str,
    model: str,
    base_url: str,
    out_path: str,
    ngram_n: int,
    max_tokens: int,
    temperature: float,
    limit: Optional[int],
) -> dict:
    if limit is not None:
        questions = questions[:limit]

    # Resume from prior partial run if present.
    prior_by_id: Dict[str, dict] = {}
    if os.path.isfile(out_path):
        try:
            prior = json.load(open(out_path))
            for row in prior.get("per_question") or []:
                if row.get("answer") and not row.get("error"):
                    prior_by_id[str(row.get("question_id"))] = row
        except (OSError, json.JSONDecodeError):
            prior_by_id = {}

    client = OpenAI(api_key="EMPTY", base_url=base_url)
    # Preflight
    smoke = _chat(
        client,
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: Ready"}],
        max_tokens=8,
        temperature=0.0,
    )
    print(f"preflight ok → {smoke!r}", flush=True)

    rows = []
    for i, q in enumerate(questions):
        qid = str(q.get("id") or f"q{i}")
        if qid in prior_by_id:
            rows.append(prior_by_id[qid])
            print(f"  [{i+1}/{len(questions)}] {qid} SKIP (cached)", flush=True)
            continue
        messages = closed_book_prompt(q.get("question") or "")
        t0 = time.time()
        try:
            answer = _chat(
                client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            ov = answer_corpus_overlap(answer, corpus_text, n=ngram_n)
            row = {
                "question_id": qid,
                "question": q.get("question"),
                "answer": answer,
                "n_chars": len(answer),
                "idk": looks_like_idk(answer),
                "latency_s": round(time.time() - t0, 2),
                "error": None,
                **{f"overlap_{k}": v for k, v in ov.items()},
            }
            # flatten overlap keys cleaner
            row = {
                "question_id": qid,
                "question": q.get("question"),
                "answer": answer,
                "n_chars": len(answer),
                "idk": looks_like_idk(answer),
                "latency_s": round(time.time() - t0, 2),
                "error": None,
                "overlap": ov,
            }
            print(
                f"  [{i+1}/{len(questions)}] {qid} "
                f"chars={len(answer)} idk={row['idk']} "
                f"ov5={ov['overlap']:.3f} ({row['latency_s']}s)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — record and continue
            row = {
                "question_id": qid,
                "question": q.get("question"),
                "answer": None,
                "n_chars": 0,
                "idk": None,
                "latency_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}",
                "overlap": None,
            }
            print(f"  [{i+1}/{len(questions)}] {qid} ERROR {row['error']}", flush=True)
        rows.append(row)

        # checkpoint
        summary = _summarise(rows, model=model, base_url=base_url, ngram_n=ngram_n)
        summary["per_question"] = rows
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, out_path)

    summary = _summarise(rows, model=model, base_url=base_url, ngram_n=ngram_n)
    summary["per_question"] = rows
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _summarise(rows: List[dict], *, model: str, base_url: str, ngram_n: int) -> dict:
    ok = [r for r in rows if r.get("answer") and not r.get("error")]
    overlaps = [float(r["overlap"]["overlap"]) for r in ok if r.get("overlap")]
    idk_n = sum(1 for r in ok if r.get("idk"))
    n = len(ok)
    return {
        "n_questions": len(rows),
        "n_answered": n,
        "n_errors": sum(1 for r in rows if r.get("error")),
        "model": model,
        "base_url": base_url,
        "ngram_n": ngram_n,
        "mean_answer_chars": (sum(r["n_chars"] for r in ok) / n) if n else None,
        "frac_idk": (idk_n / n) if n else None,
        "mean_answer_corpus_overlap": (sum(overlaps) / len(overlaps)) if overlaps else None,
        "median_answer_corpus_overlap": (
            sorted(overlaps)[len(overlaps) // 2] if overlaps else None
        ),
        "frac_overlap_gt_0.5": (
            sum(1 for o in overlaps if o > 0.5) / len(overlaps) if overlaps else None
        ),
        "reading": (
            "Closed-book answers from parametric memory only (no retrieval). "
            "mean_answer_corpus_overlap is the fraction of answer n-grams that "
            "also appear in the FAA/OMIn corpus — high values suggest the model "
            "is regurgitating corpus text; low values + high frac_idk suggest it "
            "does not have the records memorised. Not a Claude score; not a "
            "human contamination audit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call local OpenAI-compatible answerer and score corpus overlap",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        help="CSV path(s) for answer↔corpus n-gram scoring (required with --live)",
    )
    parser.add_argument(
        "--text-cols",
        default="c5,c119,narrative,text,problem,action,description",
    )
    parser.add_argument("--n", type=int, default=5, help="n-gram size for overlap")
    parser.add_argument("--model", default="mlx-community/phi-4-4bit")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    questions = json.load(open(args.questions))
    if args.limit and not args.live:
        questions = questions[: args.limit]

    if not args.live:
        pack = build_probe_pack(questions)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump(pack, open(args.out, "w"), indent=2)
        print(f"n={pack['n_questions']} closed-book prompts → {args.out}")
        print(pack["note"])
        return

    if not args.corpus:
        raise SystemExit("--live requires at least one --corpus CSV")
    corpus = load_corpus_text(args.corpus, args.text_cols.split(","))
    print(f"corpus chars={len(corpus)} from {args.corpus}", flush=True)
    summary = run_live(
        questions,
        corpus_text=corpus,
        model=args.model,
        base_url=args.base_url,
        out_path=args.out,
        ngram_n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit,
    )
    print(
        f"n_answered={summary['n_answered']}  "
        f"mean_ov={summary['mean_answer_corpus_overlap']}  "
        f"frac_idk={summary['frac_idk']}  "
        f"errors={summary['n_errors']}"
    )
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
