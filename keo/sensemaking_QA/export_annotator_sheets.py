#!/usr/bin/env python3
"""Build the two expert sheets. No model calls.

Answer calibration: 40 sensemaking questions, Phi-4 graph answer vs length-matched
chunks, side randomized, judge scores kept in a separate key file.

Severity: 100 triples from the labeled graph, stratified by the lexicon, with the
prior stripped from the sheet annotators see.

  python export_annotator_sheets.py answers
  python export_annotator_sheets.py severity
  python export_annotator_sheets.py both
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(HERE, "human_validation", "ratings")
PHI4 = os.path.join(HERE, "evaluation_results", "ablations", "phi4")
LABELED_KG = os.path.normpath(os.path.join(
    HERE, "../kg/output/kg_llm/gs100_total500/gemma3_4b_it/kg_gemma3_4b_it.gml",
))

INSTRUCTIONS_ANSWERS = (
    "Score each answer 1-5 on overall sensemaking quality for the question "
    "(fill overall_left and overall_right). Left and right are two systems; "
    "do not try to identify which. preference is left, right, or tie. "
    "Two annotators fill annotator_a_* and annotator_b_* independently. "
    "Leave cells blank if unsure; notes are optional."
)

ANSWER_CSV_COLS = [
    "item_id",
    "question",
    "answer_left",
    "answer_right",
    "annotator_a_overall_left",
    "annotator_a_overall_right",
    "annotator_a_preference",
    "annotator_b_overall_left",
    "annotator_b_overall_right",
    "annotator_b_preference",
    "notes",
]
INSTRUCTIONS_SEVERITY = (
    "Assign one class to each fact, for the worst plausible outcome of that fact alone: "
    "catastrophic, hazardous, major, minor, or none. "
    "Two annotators fill annotator_a_class and annotator_b_class independently. "
    "This sheet does not include the lexicon prior."
)


def _mean_metrics(item: dict) -> Optional[float]:
    metrics = item.get("global_metrics") or {}
    vals = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _eval_scores(arm_dir: str, block: str) -> Dict[str, float]:
    matches = sorted(
        p for p in glob.glob(os.path.join(arm_dir, "global_sensemaking_evaluation_*.json"))
        if not p.endswith("_judge_meta.json")
    )
    if not matches:
        return {}
    blob = json.load(open(matches[-1]))
    section = blob.get(block) or {}
    out = {}
    for item in section.get("individual_evaluations") or []:
        score = _mean_metrics(item)
        if score is not None:
            out[item["question_id"]] = score
    return out


def _answers_by_id(path: str, field: str) -> Dict[str, dict]:
    rows = json.load(open(path))
    return {row["question_id"]: row for row in rows if row.get(field)}


def build_answer_sheet(n: int = 40, seed: int = 17) -> tuple:
    keo_path = os.path.join(PHI4, "mst_on_freq", "answers.json")
    chunk_path = os.path.join(PHI4, "ctrl_chunk_tok", "answers.json")
    keo = _answers_by_id(keo_path, "keo_answer")
    chunk = _answers_by_id(chunk_path, "chunk_tok_answer")
    haiku_keo = _eval_scores(os.path.join(PHI4, "mst_on_freq"), "keo_global")
    haiku_chunk = _eval_scores(os.path.join(PHI4, "ctrl_chunk_tok"), "chunk_tok_global")
    sonnet_root = os.path.join(HERE, "evaluation_results", "ablations", "phi4_sonnet")
    sonnet_keo = _eval_scores(os.path.join(sonnet_root, "mst_on_freq"), "keo_global")
    sonnet_chunk = _eval_scores(os.path.join(sonnet_root, "ctrl_chunk_tok"), "chunk_tok_global")

    ids = sorted(set(haiku_keo) & set(haiku_chunk) & set(keo) & set(chunk))
    rng = random.Random(seed)
    chosen = ids if len(ids) <= n else rng.sample(ids, n)
    rng.shuffle(chosen)

    blind_items = []
    key_items = []
    for qid in chosen:
        left_is_keo = rng.random() < 0.5
        keo_text = keo[qid]["keo_answer"]
        chunk_text = chunk[qid]["chunk_tok_answer"]
        left, right = (keo_text, chunk_text) if left_is_keo else (chunk_text, keo_text)
        # Plain answer strings only — never dump full answer records (method leaks).
        blind_items.append({
            "item_id": qid,
            "question": keo[qid].get("question", ""),
            "answer_left": left if isinstance(left, str) else str(left.get("answer") or ""),
            "answer_right": right if isinstance(right, str) else str(right.get("answer") or ""),
            "annotator_a_overall_left": None,
            "annotator_a_overall_right": None,
            "annotator_a_preference": None,
            "annotator_b_overall_left": None,
            "annotator_b_overall_right": None,
            "annotator_b_preference": None,
            "notes": None,
        })
        key_items.append({
            "item_id": qid,
            "left_system": "keo" if left_is_keo else "chunk_tok",
            "right_system": "chunk_tok" if left_is_keo else "keo",
            "haiku_keo": haiku_keo.get(qid),
            "haiku_chunk_tok": haiku_chunk.get(qid),
            "sonnet_keo": sonnet_keo.get(qid),
            "sonnet_chunk_tok": sonnet_chunk.get(qid),
        })
    blind = {
        "batch_id": "answer_calibration_n40",
        "role": "annotator",
        "n": len(blind_items),
        "seed": seed,
        "instructions": INSTRUCTIONS_ANSWERS,
        "items": blind_items,
    }
    key = {
        "batch_id": "answer_calibration_n40_key",
        "role": "scoring_key_not_for_annotators",
        "generator": "phi4",
        "note": "Haiku and Sonnet scores are for later agreement. Do not show this file to annotators.",
        "items": key_items,
    }
    return blind, key


def build_severity_sheet(n: int = 100, seed: int = 17, kg_path: str = LABELED_KG) -> tuple:
    import networkx as nx

    from risk_retrieval import _edge_records
    from severity_lexicon import export_for_adjudication

    os.makedirs(RATINGS, exist_ok=True)
    graph = nx.read_gml(kg_path)
    triples = [
        (rec["head"], rec["relation"], rec["tail"])
        for rec in _edge_records(graph)
        if rec["head"].strip() and rec["tail"].strip()
    ]
    tmp = os.path.join(RATINGS, "_severity_tmp.json")
    export_for_adjudication(triples, out_path=tmp, n_sample=n, seed=seed)
    items = json.load(open(tmp))["items"]
    os.remove(tmp)
    blind_items = []
    key_items = []
    for i, item in enumerate(items):
        blind_items.append({
            "i": i,
            "head": item["head"],
            "relation": item["relation"],
            "tail": item["tail"],
            "annotator_a_class": None,
            "annotator_b_class": None,
        })
        key_items.append({
            "i": i,
            "prior_class": item.get("prior_class"),
            "sampling_weight": item.get("sampling_weight"),
        })
    blind = {
        "batch_id": "severity_expert_n100",
        "role": "annotator",
        "n": len(blind_items),
        "seed": seed,
        "instructions": INSTRUCTIONS_SEVERITY,
        "items": blind_items,
    }
    key = {
        "batch_id": "severity_expert_n100_key",
        "role": "scoring_key_not_for_annotators",
        "kg": kg_path,
        "items": key_items,
    }
    return blind, key


def _write(name: str, payload: dict) -> str:
    os.makedirs(RATINGS, exist_ok=True)
    path = os.path.join(RATINGS, name)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _clean_answer_text(text: str) -> str:
    import re

    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", text).strip()


def _write_answer_annotator_pack(blind: dict) -> list:
    """Readable HTML + slim per-annotator rating CSVs (do not put long answers in the score sheet)."""
    import csv
    import html as html_lib
    import re

    os.makedirs(RATINGS, exist_ok=True)
    items = blind.get("items") or []
    written = []

    rubric_path = os.path.join(RATINGS, "answer_calibration_RUBRIC.txt")
    rubric = (
        "Answer calibration — annotator rubric (KEO)\n"
        "=============================================\n\n"
        "Read answers in answer_calibration_n40_READ.html.\n"
        "Enter scores in answer_calibration_n40_annotator_A.csv or _B.csv.\n"
        "Do NOT open answer_calibration_n40_key.json.\n\n"
        "overall_left / overall_right: integer 1–5\n"
        "preference: left | right | tie\n"
        "notes: optional\n\n"
        "Anchors: 5 excellent synthesis; 4 good; 3 adequate; 2 weak; 1 poor.\n"
        "Typical pace: 3–5 min/item (~2–3.5 h for 40).\n"
    )
    with open(rubric_path, "w") as handle:
        handle.write(rubric)
    written.append(rubric_path)

    rate_cols = [
        "item_num", "item_id", "question",
        "overall_left", "overall_right", "preference", "notes",
    ]
    for who in ("A", "B"):
        path = os.path.join(RATINGS, f"answer_calibration_n40_annotator_{who}.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rate_cols)
            writer.writeheader()
            for i, item in enumerate(items, 1):
                writer.writerow({
                    "item_num": i,
                    "item_id": item["item_id"],
                    "question": item.get("question", ""),
                    "overall_left": "",
                    "overall_right": "",
                    "preference": "",
                    "notes": "",
                })
        written.append(path)

    cards = []
    for i, item in enumerate(items, 1):
        q = html_lib.escape(item.get("question") or "")
        left = html_lib.escape(_clean_answer_text(item.get("answer_left") or "")).replace("\n", "<br>\n")
        right = html_lib.escape(_clean_answer_text(item.get("answer_right") or "")).replace("\n", "<br>\n")
        iid = html_lib.escape(item["item_id"])
        cards.append(
            f'<section class="item" id="item-{i}">'
            f'<header><span class="num">Item {i} / {len(items)}</span> <code>{iid}</code></header>'
            f"<h2>{q}</h2>"
            f'<div class="pair">'
            f'<article class="ans"><h3>Left</h3><div class="body">{left}</div></article>'
            f'<article class="ans"><h3>Right</h3><div class="body">{right}</div></article>'
            f"</div>"
            f'<p class="hint">CSV row {i}: overall_left, overall_right (1–5), preference (left/right/tie).</p>'
            f"</section>"
        )
    toc = "".join(f'<a href="#item-{i}">{i}</a>' for i in range(1, len(items) + 1))
    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>KEO answer calibration — read sheet</title>
<style>
body{{margin:0;font:16px/1.45 Georgia,serif;background:#f7f7f5;color:#1a1a1a}}
.banner{{background:#111;color:#f5f5f5;padding:1rem 1.25rem}}
.banner h1{{margin:0 0 .35rem;font:600 1.25rem system-ui,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:1.25rem 1rem 3rem}}
.toc{{display:flex;flex-wrap:wrap;gap:.35rem;margin:1rem 0 1.5rem}}
.toc a{{font:13px system-ui,sans-serif;padding:.25rem .5rem;background:#fff;border:1px solid #ddd;border-radius:4px;color:inherit;text-decoration:none}}
.item{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem;margin:0 0 1.25rem}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}
.ans{{border:1px solid #ddd;border-radius:6px;padding:.75rem;background:#fafafa}}
.hint{{font:13px system-ui,sans-serif;color:#555}}
</style></head><body>
<div class="banner"><h1>KEO answer calibration — reading sheet</h1>
<p>Blind pairwise. Score in your annotator CSV. See RUBRIC.txt. Do not open the key.</p></div>
<div class="wrap"><nav class="toc">{toc}</nav>{''.join(cards)}</div>
</body></html>
"""
    html_path = os.path.join(RATINGS, "answer_calibration_n40_READ.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html_doc)
    written.append(html_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", choices=["answers", "severity", "both"])
    parser.add_argument("--n-answers", type=int, default=40)
    parser.add_argument("--n-triples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.which in ("answers", "both"):
        blind, key = build_answer_sheet(args.n_answers, args.seed)
        # Flatten rating fields for the JSON source of truth
        for item in blind["items"]:
            item["answer_left"] = (
                item["answer_left"]
                if isinstance(item["answer_left"], str)
                else str((item["answer_left"] or {}).get("answer") or "")
            )
            item["answer_right"] = (
                item["answer_right"]
                if isinstance(item["answer_right"], str)
                else str((item["answer_right"] or {}).get("answer") or "")
            )
        print(_write("answer_calibration_n40.json", blind))
        print(_write("answer_calibration_n40_key.json", key))
        for path in _write_answer_annotator_pack(blind):
            print(path)
    if args.which in ("severity", "both"):
        blind, key = build_severity_sheet(args.n_triples, args.seed)
        print(_write("severity_expert_n100.json", blind))
        print(_write("severity_expert_n100_key.json", key))


if __name__ == "__main__":
    main()
