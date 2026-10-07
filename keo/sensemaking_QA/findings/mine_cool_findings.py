#!/usr/bin/env python3
"""
Mine existing KEO evaluation JSONs for cool, paper-ready findings.

Outputs:
  findings/out/cool_findings.json
  findings/out/cool_findings.md
  findings/out/dod_gap_checklist.json

No API calls — pure offline analysis of cached runs + KG geometry.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

ROOT = Path(__file__).resolve().parent
QA_ROOT = ROOT.parent
EVAL_ROOT = QA_ROOT / "evaluation_results"
OUT = ROOT / "out"
QUESTIONS = QA_ROOT / "output" / "aviation_sensemaking_questions.json"
KG_PATH = QA_ROOT.parent / "kg" / "output" / "knowledge_graph.gml"

SEVERITY = {
    "fire": 3.0,
    "flame": 3.0,
    "explosion": 3.0,
    "crash": 3.0,
    "quit": 2.5,
    "starvation": 2.5,
    "uncontained": 2.5,
    "smoke": 2.2,
    "overheat": 2.0,
    "engine": 1.8,
    "fuel": 1.6,
    "hydraulic": 1.5,
    "electrical": 1.4,
    "icing": 1.6,
    "frozen": 1.5,
    "leak": 1.4,
}


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def parse_run_dir(name: str) -> Dict[str, str]:
    m = re.match(r"answer_(.+)_evaluator_(.+)_kg_(.+)$", name)
    if not m:
        return {"raw": name}
    return {"answer_model": m.group(1), "evaluator": m.group(2), "kg": m.group(3)}


def load_global_scores() -> List[Dict[str, Any]]:
    rows = []
    for d in sorted(EVAL_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("answer_"):
            continue
        meta = parse_run_dir(d.name)
        globals_ = list(d.glob("global_sensemaking_evaluation_*.json"))
        if not globals_:
            continue
        data = json.loads(globals_[-1].read_text())
        row: Dict[str, Any] = {**meta, "run_dir": d.name, "file": globals_[-1].name}
        for method_key, label in [
            ("vanilla_global", "vanilla"),
            ("textchunk_global", "textchunk"),
            ("graphrag_global", "keo"),
        ]:
            metrics = (data.get(method_key) or {}).get("global_metrics") or {}
            row[f"{label}_overall"] = metrics.get("overall_avg_score")
            row[f"{label}_n"] = metrics.get("total_evaluated")
        pw = (data.get("pairwise_comparison") or {}).get("TextChunk RAG_vs_GraphRAG") or {}
        tot = pw.get("total_score_comparison") or {}
        row["keo_pairwise_winrate_vs_chunk"] = tot.get("method2_winrate")
        row["chunk_pairwise_winrate_vs_keo"] = tot.get("method1_winrate")
        row["pairwise_ties"] = tot.get("ties")
        if row.get("keo_overall") is not None:
            rows.append(row)
    return rows


def router_oracle_from_paper_tables() -> Dict[str, Any]:
    sense = {
        "gemma-3-it": {"TC": 4.12, "VN": 3.70, "KG": 4.31},
        "phi-4": {"TC": 3.90, "VN": 3.68, "KG": 4.09},
        "mistral-nemo-it": {"TC": 3.84, "VN": 3.39, "KG": 3.87},
    }
    action = {
        "gemma-3-it": {"TC": 3.84, "VN": 3.75, "KG": 3.86},
        "phi-4": {"TC": 4.17, "VN": 4.01, "KG": 3.96},
        "mistral-nemo-it": {"TC": 3.72, "VN": 3.70, "KG": 3.68},
    }
    n_sense, n_action = 83, 50
    out = {}
    for model in sense:
        always_keo = (sense[model]["KG"] * n_sense + action[model]["KG"] * n_action) / (n_sense + n_action)
        always_tc = (sense[model]["TC"] * n_sense + action[model]["TC"] * n_action) / (n_sense + n_action)
        routed = (sense[model]["KG"] * n_sense + action[model]["TC"] * n_action) / (n_sense + n_action)
        out[model] = {
            "always_keo": round(always_keo, 3),
            "always_textchunk": round(always_tc, 3),
            "oracle_routed": round(routed, 3),
            "gain_vs_always_keo": round(routed - always_keo, 3),
            "gain_vs_always_chunk": round(routed - always_tc, 3),
        }
    return out


def rarity_demo(kg_path: Path) -> Dict[str, Any]:
    if not kg_path.exists():
        return {"error": f"missing {kg_path}"}
    g = nx.read_gml(kg_path)
    edges = []
    for u, v, data in g.edges(data=True):
        try:
            w = float(data.get("weight", 1.0))
        except Exception:
            w = 1.0
        text = f"{u} {v}".lower()
        sev = 1.0
        hit = None
        for term, boost in SEVERITY.items():
            if term in text and boost >= sev:
                sev = boost
                hit = term
        rarity = (1.0 / (1.0 + w)) * sev
        edges.append(
            {
                "head": str(u),
                "tail": str(v),
                "freq": w,
                "severity_term": hit,
                "severity": sev,
                "rarity_score": rarity,
            }
        )
    critical = [e for e in edges if e["severity"] >= 2.0]
    safety = [e for e in edges if e["severity"] >= 1.5]
    # Rank by frequency; "buried" = outside top 20% by count
    ranked = sorted(edges, key=lambda e: (-e["freq"], e["head"], e["tail"]))
    top_cut = max(1, int(0.2 * len(ranked)))
    top_set = {(e["head"], e["tail"]) for e in ranked[:top_cut]}
    buried = sum(1 for e in critical if (e["head"], e["tail"]) not in top_set)
    buried_safety = sum(1 for e in safety if (e["head"], e["tail"]) not in top_set)
    return {
        "n_edges": len(edges),
        "n_severity_critical": len(critical),
        "n_severity_safety": len(safety),
        "top20pct_edge_count": top_cut,
        "freq_top5": ranked[:5],
        "critical_top_by_freq": sorted(critical, key=lambda e: -e["freq"])[:8],
        "critical_top_by_rarity": sorted(critical, key=lambda e: -e["rarity_score"])[:8],
        "critical_buried_outside_top20pct_freq": buried,
        "safety_buried_outside_top20pct_freq": buried_safety,
        # aliases used by latex emitter
        "critical_buried_in_bottom_80pct_freq": buried,
        "safety_buried_in_bottom_80pct_freq": buried_safety,
        "takeaway": (
            f"{buried}/{len(critical)} high-severity edges and "
            f"{buried_safety}/{len(safety)} safety-relevant edges fall outside the top 20% by frequency — "
            "frequency-only MST can under-prioritize rare but safety-critical relations."
        ),
    }


def question_mix() -> Dict[str, Any]:
    qs = json.loads(QUESTIONS.read_text()) if QUESTIONS.exists() else []
    return {
        "n": len(qs),
        "by_type": dict(Counter(q.get("type") for q in qs)),
        "by_category": dict(Counter(q.get("category") for q in qs)),
        "implication": (
            "Benchmark is dual-use: ~63% sensemaking-like (analytical/contextual/global) "
            "vs ~37% actionable — a single retrieval policy cannot dominate both."
        ),
    }


def aggregate_model_findings(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    preferred = [r for r in rows if r.get("kg") == "gs100" and "gpt-4o" in (r.get("evaluator") or "")]
    use = preferred or rows
    by_model: Dict[str, List] = defaultdict(list)
    for r in use:
        by_model[r["answer_model"]].append(r)
    summary = {}
    for model, rs in by_model.items():
        keo = [r["keo_overall"] for r in rs if r.get("keo_overall") is not None]
        tc = [r["textchunk_overall"] for r in rs if r.get("textchunk_overall") is not None]
        vn = [r["vanilla_overall"] for r in rs if r.get("vanilla_overall") is not None]
        win = [r["keo_pairwise_winrate_vs_chunk"] for r in rs if r.get("keo_pairwise_winrate_vs_chunk") is not None]
        summary[model] = {
            "n_runs": len(rs),
            "keo_overall_mean": mean(keo),
            "textchunk_overall_mean": mean(tc),
            "vanilla_overall_mean": mean(vn),
            "keo_minus_chunk": ((mean(keo) or 0) - (mean(tc) or 0)) if keo and tc else None,
            "keo_pairwise_winrate_vs_chunk_mean": mean(win),
        }
    return {"prefer_filter": "gs100+gpt-4o_eval" if preferred else "all", "by_model": summary}


def dod_gap() -> Dict[str, Any]:
    return {
        "title": "DoD gap KEO addresses",
        "gaps": [
            {
                "gap": "Data gravity / accreditation",
                "problem": "Sensitive maintenance & military-adjacent text cannot freely hit commercial LLM APIs (FedRAMP/ITAR/FISMA).",
                "keo": "Operational answers on local Gemma/Phi/Mistral; GPT-4o only offline for KG/Q/judge; optional local embeddings.",
            },
            {
                "gap": "Gov API capacity vs mission tempo",
                "problem": "Army Enterprise LLM Workspace / Azure OpenAI Gov help but are limited relative to volume and classification needs.",
                "keo": "On-prem / air-gapped path with structured retrieval so smaller local models punch above weight.",
            },
            {
                "gap": "Wrong retrieval for the wrong job",
                "problem": "One RAG style fails: graphs help corpus sensemaking; chunks help procedural action.",
                "keo": "Dual benchmark + task router (sensemaking→KG, action→chunks, mixed→hybrid).",
            },
            {
                "gap": "Auditability for safety decisions",
                "problem": "Black-box answers are unacceptable for airworthiness / maintenance advice.",
                "keo": "Evidence cards with triple/chunk IDs and retrieval confidence for human verification.",
            },
            {
                "gap": "Rare but critical faults",
                "problem": "Frequency-weighted graphs bury low-count, high-severity relations.",
                "keo": "Rarity/severity edge weights + MST ablations.",
            },
        ],
        "why_others_should_use": [
            "Any org with regulated data (defense, aviation, energy, healthcare) that cannot ship prompts to public APIs.",
            "Teams that need both dashboard-style sensemaking and step-level procedures from the same corpus.",
            "Operators who need citation-like evidence, not just fluent text.",
            "Researchers needing an OMIn-grounded aviation QA split that exposes the graph-vs-chunk tradeoff.",
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_global_scores()
    findings = {
        "generated_at": datetime.now().isoformat(),
        "dod_gap": dod_gap(),
        "question_mix": question_mix(),
        "sensemaking_runs_mined": len(rows),
        "sensemaking_by_model": aggregate_model_findings(rows),
        "oracle_router_from_paper_tables": router_oracle_from_paper_tables(),
        "rarity_demo": rarity_demo(KG_PATH),
        "headline_takeaways": [],
    }

    sm = findings["sensemaking_by_model"]["by_model"]
    for model, s in sm.items():
        if s.get("keo_minus_chunk") is not None and s["keo_minus_chunk"] > 0:
            findings["headline_takeaways"].append(
                f"{model}: KEO beats text-chunk on sensemaking by {s['keo_minus_chunk']:.2f} overall (mean over mined runs)."
            )
    for model, o in findings["oracle_router_from_paper_tables"].items():
        sign = "+" if o["gain_vs_always_keo"] >= 0 else ""
        findings["headline_takeaways"].append(
            f"{model}: oracle task routing scores {o['oracle_routed']} vs always-KEO {o['always_keo']} "
            f"({sign}{o['gain_vs_always_keo']}) and always-chunk {o['always_textchunk']} (+{o['gain_vs_always_chunk']})."
        )
    if findings["rarity_demo"].get("takeaway"):
        findings["headline_takeaways"].append(findings["rarity_demo"]["takeaway"])

    (OUT / "cool_findings.json").write_text(json.dumps(findings, indent=2))
    (OUT / "dod_gap_checklist.json").write_text(json.dumps(findings["dod_gap"], indent=2))

    lines = [
        "# KEO Cool Findings (offline mine)",
        "",
        f"Generated: {findings['generated_at']}",
        "",
        "## DoD gap (why this exists)",
    ]
    for g in findings["dod_gap"]["gaps"]:
        lines += [f"### {g['gap']}", f"- **Problem:** {g['problem']}", f"- **KEO:** {g['keo']}", ""]
    lines += ["## Why others should use this", ""]
    for w in findings["dod_gap"]["why_others_should_use"]:
        lines.append(f"- {w}")
    lines += ["", "## Headline takeaways", ""]
    for t in findings["headline_takeaways"]:
        lines.append(f"- {t}")
    lines += ["", "## Oracle router (from paper tables)", ""]
    lines.append("| Model | Always KEO | Always chunk | Oracle routed | Δ vs KEO |")
    lines.append("|-------|------------|--------------|---------------|----------|")
    for model, o in findings["oracle_router_from_paper_tables"].items():
        sign = "+" if o["gain_vs_always_keo"] >= 0 else ""
        lines.append(
            f"| {model} | {o['always_keo']} | {o['always_textchunk']} | {o['oracle_routed']} | {sign}{o['gain_vs_always_keo']} |"
        )
    lines += ["", "## Rarity demo", findings["rarity_demo"].get("takeaway", ""), ""]
    (OUT / "cool_findings.md").write_text("\n".join(lines))
    print(f"Wrote {OUT / 'cool_findings.json'}")
    print(f"Wrote {OUT / 'cool_findings.md'}")
    print(f"Mined {len(rows)} sensemaking eval runs")
    for t in findings["headline_takeaways"][:8]:
        print(" •", t)


if __name__ == "__main__":
    main()
