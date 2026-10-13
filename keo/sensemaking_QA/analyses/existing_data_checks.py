"""Checks that use answer files, judge scores, human votes, and graphs already on disk.

Writes a JSON summary next to this script. Does not call a model.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ABL = ROOT / "keo" / "sensemaking_QA" / "evaluation_results" / "ablations"
RAT = ROOT / "keo" / "sensemaking_QA" / "human_validation" / "ratings"
KG = ROOT / "keo" / "kg" / "output"
OUT = Path(__file__).resolve().parent / "existing_data_checks.json"

AXES = [
    "global_perspective",
    "theme_identification",
    "synthesis_quality",
    "strategic_value",
    "pattern_recognition",
]
TIE = 0.1
ID_RE = re.compile(r"\b(?:[A-Z]{2,}|\d[\w./-]*|[A-Za-z]+(?:-[A-Za-z0-9]+)+)\b")


def overall(item: dict) -> float:
    metrics = item["global_metrics"]
    return sum(float(metrics[axis]) for axis in AXES) / len(AXES)


def load_scores(path: Path, method: str) -> dict[str, float]:
    payload = json.loads(path.read_text())
    rows = payload["by_method"][method]["individual_evaluations"]
    return {row["question_id"]: overall(row) for row in rows}


def preference(graph: float, chunk: float, tie: float = TIE) -> str:
    gap = graph - chunk
    if abs(gap) < tie:
        return "tie"
    return "keo" if gap > 0 else "chunk_tok"


def cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    matrix = {(a, b): 0 for a in labels for b in labels}
    for left, right in pairs:
        matrix[(left, right)] += 1
    agree = sum(matrix[(label, label)] for label in labels) / n
    row = {label: sum(matrix[(label, other)] for other in labels) / n for label in labels}
    col = {label: sum(matrix[(other, label)] for other in labels) / n for label in labels}
    chance = sum(row[label] * col[label] for label in labels)
    if math.isclose(chance, 1.0):
        return 1.0
    return (agree - chance) / (1.0 - chance)


def binom_two_sided(wins: int, trials: int) -> float:
    """Two-sided exact test of a fair coin, summing tails at least as extreme."""
    if trials == 0:
        return float("nan")

    def pmf(k: int) -> float:
        return math.comb(trials, k) / (2**trials)

    observed = pmf(wins)
    return sum(pmf(k) for k in range(trials + 1) if pmf(k) <= observed + 1e-15)


def human_judge() -> dict:
    consensus = json.loads((RAT / "human_pref_consensus.json").read_text())
    items = consensus["items"]
    haiku_keo = load_scores(
        ABL / "phi4/mst_on_freq/global_sensemaking_evaluation_20260925_091225.json",
        "keo",
    )
    haiku_chunk = load_scores(
        ABL / "phi4/ctrl_chunk_tok/global_sensemaking_evaluation_20260925_102909.json",
        "chunk_tok",
    )
    sonnet_keo = load_scores(
        ABL / "phi4_sonnet/mst_on_freq/global_sensemaking_evaluation_20260926_185115.json",
        "keo",
    )
    sonnet_chunk = load_scores(
        ABL / "phi4_sonnet/ctrl_chunk_tok/global_sensemaking_evaluation_20260926_190439.json",
        "chunk_tok",
    )
    labels = ["keo", "chunk_tok", "tie"]
    out = {"n": len(items), "tie_band": TIE, "judges": {}}
    for name, graph, chunk in (
        ("haiku", haiku_keo, haiku_chunk),
        ("sonnet", sonnet_keo, sonnet_chunk),
    ):
        pairs = []
        missing = 0
        for item in items:
            qid = item["item_id"]
            if qid not in graph or qid not in chunk:
                missing += 1
                continue
            pairs.append((preference(graph[qid], chunk[qid]), item["final_system"]))
        agree = sum(left == right for left, right in pairs)
        out["judges"][name] = {
            "n": len(pairs),
            "missing": missing,
            "exact_agree": agree,
            "exact_rate": agree / len(pairs) if pairs else None,
            "kappa": cohen_kappa(pairs, labels),
            "pred_counts": dict(Counter(left for left, _ in pairs)),
        }
    # Non-tie human labels only: does the judge pick the same side?
    for name, graph, chunk in (
        ("haiku", haiku_keo, haiku_chunk),
        ("sonnet", sonnet_keo, sonnet_chunk),
    ):
        sided = []
        for item in items:
            final = item["final_system"]
            if final == "tie":
                continue
            qid = item["item_id"]
            sided.append(preference(graph[qid], chunk[qid], tie=0.0) == final)
        out["judges"][name]["nontie_human"] = len(sided)
        out["judges"][name]["nontie_match"] = sum(sided)
    votes = {}
    for who, filename in (("ann1", "human_pref_votes_annotator1.json"), ("ann2", "human_pref_votes_annotator2.json")):
        payload = json.loads((RAT / filename).read_text())
        systems = [vote["preference_system"] for vote in payload["votes"].values()]
        ab = [vote["preference_ab"] for vote in payload["votes"].values()]
        counts = Counter(systems)
        sided = counts["keo"] + counts["chunk_tok"]
        prefer_a = sum(1 for choice in ab if choice == "A")
        decided_ab = sum(1 for choice in ab if choice in {"A", "B"})
        votes[who] = {
            "counts": dict(counts),
            "sign_test_p": binom_two_sided(counts["keo"], sided),
            "prefer_a": prefer_a,
            "decided_ab": decided_ab,
            "prefer_a_rate": prefer_a / decided_ab if decided_ab else None,
        }
    finals = Counter(item["final_system"] for item in items)
    sided_final = finals["keo"] + finals["chunk_tok"]
    out["humans"] = votes
    out["final_counts"] = dict(finals)
    out["final_sign_test_p"] = binom_two_sided(finals["keo"], sided_final)
    return out


def _answer_rows(path: Path, field: str) -> list[dict]:
    rows = json.loads(path.read_text())
    kept = []
    for row in rows:
        payload = row.get(field)
        if isinstance(payload, dict) and payload.get("answer"):
            kept.append(payload)
    return kept


def faithfulness() -> dict:
    stacks = {
        "phi4": ABL / "phi4/mst_on_freq/answers.json",
        "gemma4e4b": ABL / "_gemma4e4b/mst_on_freq/answers.json",
        "gemma4_31b": ABL / "openrouter_gemma4_31b/mst_on_freq/answers.json",
    }
    # Fall back if the 31B path differs.
    if not stacks["gemma4_31b"].exists():
        matches = list(ABL.glob("openrouter_gemma4_31b/**/answers.json"))
        stacks["gemma4_31b"] = matches[0] if matches else stacks["gemma4_31b"]
    report = {}
    for name, path in stacks.items():
        if not path.exists():
            report[name] = {"missing": str(path)}
            continue
        raw = json.loads(path.read_text())
        rows = []
        for item in raw:
            if item.get("question_category") == "action_specific":
                continue
            payload = item.get("keo_answer")
            if isinstance(payload, dict) and payload.get("answer"):
                rows.append(payload)
        covered = []
        copied = []
        for row in rows:
            answer = row.get("answer") or ""
            triples = (row.get("evidence") or {}).get("triples") or []
            context = " ".join(
                f"{triple.get('head','')} {triple.get('relation','')} {triple.get('tail','')}"
                for triple in triples
            ).casefold()
            ids = ID_RE.findall(answer)
            if ids:
                covered.append(sum(token.casefold() in context for token in ids) / len(ids))
            entities = []
            for triple in triples:
                for side in ("head", "tail"):
                    text = str(triple.get(side) or "").strip()
                    if len(text) >= 4:
                        entities.append(text)
            unique = list(dict.fromkeys(entities))
            if unique:
                copied.append(sum(text.casefold() in answer.casefold() for text in unique) / len(unique))
        report[name] = {
            "n_answers": len(rows),
            "mean_identifier_in_triples": sum(covered) / len(covered) if covered else None,
            "n_with_identifiers": len(covered),
            "mean_triple_entity_copied": sum(copied) / len(copied) if copied else None,
        }
    chunk = ABL / "phi4/ctrl_chunk_tok/answers.json"
    sample = _answer_rows(chunk, "chunk_tok_answer")
    report["chunk_tok_phi4_context_is_text"] = bool(sample) and isinstance(sample[0].get("context_used"), str)
    report["chunk_tok_context_field_type"] = type(sample[0].get("context_used")).__name__ if sample else None
    return report


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _close(left: str, right: str) -> bool:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.5


def _gold_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines()[1:]:
        # The triple cell is the last CSV field and may itself contain commas.
        cell = line.split(",", 4)[-1].strip().strip('"')
        if not cell or cell.lower() in {"nan", "none"}:
            continue
        for piece in cell.split(";"):
            parts = [part.strip() for part in piece.split(",")]
            if len(parts) >= 3 and parts[0] and parts[2]:
                pairs.append((parts[0], parts[2]))
    return pairs


def _gml_labels(path: Path) -> tuple[dict[int, str], list[tuple[str, str]]]:
    labels: dict[int, str] = {}
    edges = []
    node_id = None
    source = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("id "):
            node_id = int(stripped.split()[1])
        elif stripped.startswith('label "'):
            labels[node_id] = stripped.split('"', 2)[1]
        elif stripped.startswith("source "):
            source = int(stripped.split()[1])
        elif stripped.startswith("target "):
            target = int(stripped.split()[1])
            edges.append((labels[source], labels[target]))
    return labels, edges


def re_overlap() -> dict:
    gml = KG / "kg_llm/gpt4o_with_nodes_batches/100/gpt4o_withprevnodes_100_with_entity_mentions_fixed.gml"
    _, edges = _gml_labels(gml)
    report = {"graph_edges": len(edges)}
    for name, filename in (("strict", "re_gs_strict.csv"), ("loose", "re_gs_loose.csv")):
        gold = _gold_pairs(KG / filename)
        matched_gold = 0
        for head, tail in gold:
            if any(
                (_close(head, src) and _close(tail, dst)) or (_close(head, dst) and _close(tail, src))
                for src, dst in edges
            ):
                matched_gold += 1
        matched_graph = 0
        for src, dst in edges:
            if any(
                (_close(head, src) and _close(tail, dst)) or (_close(head, dst) and _close(tail, src))
                for head, tail in gold
            ):
                matched_graph += 1
        report[name] = {
            "gold_pairs": len(gold),
            "gold_recalled": matched_gold,
            "recall": matched_gold / len(gold) if gold else None,
            "graph_supported": matched_graph,
            "precision": matched_graph / len(edges) if edges else None,
        }
    return report


def _triple_ids(path: Path) -> dict[str, set[str]]:
    rows = json.loads(path.read_text())
    found = {}
    for row in rows:
        payload = row.get("keo_answer")
        if not isinstance(payload, dict):
            continue
        triples = (payload.get("evidence") or {}).get("triples") or []
        found[row["question_id"]] = {str(triple.get("id")) for triple in triples if triple.get("id")}
    return found


def edge_weight_jaccard() -> dict:
    base = _triple_ids(ABL / "mst_on_freq/answers.json")
    report = {"n_freq": len(base)}
    for name in ("pmi", "tfidf"):
        other = _triple_ids(ABL / f"mst_on_{name}/answers.json")
        shared = sorted(set(base) & set(other))
        scores = []
        for qid in shared:
            left, right = base[qid], other[qid]
            union = left | right
            scores.append(len(left & right) / len(union) if union else 1.0)
        report[name] = {
            "n": len(scores),
            "mean_jaccard": sum(scores) / len(scores) if scores else None,
            "median_jaccard": sorted(scores)[len(scores) // 2] if scores else None,
        }
    return report


def benchmark_stats() -> dict:
    rows = json.loads((ABL / "phi4/mst_on_freq/answers.json").read_text())
    questions = [row["question"] for row in rows if row.get("question")]
    categories = Counter(row.get("question_category") or "unknown" for row in rows)
    tokens = [re.findall(r"[A-Za-z0-9'-]+", question) for question in questions]
    lengths = [len(toks) for toks in tokens]
    unigrams = [tok.casefold() for toks in tokens for tok in toks]
    bigrams = [f"{a.casefold()} {b.casefold()}" for toks in tokens for a, b in zip(toks, toks[1:])]

    def distinct(items: list[str]) -> float:
        return len(set(items)) / len(items) if items else float("nan")

    report = {
        "n": len(questions),
        "categories": dict(categories),
        "mean_tokens": sum(lengths) / len(lengths),
        "median_tokens": sorted(lengths)[len(lengths) // 2],
        "distinct_1": distinct(unigrams),
        "distinct_2": distinct(bigrams),
    }
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(questions, normalize_embeddings=True)
        sim = vectors @ vectors.T
        mask = ~np.eye(len(questions), dtype=bool)
        report["mean_pairwise_minilm"] = float(sim[mask].mean())
    except Exception as exc:  # pragma: no cover - environment dependent
        report["minilm_error"] = type(exc).__name__
    return report


def main() -> None:
    summary = {
        "human_judge": human_judge(),
        "faithfulness": faithfulness(),
        "re_overlap": re_overlap(),
        "edge_weight_jaccard": edge_weight_jaccard(),
        "benchmark": benchmark_stats(),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
