#!/usr/bin/env python3
"""Pivot A pilot: does a label-free question property predict KG − chunk Δ?

ACL_STRATEGY.md names four dispersion statistics. D_agg is lexical; D_ent is
string-overlap with the KG; D_type is the cue-routed task split. D_sim / D_rec
need a corpus + embedder: use `--embedding-backend hash` while the answerer GPU
is busy, or `local` (MiniLM) when RAM is free. Never invent them when the corpus
flag is absent — the note in the JSON says so explicitly.

Reads an already-judged paired comparison (KEO vs chunk_tok) and reports Spearman
ρ between each property and the per-question Δ. A positive ρ for D_agg means
more aggregation-shaped questions favour the graph arm — the Pivot A prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from local_embeddings import EmbeddingBackend
from task_router import ACTION_CUES, SENSE_CUES, route_question

HERE = os.path.dirname(os.path.abspath(__file__))

AGG_TERMS = re.compile(
    r"\b(across|overall|pattern|patterns|trend|trends|recurring|compare|"
    r"comparison|distribution|most common|system-?level|fleet|throughout|"
    r"among|between|seasonal|aircraft types|to what extent|synthesis)\b",
    re.I,
)

# (record_id, narrative text)
CorpusRecord = Tuple[str, str]


def _overall(metrics: Dict) -> float:
    keys = (
        "global_perspective",
        "theme_identification",
        "synthesis_quality",
        "strategic_value",
        "pattern_recognition",
    )
    vals = [float(metrics[k]) for k in keys if k in metrics]
    if not vals:
        raise ValueError("no criterion scores in evaluation item")
    return sum(vals) / len(vals)


def load_per_question_overall(eval_path: str, method: str) -> Dict[str, float]:
    blob = json.load(open(eval_path))
    block = (blob.get("by_method") or {}).get(method)
    if not block:
        raise ValueError(f"{eval_path} has no by_method[{method!r}]")
    out = {}
    for item in block.get("individual_evaluations") or []:
        qid = item.get("question_id")
        metrics = item.get("global_metrics") or {}
        if qid and metrics:
            out[qid] = _overall(metrics)
    if not out:
        raise ValueError(f"{eval_path} / {method}: no per-question scores")
    return out


def d_agg(question_text: str) -> float:
    """Count of aggregation / cross-record cues, normalised by question length."""
    hits = AGG_TERMS.findall(question_text or "")
    tokens = max(1, len(re.findall(r"\w+", question_text or "")))
    return len(hits) / math.sqrt(tokens)


def d_ent(question_text: str, nodes: List[str]) -> float:
    """How many KG entities appear in the question (case-insensitive substring).

    Proxy for evidence dispersion when record IDs are absent from the GML: a
    question that names many graph entities is more likely to need multi-hop
    context. Empty edges would make a record-count version silently zero.
    """
    text = (question_text or "").lower()
    if not text:
        return 0.0
    hits = 0
    for name in nodes:
        n = name.strip().lower()
        if len(n) < 3:
            continue
        if n in text:
            hits += 1
    return float(hits)


def d_type(question: Dict) -> float:
    """1.0 for sensemaking-routed, 0.0 for action, 0.5 for hybrid."""
    policy = route_question(question)
    return {"keo": 1.0, "textchunk": 0.0, "hybrid": 0.5}[policy]


def load_corpus_records(
    path: str,
    text_col: str = "c119",
    id_col: str = "c5",
    limit: int = 500,
) -> List[CorpusRecord]:
    """Load up to `limit` (id, text) pairs from a CSV — no pandas, low RAM."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if limit < 5:
        raise ValueError("corpus limit must be ≥5")
    rows: List[CorpusRecord] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty CSV")
        fields = set(reader.fieldnames)
        if text_col not in fields:
            # Prefer the longest-looking free-text column among common names.
            for cand in ("c119", "narrative", "text", "description", "report"):
                if cand in fields:
                    text_col = cand
                    break
            else:
                raise ValueError(f"{path}: no text column (tried {text_col})")
        use_id = id_col if id_col in fields else None
        for i, row in enumerate(reader):
            text = (row.get(text_col) or "").strip()
            if len(text) < 20:
                continue
            rid = (row.get(use_id) or "").strip() if use_id else ""
            if not rid:
                rid = f"row_{i}"
            rows.append((rid, text))
            if len(rows) >= limit:
                break
    if len(rows) < 5:
        raise ValueError(f"{path}: only {len(rows)} usable rows after filtering")
    return rows


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def d_sim_d_rec(
    question_text: str,
    records: Sequence[CorpusRecord],
    embedder: EmbeddingBackend,
    top_n: int = 20,
    top_k: int = 10,
    record_vecs: Optional[List[np.ndarray]] = None,
) -> Tuple[float, float]:
    """D_sim = normalised entropy of top-N chunk similarities; D_rec = #distinct records in top-k.

    Flat similarity mass ⇒ high D_sim (dispersed). Many source records in the
    top-k ⇒ high D_rec. Cosines are clipped at 0 before normalisation so a
    single negative-mass spike cannot invent a dispersed distribution.
    """
    if not records:
        raise ValueError("empty corpus")
    top_n = min(top_n, len(records))
    top_k = min(top_k, len(records))
    qv = np.asarray(embedder.embed(question_text or " "), dtype=float)
    if record_vecs is None:
        record_vecs = [np.asarray(embedder.embed(t), dtype=float) for _, t in records]
    scored = [
        (records[i][0], _cosine(qv, record_vecs[i]))
        for i in range(len(records))
    ]
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]
    mass = np.array([max(0.0, s) for _, s in top], dtype=float)
    if float(mass.sum()) <= 0.0:
        d_sim = 0.0
    else:
        p = mass / mass.sum()
        # Normalise by log(N) so a uniform top-N sits at 1.0 regardless of N.
        raw = float(-(p * np.log(p + 1e-12)).sum())
        d_sim = raw / math.log(len(p)) if len(p) > 1 else 0.0
    d_rec = float(len({rid for rid, _ in scored[:top_k]}))
    return d_sim, d_rec


def spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Spearman ρ and a two-sided permutation p-value (label-free, no SciPy)."""
    if len(x) != len(y) or len(x) < 5:
        raise ValueError("need ≥5 paired observations")
    xa, ya = np.asarray(x, float), np.asarray(y, float)

    def rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        # average ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        for i, c in enumerate(counts):
            if c > 1:
                ranks[inv == i] = ranks[inv == i].mean()
        return ranks

    rx, ry = rank(xa), rank(ya)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return 0.0, 1.0
    rho = float((rx * ry).sum() / denom)
    rng = np.random.default_rng(0)
    extreme = 0
    n_perm = 5000
    for _ in range(n_perm):
        yp = rng.permutation(ya)
        ryp = rank(yp) - rank(yp).mean()
        d = np.sqrt((rx * rx).sum() * (ryp * ryp).sum())
        if d == 0:
            continue
        if abs((rx * ryp).sum() / d) >= abs(rho):
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return rho, p


def analyse(
    questions: List[Dict],
    keo_scores: Dict[str, float],
    chunk_scores: Dict[str, float],
    kg_path: Optional[str],
    corpus: Optional[Sequence[CorpusRecord]] = None,
    embedder: Optional[EmbeddingBackend] = None,
    top_n: int = 20,
    top_k: int = 10,
) -> Dict:
    nodes: List[str] = []
    if kg_path and os.path.isfile(kg_path):
        g = nx.read_gml(kg_path)
        nodes = list(g.nodes())

    record_vecs: Optional[List[np.ndarray]] = None
    if corpus is not None:
        if embedder is None:
            raise ValueError("embedder required when corpus is provided")
        # Precompute once — otherwise every question re-embeds the corpus.
        record_vecs = [np.asarray(embedder.embed(t), dtype=float) for _, t in corpus]

    rows = []
    for q in questions:
        qid = q.get("id")
        if qid not in keo_scores or qid not in chunk_scores:
            continue
        text = q.get("question") or ""
        delta = keo_scores[qid] - chunk_scores[qid]  # >0 ⇒ KEO wins
        row = {
            "question_id": qid,
            "delta_keo_minus_chunk": delta,
            "keo": keo_scores[qid],
            "chunk": chunk_scores[qid],
            "D_agg": d_agg(text),
            "D_ent": d_ent(text, nodes),
            "D_type": d_type(q),
            "requires_synthesis": 1.0 if q.get("requires_synthesis") else 0.0,
            "route": route_question(q),
            "type": q.get("type"),
            "category": q.get("category"),
        }
        if corpus is not None and embedder is not None:
            sim, rec = d_sim_d_rec(
                text, corpus, embedder, top_n=top_n, top_k=top_k, record_vecs=record_vecs
            )
            row["D_sim"] = sim
            row["D_rec"] = rec
        rows.append(row)

    if len(rows) < 5:
        raise ValueError(f"only {len(rows)} overlapping scored questions")

    deltas = [r["delta_keo_minus_chunk"] for r in rows]
    properties = ["D_agg", "D_ent", "D_type", "requires_synthesis"]
    if corpus is not None:
        properties.extend(["D_sim", "D_rec"])
    correlations = {}
    for prop in properties:
        rho, p = spearman([r[prop] for r in rows], deltas)
        correlations[prop] = {
            "spearman_rho": rho,
            "p_perm": p,
            "n": len(rows),
            "reading": (
                "higher property → KEO advantage"
                if rho > 0
                else "higher property → chunk advantage"
                if rho < 0
                else "no rank association"
            ),
        }

    # Tertile of D_agg: does the mean Δ flip sign?
    agg_vals = sorted(r["D_agg"] for r in rows)
    lo_cut = agg_vals[len(agg_vals) // 3]
    hi_cut = agg_vals[(2 * len(agg_vals)) // 3]
    tertiles = {}
    for label, pred in (
        ("low_D_agg", lambda r: r["D_agg"] <= lo_cut),
        ("mid_D_agg", lambda r: lo_cut < r["D_agg"] < hi_cut),
        ("high_D_agg", lambda r: r["D_agg"] >= hi_cut),
    ):
        sub = [r["delta_keo_minus_chunk"] for r in rows if pred(r)]
        tertiles[label] = {
            "n": len(sub),
            "mean_delta": float(np.mean(sub)) if sub else None,
        }

    note = (
        f"D_sim/D_rec from {len(corpus)} corpus records via {embedder.backend} "
        f"({embedder.model}), top_n={top_n}, top_k={top_k}."
        if corpus is not None and embedder is not None
        else (
            "D_sim and D_rec not computed: pass --corpus (and optionally "
            "--embedding-backend hash|local). Do not invent them."
        )
    )

    return {
        "n_questions": len(rows),
        "mean_delta_keo_minus_chunk": float(np.mean(deltas)),
        "kg_nodes": len(nodes),
        "kg_path": kg_path,
        "corpus_n": len(corpus) if corpus is not None else 0,
        "embedding_backend": embedder.backend if embedder is not None else None,
        "correlations": correlations,
        "tertiles_D_agg": tertiles,
        "note": note,
        "per_question": rows,
    }


def latex_table(result: Dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Property & Spearman $\rho$ with $\Delta$ & $p_{\mathrm{perm}}$ & Reading \\",
        r"\midrule",
    ]
    for prop, block in result["correlations"].items():
        prop_tex = prop.replace("_", r"\_")
        lines.append(
            f"{prop_tex} & ${block['spearman_rho']:+.3f}$ & "
            f"{block['p_perm']:.3f} & {block['reading']} \\\\"
        )
    if result.get("corpus_n"):
        caption = (
            r"\caption{Pivot~A pilot on $n="
            + str(result["n_questions"])
            + r"$ Claude-Haiku questions. "
            r"$\Delta = \mathrm{score}(\mathrm{KEO}) - \mathrm{score}(\texttt{chunk\_tok})$. "
            r"$D_{\mathrm{agg}}$ lexical; $D_{\mathrm{ent}}$ KG overlap; $D_{\mathrm{type}}$ cue router; "
            r"$D_{\mathrm{sim}}$/$D_{\mathrm{rec}}$ from the retrieval corpus "
            f"($n={result['corpus_n']}$, backend={result.get('embedding_backend')}).}}"
        )
    else:
        caption = (
            r"\caption{Pivot~A pilot on $n="
            + str(result["n_questions"])
            + r"$ Claude-Haiku questions. "
            r"$\Delta = \mathrm{score}(\mathrm{KEO}) - \mathrm{score}(\texttt{chunk\_tok})$. "
            r"$D_{\mathrm{agg}}$ lexical; $D_{\mathrm{ent}}$ KG overlap; $D_{\mathrm{type}}$ cue router. "
            r"$D_{\mathrm{sim}}$/$D_{\mathrm{rec}}$ omitted (no corpus passed).}}"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:dispersion-pilot}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--keo-eval", required=True, help="Judged KEO evaluation JSON")
    parser.add_argument("--chunk-eval", required=True, help="Judged chunk_tok evaluation JSON")
    parser.add_argument("--keo-method", default="keo")
    parser.add_argument("--chunk-method", default="chunk_tok")
    parser.add_argument("--kg", default=None)
    parser.add_argument(
        "--corpus",
        default=None,
        help="CSV of source records for D_sim/D_rec (omit to skip those properties)",
    )
    parser.add_argument("--text-col", default="c119")
    parser.add_argument("--id-col", default="c5")
    parser.add_argument("--corpus-limit", type=int, default=500)
    parser.add_argument(
        "--embedding-backend",
        default="hash",
        choices=("hash", "local"),
        help="hash = CPU-only (safe while the answerer holds the GPU); local = MiniLM",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--latex", default=None)
    args = parser.parse_args()

    questions = json.load(open(args.questions))
    keo = load_per_question_overall(args.keo_eval, args.keo_method)
    chunk = load_per_question_overall(args.chunk_eval, args.chunk_method)

    corpus = None
    embedder = None
    if args.corpus:
        corpus = load_corpus_records(
            args.corpus, text_col=args.text_col, id_col=args.id_col, limit=args.corpus_limit
        )
        embedder = EmbeddingBackend(backend=args.embedding_backend)

    result = analyse(
        questions,
        keo,
        chunk,
        args.kg,
        corpus=corpus,
        embedder=embedder,
        top_n=args.top_n,
        top_k=args.top_k,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"n={result['n_questions']}  mean Δ(KEO−chunk)={result['mean_delta_keo_minus_chunk']:+.3f}")
    for prop, block in result["correlations"].items():
        print(f"  {prop:20} ρ={block['spearman_rho']:+.3f}  p={block['p_perm']:.4f}  {block['reading']}")
    print("  tertiles D_agg:", result["tertiles_D_agg"])
    print(f"  note: {result['note']}")
    print(f"✓ {args.out}")
    if args.latex:
        open(args.latex, "w").write(latex_table(result))
        print(f"✓ {args.latex}")


if __name__ == "__main__":
    main()
