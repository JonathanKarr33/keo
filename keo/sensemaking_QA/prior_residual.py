#!/usr/bin/env python3
"""Prior-residual retrieval: score a fact by what the closed book does not already say.

GraphRAG ranks an edge by similarity to the question. On this corpus that
retrieves the hubs a local model would have named with no context at all
("leakage", "wear", "improper installation"). The judged controls already
show the consequence: shuffling edges and stripping relation labels moves
the score by ~0, and length-matched chunks beat the graph. The generator
is answering from its prior. Topology never gets a vote.

The quantity that should have been in the objective is the residual against
that prior. For an edge e, a question q, and a closed-book answer a_∅:

    u(e | q, a_∅) = spec(e, q) · sev(e) · idf(e) · (1 − π(e, a_∅))

    spec  = fraction of the edge's content words that are in the question
    π     = fraction of the edge's content words already in a_∅
    idf   = how uncommon those words are across retrieved neighborhoods
    sev   = graded severity (severity_lexicon.edge_severity)

u is zero for a fact the unconditioned model already asserts, and largest
for a question-relevant, uncommon, safety-relevant fact the prior missed.

Two uses, both CPU-only (no answerer call):

1. Mechanism. On an already-judged sweep, Π(q) — the fraction of retrieved
   seeds already named by a_∅ — is known at retrieval time. If high Π
   predicts a chunk win, the graph lost because retrieval landed inside
   the prior. Citation rate and prior-echo are reported as post-hoc
   explanations; they use the answer, so they are not routing features.

2. Policy. Re-rank the edges a run already retrieved by u, and route the
   question to chunks when Π is high. Leave-one-out thresholding is the
   honest estimate of that router. It does not generate new answers.

The closed-book prior for the Phi-4 cohort is the same weights
(closed_book_phi4_live.json). Using it on Gemma answers is a textbook
proxy from a different model and must be labeled as such.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dispersion_law import spearman
from severity_lexicon import edge_severity

HERE = os.path.dirname(os.path.abspath(__file__))

CRITERIA = (
    "global_perspective",
    "theme_identification",
    "synthesis_quality",
    "strategic_value",
    "pattern_recognition",
)

# Binding is what a graph is for; coverage is what prose already supplies.
BINDING = ("synthesis_quality", "pattern_recognition")
COVERAGE = ("theme_identification", "global_perspective")

_STOP = {
    "that", "this", "with", "from", "have", "were", "been", "their", "there",
    "which", "would", "could", "should", "about", "into", "after", "before",
    "these", "those", "other", "than", "then", "them", "they", "what", "when",
    "where", "while", "your", "most", "more", "some", "such", "only", "also",
    "across", "between", "during", "under", "over", "each", "both", "many",
    "much", "very", "just", "like", "make", "made", "using", "used", "based",
    "related", "including", "within", "without", "through", "because",
}


def content_tokens(text: str) -> set:
    """Lowercased content words. Short tokens and stopwords carry no fact."""
    out = set()
    tok = []
    for ch in (text or "").lower():
        if ch.isalpha() or ch.isdigit():
            tok.append(ch)
        else:
            if tok:
                w = "".join(tok)
                if len(w) >= 4 and w not in _STOP:
                    out.add(w)
                tok = []
    if tok:
        w = "".join(tok)
        if len(w) >= 4 and w not in _STOP:
            out.add(w)
    return out


def jaccard(a: Iterable[str], b: Iterable[str]) -> Optional[float]:
    aa, bb = set(a), set(b)
    if not aa or not bb:
        return None
    return len(aa & bb) / len(aa | bb)


def prior_echo(answer: str, prior: str, question: str) -> Optional[float]:
    """Fraction of the answer's distinctive words that the closed book already used.

    Question words are removed from both sides so shared task vocabulary
    ("leakage", "maintenance") cannot masquerade as a prior hit.
    """
    q = content_tokens(question)
    a = content_tokens(answer) - q
    p = content_tokens(prior) - q
    if not a:
        return None
    return len(a & p) / len(a)


def distinctive(entity: str, question: str) -> set:
    """Entity words that are not already in the question. Empty ⇒ uninformative."""
    return content_tokens(entity) - content_tokens(question)


def _coverage(items: Sequence[str], haystack: set, question: str) -> Optional[float]:
    """Mean fraction of each item's distinctive words that land in haystack.

    Items with no distinctive words are dropped, not scored as zero — a seed
    that only repeats the question cannot tell us whether context was used.
    """
    fracs = []
    for item in items:
        toks = distinctive(item, question)
        if not toks:
            continue
        fracs.append(len(toks & haystack) / len(toks))
    if not fracs:
        return None
    return float(sum(fracs) / len(fracs))


def seed_prior_contamination(seeds: Sequence[str], prior: str, question: str) -> Optional[float]:
    """Π(q): how much of the retrieved neighborhood a_∅ already named.

    Knowable before generation, once a closed-book cache exists.
    """
    return _coverage(seeds, content_tokens(prior), question)


def citation_rate(entities: Sequence[str], answer: str, question: str) -> Optional[float]:
    """Fraction of retrieved entities the answer actually names, past the question."""
    return _coverage(entities, content_tokens(answer), question)


def seed_entropy(sims: Sequence[float]) -> Optional[float]:
    """Normalised entropy of seed similarities. 1 = flat neighborhood, 0 = peaked."""
    mass = np.array([max(0.0, float(s)) for s in sims], dtype=float)
    if mass.size < 2 or float(mass.sum()) <= 0.0:
        return None
    p = mass / mass.sum()
    raw = float(-(p * np.log(p + 1e-12)).sum())
    return raw / math.log(len(p))


def edge_tokens(head: str, relation: str, tail: str) -> set:
    return content_tokens(f"{head} {relation} {tail}")


def residual_utility(
    head: str,
    relation: str,
    tail: str,
    question: str,
    prior: str,
    idf: Dict[str, float],
    severity: Optional[float] = None,
) -> float:
    """u(e | q, a_∅) = spec · sev · idf · (1 − π).

    A fact fully contained in the closed-book answer scores 0. A fact with
    no content words scores 0 rather than a silent 1 — empty edges must not
    outrank real ones.
    """
    toks = edge_tokens(head, relation, tail)
    if not toks:
        return 0.0
    q = content_tokens(question)
    p = content_tokens(prior)
    spec = len(toks & q) / len(toks)
    pi = len(toks & p) / len(toks)
    sev = edge_severity(head, relation, tail) if severity is None else float(severity)
    idf_mean = sum(idf.get(t, 0.0) for t in toks) / len(toks)
    return float(spec * sev * idf_mean * (1.0 - pi))


def neighborhood_idf(per_question_tokens: Sequence[Iterable[str]]) -> Dict[str, float]:
    """IDF over retrieved neighborhoods. A word in every question's graph is a hub."""
    n = len(per_question_tokens)
    if n == 0:
        return {}
    df: Dict[str, int] = {}
    for toks in per_question_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Dict:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        raise ValueError("mean_ci on empty sample")
    rng = np.random.default_rng(seed)
    boots = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
    }


def _best_rule(feature: np.ndarray, keo: np.ndarray, chunk: np.ndarray):
    """Train-set threshold. Returns (direction, tau or None).

    direction 'constant' abstains: the feature never beat the better arm.
    'high_chunk' sends feature >= tau to chunks. 'low_chunk' sends feature <= tau.
    """
    best_score = max(float(keo.mean()), float(chunk.mean()))
    best = ("constant", None, "keo" if float(keo.mean()) >= float(chunk.mean()) else "chunk")
    # Unique values only — duplicate taus cannot change the mask.
    for tau in np.unique(feature):
        for direction, mask in (
            ("high_chunk", feature >= tau),
            ("low_chunk", feature <= tau),
        ):
            # A rule that picks one arm for every train point is the constant policy.
            if mask.all() or (~mask).all():
                continue
            score = float(np.where(mask, chunk, keo).mean())
            if score > best_score + 1e-12:
                best_score = score
                best = (direction, float(tau), None)
    return best


def apply_rule(direction: str, tau, constant_arm: Optional[str], value: float, keo: float, chunk: float) -> float:
    if direction == "constant":
        return keo if constant_arm == "keo" else chunk
    use_chunk = (value >= tau) if direction == "high_chunk" else (value <= tau)
    return chunk if use_chunk else keo


def loo_route(
    feature: Sequence[float],
    keo: Sequence[float],
    chunk: Sequence[float],
) -> Dict:
    """Leave-one-out threshold router. The held-out label is never used to pick τ."""
    f = np.asarray(feature, dtype=float)
    k = np.asarray(keo, dtype=float)
    c = np.asarray(chunk, dtype=float)
    if not (len(f) == len(k) == len(c)) or len(f) < 8:
        raise ValueError("loo_route needs ≥8 aligned rows")
    chosen = np.empty(len(f))
    n_abstain = 0
    n_chunk = 0
    for i in range(len(f)):
        tr = np.ones(len(f), dtype=bool)
        tr[i] = False
        direction, tau, arm = _best_rule(f[tr], k[tr], c[tr])
        if direction == "constant":
            n_abstain += 1
        chosen[i] = apply_rule(direction, tau, arm, float(f[i]), float(k[i]), float(c[i]))
        if chosen[i] == float(c[i]) and chosen[i] != float(k[i]):
            n_chunk += 1
    always_keo = float(k.mean())
    always_chunk = float(c.mean())
    oracle = float(np.maximum(k, c).mean())
    policy = float(chosen.mean())
    best_const = max(always_keo, always_chunk)
    gap = oracle - best_const
    return {
        "n": int(len(f)),
        "policy_mean": policy,
        "always_keo": always_keo,
        "always_chunk": always_chunk,
        "oracle": oracle,
        "gap_closed": (policy - best_const) / gap if gap > 1e-9 else None,
        "folds_abstained": n_abstain,
        "heldout_routed_chunk": n_chunk,
    }


def load_eval_items(path: str, method: str) -> Dict[str, Dict[str, float]]:
    blob = json.load(open(path))
    block = (blob.get("by_method") or {}).get(method)
    if not block:
        raise ValueError(f"{path} has no by_method[{method!r}]")
    out = {}
    for item in block.get("individual_evaluations") or []:
        qid = item.get("question_id")
        metrics = item.get("global_metrics") or {}
        if not qid or not metrics:
            continue
        scores = {}
        for key in CRITERIA:
            if key not in metrics:
                raise ValueError(f"{path} / {qid} missing {key}")
            scores[key] = float(metrics[key])
        scores["overall"] = sum(scores[c] for c in CRITERIA) / len(CRITERIA)
        out[qid] = scores
    if not out:
        raise ValueError(f"{path} / {method}: no per-question scores")
    return out


def load_answers(path: str, field: str) -> Dict[str, Dict]:
    rows = json.load(open(path))
    out = {}
    for row in rows:
        ans = row.get(field) or {}
        qid = ans.get("question_id") or row.get("question_id")
        if qid and ans.get("answer"):
            out[qid] = ans
    if not out:
        raise ValueError(f"{path} field {field}: no answers")
    return out


def load_closed_book(path: str) -> Dict[str, Dict]:
    blob = json.load(open(path))
    out = {}
    for row in blob.get("per_question") or []:
        qid = row.get("question_id")
        if qid and row.get("answer"):
            out[qid] = row
    if not out:
        raise ValueError(f"{path}: no closed-book answers")
    return out


def _entity_names(answer: Dict) -> Tuple[List[str], List[float], List[Dict]]:
    ev = answer.get("evidence") or {}
    seeds = ev.get("seed_nodes") or []
    triples = ev.get("triples") or []
    names = [str(s.get("id") or "") for s in seeds if s.get("id")]
    sims = [float(s.get("sim") or 0.0) for s in seeds if s.get("id")]
    return names, sims, triples


def copy_utility(name: str) -> float:
    """Rank a retrieved name by how copyable it is, not by similarity to the question.

    On the Phi-4 neighborhoods, cosine similarity to the question has ρ ≈ 0
    with whether the answer copies the name. Severity raises the copy rate
    (0.27 → 0.44); extra content tokens lower it. Similarity is deliberately
    absent from u.
    """
    toks = content_tokens(name)
    if not toks:
        return 0.0
    return float(edge_severity(name, "", "") / (1.0 + len(toks)))


def seed_copy_association(answers: Dict[str, Dict], ids: Sequence[str]) -> Dict:
    """Does similarity or severity predict whether a retrieved name is copied?

    Computed on the graph arm's seed list. Similarity is the retriever's rank
    feature; severity is the lexicon prior. They answer different questions,
    and this is where that split shows up in the generated text.
    """
    cited_fracs: List[float] = []
    sims: List[float] = []
    sevs: List[float] = []
    for qid in ids:
        ans = answers[qid]
        question = ans.get("question") or ""
        haystack = content_tokens(ans.get("answer") or "")
        for seed in (ans.get("evidence") or {}).get("seed_nodes") or []:
            name = str(seed.get("id") or "")
            dist = distinctive(name, question)
            if not dist:
                continue
            cited_fracs.append(len(dist & haystack) / len(dist))
            sims.append(float(seed.get("sim") or 0.0))
            sevs.append(edge_severity(name, "", ""))
    if len(cited_fracs) < 8:
        raise ValueError("too few seeds to relate copying to similarity or severity")
    rho_sim, p_sim = spearman(sims, cited_fracs)
    rho_sev, p_sev = spearman(sevs, cited_fracs)
    marked = [c for c, s in zip(cited_fracs, sevs) if s > 1.0]
    unmarked = [c for c, s in zip(cited_fracs, sevs) if s <= 1.0]
    return {
        "n_seeds": len(cited_fracs),
        "rho_sim": rho_sim,
        "p_sim": p_sim,
        "rho_severity": rho_sev,
        "p_severity": p_sev,
        "mean_copy_marked": float(np.mean(marked)) if marked else None,
        "n_marked": len(marked),
        "mean_copy_unmarked": float(np.mean(unmarked)) if unmarked else None,
        "n_unmarked": len(unmarked),
    }


def partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """Spearman of x and y after linear residualizing both rank vectors on z.

    This is the length check: if yield only predicts theme because longer
    answers mention more names, the partial correlation falls apart.
    """
    if not (len(x) == len(y) == len(z)) or len(x) < 8:
        raise ValueError("partial_spearman needs ≥8 aligned rows")

    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(1, len(a) + 1, dtype=float)
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        for i, c in enumerate(counts):
            if c > 1:
                r[inv == i] = r[inv == i].mean()
        return r

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        b = b - b.mean()
        a = a - a.mean()
        denom = float(np.dot(b, b))
        if denom == 0.0:
            return a
        return a - (float(np.dot(b, a)) / denom) * b

    rx, ry, rz = rank(np.asarray(x, float)), rank(np.asarray(y, float)), rank(np.asarray(z, float))
    xr, yr = resid(rx, rz), resid(ry, rz)
    xr, yr = xr - xr.mean(), yr - yr.mean()
    denom = float(np.sqrt((xr * xr).sum() * (yr * yr).sum()))
    if denom == 0.0:
        return 0.0
    return float((xr * yr).sum() / denom)


def yield_checks(
    keo_answers: Dict[str, Dict],
    chunk_answers: Dict[str, Dict],
    keo_scores: Dict[str, Dict[str, float]],
    yield_rows: Sequence[Dict],
) -> Dict:
    """Length and cross-arm checks on an already-computed yield.

    Two ways the theme correlation could be an artifact: longer answers cite
    more names, or any fluent answer would mention those entities. The first
    is a partial correlation on character length. The second scores the
    length-matched chunk answer against the same entity list.
    """
    rows = []
    for rec in yield_rows:
        qid = rec["question_id"]
        if qid not in keo_answers or qid not in chunk_answers or qid not in keo_scores:
            continue
        question = keo_answers[qid].get("question") or ""
        entities = entity_lists(keo_answers[qid])
        c_graph = citation_rate(entities, keo_answers[qid].get("answer") or "", question)
        c_chunk = citation_rate(entities, chunk_answers[qid].get("answer") or "", question)
        if c_graph is None or c_chunk is None:
            continue
        rows.append({
            "Y": float(rec["Y"]),
            "len": float(len(keo_answers[qid].get("answer") or "")),
            "theme": float(keo_scores[qid]["theme_identification"]),
            "overall": float(keo_scores[qid]["overall"]),
            "c_graph": float(c_graph),
            "c_chunk": float(c_chunk),
        })
    if len(rows) < 8:
        raise ValueError(f"yield checks defined on only {len(rows)} questions")
    y = [r["Y"] for r in rows]
    length = [r["len"] for r in rows]
    theme = [r["theme"] for r in rows]
    overall = [r["overall"] for r in rows]
    rho_len, p_len = spearman(y, length)
    partial_theme = partial_spearman(y, theme, length)
    partial_overall = partial_spearman(y, overall, length)

    def perm_p(observed: float, score_key: str) -> float:
        rng = np.random.default_rng(0)
        scores = np.array([r[score_key] for r in rows], float)
        extreme = 0
        n_perm = 5000
        for _ in range(n_perm):
            trial = partial_spearman(y, rng.permutation(scores), length)
            if abs(trial) >= abs(observed):
                extreme += 1
        return (extreme + 1) / (n_perm + 1)

    order = np.argsort(np.array(y))
    n = len(rows)
    tertile_len = {}
    for label, idx in (
        ("low_Y", order[: n // 3]),
        ("mid_Y", order[n // 3 : 2 * n // 3]),
        ("high_Y", order[2 * n // 3 :]),
    ):
        tertile_len[label] = float(np.mean([length[i] for i in idx]))
    citation_gap = paired_signflip([r["c_graph"] - r["c_chunk"] for r in rows])
    return {
        "n": len(rows),
        "rho_Y_length": rho_len,
        "p_Y_length": p_len,
        "partial_rho_theme_given_length": partial_theme,
        "p_partial_theme": perm_p(partial_theme, "theme"),
        "partial_rho_overall_given_length": partial_overall,
        "p_partial_overall": perm_p(partial_overall, "overall"),
        "tertile_mean_chars": tertile_len,
        "citation_graph": float(np.mean([r["c_graph"] for r in rows])),
        "citation_chunk_same_entities": float(np.mean([r["c_chunk"] for r in rows])),
        "citation_gap_graph_minus_chunk": citation_gap,
        "hazard_channel": _hazard_channel(keo_answers, keo_scores, yield_rows),
    }


def _hazard_channel(
    keo_answers: Dict[str, Dict],
    keo_scores: Dict[str, Dict[str, float]],
    yield_rows: Sequence[Dict],
) -> Dict:
    """Does the judge pay extra for lexicon-marked names?

    Citation of marked names and of the other retrieved names are both rates,
    so a longer hazard list cannot win by counting. The contrast is their
    difference against theme. A null contrast means severity changes the
    selector and does not change the rubric channel yield already measured.
    """
    marked_c, unmarked_c, themes = [], [], []
    for rec in yield_rows:
        qid = rec["question_id"]
        if qid not in keo_answers or qid not in keo_scores:
            continue
        ans = keo_answers[qid]
        question = ans.get("question") or ""
        marked, unmarked = [], []
        for name in entity_lists(ans):
            if not distinctive(name, question):
                continue
            if edge_severity(name, "", "") > 1.0:
                marked.append(name)
            else:
                unmarked.append(name)
        cm = citation_rate(marked, ans.get("answer") or "", question) if marked else None
        cu = citation_rate(unmarked, ans.get("answer") or "", question) if unmarked else None
        if cm is None or cu is None:
            continue
        marked_c.append(cm)
        unmarked_c.append(cu)
        themes.append(float(keo_scores[qid]["theme_identification"]))
    if len(themes) < 8:
        raise ValueError(f"hazard channel defined on only {len(themes)} questions")
    rho_m, p_m = spearman(marked_c, themes)
    rho_u, p_u = spearman(unmarked_c, themes)
    diff = [m - u for m, u in zip(marked_c, unmarked_c)]
    rho_d, p_d = spearman(diff, themes)
    return {
        "n": len(themes),
        "mean_citation_marked": float(np.mean(marked_c)),
        "mean_citation_unmarked": float(np.mean(unmarked_c)),
        "rho_marked_theme": rho_m,
        "p_marked_theme": p_m,
        "rho_unmarked_theme": rho_u,
        "p_unmarked_theme": p_u,
        "rho_contrast_theme": rho_d,
        "p_contrast_theme": p_d,
    }


def entity_lists(answer: Dict) -> List[str]:
    names, _, triples = _entity_names(answer)
    out = list(names)
    for t in triples:
        out.append(str(t.get("head") or ""))
        out.append(str(t.get("tail") or ""))
    return [e for e in out if e]


def paired_signflip(diffs: Sequence[float], n_perm: int = 5000, seed: int = 0) -> Dict:
    """Two-sided sign-flip test on a paired difference. Empty input is refused."""
    a = np.asarray(list(diffs), dtype=float)
    if a.size < 8:
        raise ValueError("sign-flip needs ≥8 pairs")
    obs = float(a.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=a.size)
        if abs(float((signs * a).mean())) >= abs(obs):
            extreme += 1
    return {
        "n": int(a.size),
        "mean": obs,
        "p_signflip": (extreme + 1) / (n_perm + 1),
    }


def _derangement(ids: Sequence[str], rng: np.random.Generator) -> Dict[str, str]:
    perm = list(ids)
    for _ in range(30):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm)):
            return dict(zip(ids, perm))
    raise RuntimeError("could not draw a derangement")


def entity_yield(
    answers: Dict[str, Dict],
    scores: Dict[str, Dict[str, float]],
    ids: Sequence[str],
    seed: int = 0,
    n_derangements: int = 1,
) -> Dict:
    """Y = own citation minus citation of deranged neighborhoods.

    Own citation mixes two things: the model copied its context, and the model
    uses concrete aviation nouns that would overlap any slice of this KG.
    The derangement is the second term. With n_derangements > 1, Y is the
    expected surplus over counterfactual neighborhoods, which is the estimand
    a single shuffle is too noisy to be. A positive Y with E_top ≈ 0 means
    the graph is being read as a bag of names.
    """
    if n_derangements < 1:
        raise ValueError("n_derangements must be ≥1")
    lists = {qid: entity_lists(answers[qid]) for qid in ids}
    prepared = []
    for qid in ids:
        ans = answers[qid]
        question = ans.get("question") or ""
        text = ans.get("answer") or ""
        own = citation_rate(lists[qid], text, question)
        if own is None:
            continue
        prepared.append((qid, question, text, own, len(lists[qid])))

    crosses: Dict[str, List[float]] = {qid: [] for qid, *_ in prepared}
    for d in range(n_derangements):
        donor = _derangement(list(ids), np.random.default_rng(seed + d))
        for qid, question, text, _own, n_ent in prepared:
            cf_src = lists[donor[qid]][:n_ent]
            cross = citation_rate(cf_src, text, question)
            if cross is not None:
                crosses[qid].append(cross)

    rows = []
    for qid, _question, _text, own, _n in prepared:
        if not crosses[qid]:
            continue
        cross_mean = float(np.mean(crosses[qid]))
        rec = {
            "question_id": qid,
            "citation_own": own,
            "citation_cross": cross_mean,
            "Y": own - cross_mean,
            "overall": scores[qid]["overall"],
        }
        for c in CRITERIA:
            rec[c] = scores[qid][c]
        rows.append(rec)
    if len(rows) < 8:
        raise ValueError(f"entity yield defined on only {len(rows)} questions")

    lift = paired_signflip([r["Y"] for r in rows])
    correlations = {}
    y = [r["Y"] for r in rows]
    for key in ("overall",) + CRITERIA:
        rho, p = spearman(y, [r[key] for r in rows])
        correlations[key] = {"spearman_rho": rho, "p_perm": p, "n": len(rows)}

    order = sorted(rows, key=lambda r: r["Y"])
    n = len(order)
    tertiles = {}
    for label, sub in (
        ("low_Y", order[: n // 3]),
        ("mid_Y", order[n // 3 : 2 * n // 3]),
        ("high_Y", order[2 * n // 3 :]),
    ):
        tertiles[label] = {
            "n": len(sub),
            "mean_Y": float(np.mean([r["Y"] for r in sub])),
            "mean_overall": float(np.mean([r["overall"] for r in sub])),
            "mean_theme": float(np.mean([r["theme_identification"] for r in sub])),
            "mean_pattern": float(np.mean([r["pattern_recognition"] for r in sub])),
            "mean_synthesis": float(np.mean([r["synthesis_quality"] for r in sub])),
        }
    return {
        "n_derangements": n_derangements,
        "formula": "Y(a,E_q) = C(a,E_q) - E_pi[C(a,E_pi(q))]",
        "reading": (
            "Y is the citation surplus of the retrieved neighborhood over a "
            "deranged one. It is the part of the answer that required this "
            "graph. E_top and E_rel near zero mean the surplus is names, not edges."
        ),
        "lift": lift,
        "mean_citation_own": float(np.mean([r["citation_own"] for r in rows])),
        "mean_citation_cross": float(np.mean([r["citation_cross"] for r in rows])),
        "correlations_Y": correlations,
        "tertiles_Y": tertiles,
        "copy_utility": "u(v) = sev(v) / (1 + content_tokens(v)); similarity does not predict copying",
        "per_question": rows,
    }


def analyse_cohort(
    name: str,
    keo_eval: str,
    chunk_eval: str,
    shuf_eval: str,
    norel_eval: str,
    keo_answers: str,
    chunk_answers: str,
    closed_book: str,
    prior_note: str,
    keo_method: str = "keo",
    chunk_method: str = "chunk_tok",
    shuf_method: str = "keo_shuf",
    norel_method: str = "keo_norel",
    keo_field: str = "keo_answer",
    chunk_field: str = "chunk_tok_answer",
) -> Dict:
    keo = load_eval_items(keo_eval, keo_method)
    chunk = load_eval_items(chunk_eval, chunk_method)
    shuf = load_eval_items(shuf_eval, shuf_method)
    norel = load_eval_items(norel_eval, norel_method)
    keo_ans = load_answers(keo_answers, keo_field)
    chunk_ans = load_answers(chunk_answers, chunk_field)
    prior = load_closed_book(closed_book)

    ids = sorted(set(keo) & set(chunk) & set(shuf) & set(norel) & set(keo_ans) & set(chunk_ans) & set(prior))
    if len(ids) < 8:
        raise ValueError(f"{name}: only {len(ids)} questions aligned across arms and closed book")

    # IDF from the graph arm's retrieved neighborhoods (hubs of this sweep).
    neighborhoods = []
    for qid in ids:
        names, _, triples = _entity_names(keo_ans[qid])
        toks = set()
        for n in names:
            toks |= content_tokens(n)
        for t in triples:
            toks |= edge_tokens(str(t.get("head") or ""), str(t.get("relation") or ""), str(t.get("tail") or ""))
        neighborhoods.append(toks)
    idf = neighborhood_idf(neighborhoods)

    rows = []
    for qid in ids:
        ans = keo_ans[qid]
        question = ans.get("question") or prior[qid].get("question") or ""
        names, sims, triples = _entity_names(ans)
        prior_text = prior[qid]["answer"]
        keo_text = ans.get("answer") or ""
        chunk_text = chunk_ans[qid].get("answer") or ""
        entities = names + [str(t.get("head") or "") for t in triples] + [str(t.get("tail") or "") for t in triples]
        utilities = []
        pis = []
        for t in triples:
            head, rel, tail = str(t.get("head") or ""), str(t.get("relation") or ""), str(t.get("tail") or "")
            toks = edge_tokens(head, rel, tail)
            if toks:
                pis.append(len(toks & content_tokens(prior_text)) / len(toks))
            utilities.append(residual_utility(head, rel, tail, question, prior_text, idf))
        # Residual re-rank keeps the same number of edges; compare prior-mass.
        order = sorted(range(len(utilities)), key=lambda i: -utilities[i])
        keep = max(1, len(order) // 2) if order else 0
        residual_pi = float(np.mean([pis[i] for i in order[:keep]])) if keep and pis else None
        mst_pi = float(np.mean(pis)) if pis else None
        echo_g = prior_echo(keo_text, prior_text, question)
        echo_c = prior_echo(chunk_text, prior_text, question)
        rows.append({
            "question_id": qid,
            "delta": keo[qid]["overall"] - chunk[qid]["overall"],
            "keo": keo[qid]["overall"],
            "chunk": chunk[qid]["overall"],
            "E_top": keo[qid]["overall"] - shuf[qid]["overall"],
            "E_rel": keo[qid]["overall"] - norel[qid]["overall"],
            "E_bag": norel[qid]["overall"] - chunk[qid]["overall"],
            "Pi": seed_prior_contamination(names, prior_text, question),
            "seed_entropy": seed_entropy(sims),
            "citation": citation_rate(entities, keo_text, question),
            "echo_graph": echo_g,
            "echo_chunk": echo_c,
            "echo_gap": (None if echo_g is None or echo_c is None else echo_g - echo_c),
            "mst_prior_mass": mst_pi,
            "residual_prior_mass": residual_pi,
            "criteria": {
                c: {
                    "delta": keo[qid][c] - chunk[qid][c],
                    "E_top": keo[qid][c] - shuf[qid][c],
                    "E_rel": keo[qid][c] - norel[qid][c],
                }
                for c in CRITERIA
            },
        })

    def col(key):
        return [r[key] for r in rows if r[key] is not None]

    def paired(key):
        xs, ys = [], []
        for r in rows:
            if r[key] is None:
                continue
            xs.append(r[key])
            ys.append(r["delta"])
        return xs, ys

    effects = {
        "delta_keo_minus_chunk": mean_ci(col("delta")),
        "E_top_keo_minus_shuf": mean_ci(col("E_top")),
        "E_rel_keo_minus_norel": mean_ci(col("E_rel")),
        "E_bag_norel_minus_chunk": mean_ci(col("E_bag")),
    }
    correlations = {}
    for key in ("Pi", "seed_entropy", "citation", "echo_graph", "echo_gap", "mst_prior_mass"):
        xs, ys = paired(key)
        if len(xs) < 8:
            correlations[key] = {"n": len(xs), "note": "too few defined values"}
            continue
        rho, p = spearman(xs, ys)
        correlations[key] = {
            "n": len(xs),
            "spearman_rho_vs_delta": rho,
            "p_perm": p,
            "reading": (
                "higher → graph advantage" if rho > 0 else "higher → chunk advantage"
            ),
        }

    # Tertiles of Π, the pre-generation feature.
    pi_rows = [r for r in rows if r["Pi"] is not None]
    pi_vals = sorted(r["Pi"] for r in pi_rows)
    lo_cut = pi_vals[len(pi_vals) // 3]
    hi_cut = pi_vals[(2 * len(pi_vals)) // 3]
    tertiles = {}
    for label, pred in (
        ("low_Pi", lambda r: r["Pi"] <= lo_cut),
        ("mid_Pi", lambda r: lo_cut < r["Pi"] < hi_cut),
        ("high_Pi", lambda r: r["Pi"] >= hi_cut),
    ):
        sub = [r["delta"] for r in pi_rows if pred(r)]
        tertiles[label] = mean_ci(sub) if sub else None

    pi_defined = [r for r in rows if r["Pi"] is not None]
    router = loo_route(
        [r["Pi"] for r in pi_defined],
        [r["keo"] for r in pi_defined],
        [r["chunk"] for r in pi_defined],
    ) if len(pi_defined) >= 8 else None

    ent_defined = [r for r in rows if r["seed_entropy"] is not None]
    router_entropy = loo_route(
        [r["seed_entropy"] for r in ent_defined],
        [r["keo"] for r in ent_defined],
        [r["chunk"] for r in ent_defined],
    ) if len(ent_defined) >= 8 else None

    criterion_effects = {}
    for c in CRITERIA:
        criterion_effects[c] = {
            "delta": mean_ci([r["criteria"][c]["delta"] for r in rows]),
            "E_top": mean_ci([r["criteria"][c]["E_top"] for r in rows]),
            "E_rel": mean_ci([r["criteria"][c]["E_rel"] for r in rows]),
        }

    mst_mass = col("mst_prior_mass")
    res_mass = col("residual_prior_mass")
    rerank = None
    if mst_mass and res_mass and len(mst_mass) == len(res_mass):
        gap = [m - r for m, r in zip(mst_mass, res_mass)]
        rerank = {
            "mean_mst_prior_mass": float(np.mean(mst_mass)),
            "mean_residual_prior_mass": float(np.mean(res_mass)),
            "mean_prior_mass_dropped": mean_ci(gap),
            "reading": (
                "Re-ranking the same retrieved edges by residual utility raises "
                "prior-mass. The closed-book residual does not select a less "
                "parametric neighborhood, and Π does not predict the winner."
            ),
        }

    yield_block = entity_yield(keo_ans, keo, ids, n_derangements=40)
    delta_by = {r["question_id"]: r["delta"] for r in rows}
    y_for_delta = [rec["Y"] for rec in yield_block["per_question"] if rec["question_id"] in delta_by]
    d_for_delta = [delta_by[rec["question_id"]] for rec in yield_block["per_question"] if rec["question_id"] in delta_by]
    rho_d, p_d = spearman(y_for_delta, d_for_delta)
    yield_block["correlation_vs_delta"] = {
        "spearman_rho": rho_d,
        "p_perm": p_d,
        "n": len(y_for_delta),
        "reading": "Y versus KEO minus length-matched chunks. A null means name-yield does not pick the winner.",
    }
    yield_block["seed_copy"] = seed_copy_association(keo_ans, ids)
    yield_block["checks"] = yield_checks(keo_ans, chunk_ans, keo, yield_block["per_question"])

    return {
        "cohort": name,
        "n_questions": len(rows),
        "prior_note": prior_note,
        "formula": "u(e|q,a0) = spec(e,q) * sev(e) * idf(e) * (1 - pi(e,a0))",
        "effects": effects,
        "correlations_vs_delta": correlations,
        "tertiles_Pi": tertiles,
        "router_loo_Pi": router,
        "router_loo_seed_entropy": router_entropy,
        "criterion_effects": criterion_effects,
        "rerank": rerank,
        "entity_yield": {k: v for k, v in yield_block.items() if k != "per_question"},
        "entity_yield_per_question": yield_block["per_question"],
        "per_question": rows,
    }


def latex_table(result: Dict) -> str:
    """Compact mechanism table. Means with bootstrap intervals, no invented stars."""
    eff = result["effects"]

    def cell(block):
        return f"{block['mean']:+.3f} [{block['ci95_low']:+.3f}, {block['ci95_high']:+.3f}]"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Prior-residual decomposition ({result['cohort']}, $n={result['n_questions']}$). "
        r"$E_{\mathrm{top}}$ is true edges minus shuffled edges; "
        r"$E_{\mathrm{rel}}$ is labeled edges minus stripped labels; "
        r"$E_{\mathrm{bag}}$ is the entity bag minus length-matched chunks. "
        r"Intervals are question-level bootstrap 95\% CIs.}}",
        r"\label{tab:prior-residual}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Effect & Mean $\Delta$ [95\% CI] \\",
        r"\midrule",
        rf"Graph $-$ chunk & {cell(eff['delta_keo_minus_chunk'])} \\",
        rf"$E_{{\mathrm{{top}}}}$ (KEO $-$ shuffle) & {cell(eff['E_top_keo_minus_shuf'])} \\",
        rf"$E_{{\mathrm{{rel}}}}$ (KEO $-$ no-label) & {cell(eff['E_rel_keo_minus_norel'])} \\",
        rf"$E_{{\mathrm{{bag}}}}$ (no-label $-$ chunk) & {cell(eff['E_bag_norel_minus_chunk'])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _default_cohorts() -> List[Dict]:
    root = os.path.join(HERE, "evaluation_results", "ablations")
    stats = os.path.join(HERE, "evaluation_results", "stats")
    closed = os.path.join(stats, "closed_book_phi4_live.json")

    def ev(cohort, arm):
        folder = os.path.join(root, cohort, arm)
        matches = [
            os.path.join(folder, n)
            for n in os.listdir(folder)
            if n.startswith("global_sensemaking_evaluation_") and n.endswith(".json")
            and not n.endswith("_judge_meta.json")
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"{folder}: expected one eval json, found {matches}")
        return matches[0]

    return [
        {
            "name": "phi4_matched_prior",
            "keo_eval": ev("phi4", "mst_on_freq"),
            "chunk_eval": ev("phi4", "ctrl_chunk_tok"),
            "shuf_eval": ev("phi4", "ctrl_keo_shuf_s13"),
            "norel_eval": ev("phi4", "ctrl_keo_norel"),
            "keo_answers": os.path.join(root, "phi4", "mst_on_freq", "answers.json"),
            "chunk_answers": os.path.join(root, "phi4", "ctrl_chunk_tok", "answers.json"),
            "closed_book": closed,
            "prior_note": (
                "Closed-book prior is mlx-community/phi-4-4bit, the same weights "
                "as the graph and chunk answers. Π is a same-model prior."
            ),
        },
        {
            "name": "gemma4e4b_textbook_proxy",
            "keo_eval": ev("_gemma4e4b", "mst_on_freq"),
            "chunk_eval": ev("_gemma4e4b", "ctrl_chunk_tok"),
            "shuf_eval": ev("_gemma4e4b", "ctrl_keo_shuf_s13"),
            "norel_eval": ev("_gemma4e4b", "ctrl_keo_norel"),
            "keo_answers": os.path.join(root, "_gemma4e4b", "mst_on_freq", "answers.json"),
            "chunk_answers": os.path.join(root, "_gemma4e4b", "ctrl_chunk_tok", "answers.json"),
            "closed_book": closed,
            "prior_note": (
                "Gemma-4 answers scored against the Phi-4 closed book. This is a "
                "cross-model textbook proxy, not Gemma's own prior. Do not pool "
                "with the Phi-4 cohort."
            ),
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Prior-residual mechanism on judged sweeps")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "evaluation_results", "stats"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for spec in _default_cohorts():
        result = analyse_cohort(**spec)
        if spec["name"] == "phi4_matched_prior":
            sonnet_eval = os.path.join(
                HERE, "evaluation_results", "ablations",
                "phi4_sonnet", "mst_on_freq",
            )
            matches = [
                os.path.join(sonnet_eval, n)
                for n in os.listdir(sonnet_eval)
                if n.startswith("global_sensemaking_evaluation_")
                and n.endswith(".json") and "meta" not in n
            ]
            if len(matches) == 1:
                sonnet = load_eval_items(matches[0], "keo")
                paired = [
                    rec for rec in result["entity_yield_per_question"]
                    if rec["question_id"] in sonnet
                ]
                block = {}
                ys = [rec["Y"] for rec in paired]
                for key in ("overall",) + CRITERIA:
                    rho, p = spearman(ys, [sonnet[rec["question_id"]][key] for rec in paired])
                    block[key] = {"spearman_rho": rho, "p_perm": p, "n": len(paired)}
                result["entity_yield"]["sonnet_same_answers"] = block
        out_json = os.path.join(args.out_dir, f"prior_residual_{spec['name']}.json")
        slim = dict(result)
        slim["per_question"] = result["per_question"]
        with open(out_json, "w") as f:
            json.dump(slim, f, indent=2)
        tex_path = os.path.join(args.out_dir, f"prior_residual_{spec['name']}.tex")
        with open(tex_path, "w") as f:
            f.write(latex_table(result))
        eff = result["effects"]
        corr = result["correlations_vs_delta"].get("Pi", {})
        router = result["router_loo_Pi"] or {}
        rerank = result["rerank"] or {}
        yld = result["entity_yield"]
        theme = yld["correlations_Y"]["theme_identification"]
        delta = yld.get("correlation_vs_delta") or {}
        copy = yld.get("seed_copy") or {}
        checks = yld.get("checks") or {}
        gap = checks.get("citation_gap_graph_minus_chunk") or {}
        print(
            f"  E[Y]={yld['lift']['mean']:+.3f} p={yld['lift']['p_signflip']:.4f} "
            f"derangements={yld.get('n_derangements')} "
            f"ρ(Y,theme)={theme['spearman_rho']:+.3f} p={theme['p_perm']:.4f} "
            f"ρ(Y,Δ)={delta.get('spearman_rho')} p={delta.get('p_perm')} "
            f"ρ(Y,len)={checks.get('rho_Y_length')} "
            f"partial theme|len={checks.get('partial_rho_theme_given_length')} "
            f"cite graph-chunk={gap.get('mean')}"
        )
        print(
            f"{spec['name']}: n={result['n_questions']} "
            f"Δ={eff['delta_keo_minus_chunk']['mean']:+.3f} "
            f"E_top={eff['E_top_keo_minus_shuf']['mean']:+.3f} "
            f"E_rel={eff['E_rel_keo_minus_norel']['mean']:+.3f} "
            f"E_bag={eff['E_bag_norel_minus_chunk']['mean']:+.3f} "
            f"ρ(Π,Δ)={corr.get('spearman_rho_vs_delta')} "
            f"LOO={router.get('policy_mean')} "
            f"gap_closed={router.get('gap_closed')} "
            f"prior_mass {rerank.get('mean_mst_prior_mass')}→{rerank.get('mean_residual_prior_mass')}"
        )
        print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
