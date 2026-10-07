#!/usr/bin/env python3
"""Offline checks for the dispersion-law pilot and contamination probe."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from closed_book_probe import answer_corpus_overlap, closed_book_prompt  # noqa: E402
from contamination_probe import ngrams, probe, tokens  # noqa: E402
from dispersion_law import (  # noqa: E402
    analyse,
    d_agg,
    d_ent,
    d_sim_d_rec,
    load_corpus_records,
    spearman,
)
from local_embeddings import EmbeddingBackend  # noqa: E402
from offline_harness import run_suite  # noqa: E402

def test_d_agg_ranks_aggregation_questions_higher():
    plain = d_agg("What is the torque on the bolt?")
    rich = d_agg("What recurring patterns across aircraft types emerge overall?")
    assert rich > plain, (rich, plain)
    print("  aggregation cues raise D_agg")


def test_d_ent_counts_kg_mentions():
    nodes = ["fuel pump", "landing gear", "ACFT"]
    assert d_ent("fuel pump failure on landing gear", nodes) == 2.0
    assert d_ent("weather delay", nodes) == 0.0
    print("  D_ent counts entity substrings")


def test_spearman_perfect_and_null():
    rho, p = spearman([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])
    assert abs(rho - 1.0) < 1e-9 and p < 0.01
    rho0, p0 = spearman([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1])
    assert abs(rho0 + 1.0) < 1e-9 and p0 < 0.01
    print("  spearman recovers ±1")


def test_analyse_refuses_tiny_overlap():
    try:
        analyse(
            [{"id": "a", "question": "x", "type": "global"}],
            {"a": 1.0},
            {},
            None,
        )
        raise AssertionError("should have refused")
    except ValueError as e:
        assert "overlapping" in str(e) or "only" in str(e)
    print("  tiny overlap is refused, not zero-filled")


def test_analyse_reports_correlation_block():
    qs = [
        {"id": f"q{i}", "question": ("across patterns overall " * (i + 1)), "type": "global"}
        for i in range(12)
    ]
    # Construct Δ that rises with D_agg so ρ should be positive.
    keo = {f"q{i}": 2.0 + 0.1 * i for i in range(12)}
    chunk = {f"q{i}": 2.0 for i in range(12)}
    result = analyse(qs, keo, chunk, None)
    assert result["n_questions"] == 12
    assert "D_agg" in result["correlations"]
    assert result["correlations"]["D_agg"]["spearman_rho"] > 0.5
    assert "D_sim" not in result["correlations"]
    assert "not computed" in result["note"] or "Do not invent" in result["note"]
    print("  analyse recovers a planted D_agg signal")


def test_d_sim_flat_vs_peaked():
    emb = EmbeddingBackend(backend="hash", dim=64)
    # One record that matches the query, many that don't → peaked → low entropy.
    peaked = [
        ("hit", "fuel pump vibration during climbout"),
        ("a", "weather balloon sighting near airport fence"),
        ("b", "catering truck delayed passenger boarding gate"),
        ("c", "runway lights scheduled maintenance overnight"),
        ("d", "baggage carousel jam at terminal south"),
    ]
    flat = [
        (f"r{i}", f"fuel pump vibration climbout event number {i} aircraft system")
        for i in range(8)
    ]
    q = "fuel pump vibration during climbout"
    sim_peak, rec_peak = d_sim_d_rec(q, peaked, emb, top_n=5, top_k=3)
    sim_flat, rec_flat = d_sim_d_rec(q, flat, emb, top_n=5, top_k=3)
    assert sim_flat > sim_peak, (sim_flat, sim_peak)
    assert rec_peak == 3.0 and rec_flat == 3.0
    print("  D_sim ranks flat retrieval above peaked; D_rec counts top-k ids")


def test_analyse_with_corpus_emits_d_sim_d_rec():
    emb = EmbeddingBackend(backend="hash", dim=64)
    corpus = [
        (f"r{i}", f"maintenance record about hydraulic leak number {i} system failure")
        for i in range(12)
    ]
    qs = [
        {"id": f"q{i}", "question": f"hydraulic leak pattern {i}", "type": "global"}
        for i in range(12)
    ]
    keo = {f"q{i}": 3.0 for i in range(12)}
    chunk = {f"q{i}": 2.0 for i in range(12)}
    result = analyse(qs, keo, chunk, None, corpus=corpus, embedder=emb, top_n=5, top_k=3)
    assert "D_sim" in result["correlations"] and "D_rec" in result["correlations"]
    assert result["corpus_n"] == 12
    assert all("D_sim" in r and "D_rec" in r for r in result["per_question"])
    print("  analyse with corpus emits D_sim/D_rec")


def test_load_corpus_records_from_csv():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tiny.csv")
        with open(path, "w") as f:
            f.write("c5,c119\n")
            for i in range(8):
                f.write(f"id{i},{'word ' * 10} narrative {i} about pumps\n")
        rows = load_corpus_records(path, limit=5)
        assert len(rows) == 5
        assert rows[0][0] == "id0"
    print("  load_corpus_records respects limit and id column")


def test_contamination_overlap_detects_shared_phrase():
    questions = [
        {"id": "a", "question": "fuel pump vibration during preflight inspection procedure"},
        {"id": "b", "question": "completely unrelated weather balloon sighting near airport"},
    ]
    corpus = "fuel pump vibration during preflight inspection procedure was noted by the crew"
    result = probe(questions, corpus, n=4)
    by_id = {r["question_id"]: r["overlap"] for r in result["per_question"]}
    assert by_id["a"] > by_id["b"]
    assert by_id["a"] > 0.5
    print("  contamination probe ranks a near-copy above a novel question")


def test_ngrams_empty_when_short():
    assert ngrams(tokens("hi"), 3) == set()
    assert ngrams(tokens("one two three four"), 3)
    print("  short texts yield empty n-gram sets")


def test_closed_book_prompt_has_no_corpus():
    msgs = closed_book_prompt("What patterns emerge across fuel-pump failures?")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    blob = " ".join(m["content"] for m in msgs).lower()
    assert "question:" in blob
    assert "retrieved" not in blob and "context:" not in blob
    # Overlap helper ranks a corpus-echoing answer above a novel one
    corpus = "fuel pump vibration during climbout noted by crew"
    hi = answer_corpus_overlap(
        "fuel pump vibration during climbout noted by crew again", corpus, n=3
    )
    lo = answer_corpus_overlap("unrelated weather balloon sighting", corpus, n=3)
    assert hi["overlap"] > lo["overlap"]
    print("  closed-book prompt is corpus-free; overlap ranks echo above novel")


if __name__ == "__main__":
    run_suite(
        "dispersion+contamination",
        [
            ("d_agg", test_d_agg_ranks_aggregation_questions_higher),
            ("d_ent", test_d_ent_counts_kg_mentions),
            ("spearman", test_spearman_perfect_and_null),
            ("tiny_overlap", test_analyse_refuses_tiny_overlap),
            ("analyse_d_agg", test_analyse_reports_correlation_block),
            ("d_sim_peaked", test_d_sim_flat_vs_peaked),
            ("analyse_corpus", test_analyse_with_corpus_emits_d_sim_d_rec),
            ("load_corpus", test_load_corpus_records_from_csv),
            ("contamination", test_contamination_overlap_detects_shared_phrase),
            ("ngrams", test_ngrams_empty_when_short),
            ("closed_book", test_closed_book_prompt_has_no_corpus),
        ],
    )
