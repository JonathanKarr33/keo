#!/usr/bin/env python3
"""
Offline checks for risk-sensitive budgeted retrieval and the severity lexicon.

No API calls, no KG files: every case is a hand-built graph where the right answer is
known by construction.

Run: python test_risk_retrieval_offline.py
"""

import json
import os
import tempfile

import networkx as nx

from offline_harness import run_suite
from risk_retrieval import (
    CHARS_PER_TOKEN,
    _edge_records,
    compare_policies,
    edge_tokens,
    greedy_budgeted_coverage,
    mst_baseline,
    rarity_of_high_severity,
    safety_metrics,
    top_weight_baseline,
    weight_degeneracy,
)
from severity_lexicon import (
    HIGH_SEVERITY_THRESHOLD,
    SEVERITY_CLASSES,
    SEVERITY_TERMS,
    coverage_report,
    edge_severity,
    export_for_adjudication,
    matched_terms,
    severity_class_of_text,
)


def test_legacy_multipliers_pinned():
    """The rarity ablation must keep the weights the earlier runs used."""
    legacy = {
        "fire": 3.0, "flame": 3.0, "explosion": 3.0, "crash": 3.0,
        "engine quit": 2.5, "engine failure": 2.5, "fuel starvation": 2.5,
        "smoke": 2.0, "uncontained": 2.5, "loss of control": 2.5,
        "hydraulic": 1.5, "electrical": 1.4, "icing": 1.6, "frozen": 1.5,
    }
    for term, expected in legacy.items():
        assert SEVERITY_TERMS[term] == expected, (term, SEVERITY_TERMS[term], expected)
    print(f"  all {len(legacy)} legacy multipliers unchanged; lexicon now {len(SEVERITY_TERMS)} terms")


def test_severity_takes_worst_case():
    """A triple mentioning several hazards is scored at the worst one."""
    assert edge_severity("cylinder", "developed", "corrosion") == SEVERITY_CLASSES["minor"]
    assert edge_severity("engine", "suffered", "in-flight fire") == SEVERITY_CLASSES["catastrophic"]
    mixed = edge_severity("loose gasket", "led to", "engine fire")
    assert mixed == SEVERITY_CLASSES["catastrophic"], mixed
    assert edge_severity("placard", "was", "replaced") == SEVERITY_CLASSES["minor"]
    assert edge_severity("bracket", "attached to", "panel") == 1.0
    assert severity_class_of_text("uncontained failure") == "hazardous"
    assert "engine fire" not in matched_terms("engine fire")  # matches "fire", not a phrase
    assert "fire" in matched_terms("engine fire")
    print("  worst-case severity wins; unmatched triples score 1.0")


def test_coverage_credits_each_entity_once():
    """
    Submodularity in action: a second edge about a covered entity adds little, so the
    greedy rule spreads the budget instead of piling onto one hub.
    """
    records = [
        {"head": "hub", "relation": "r", "tail": f"leaf{i}", "weight": 1.0,
         "severity": 1.0, "tokens": 5}
        for i in range(4)
    ] + [
        {"head": f"iso{i}", "relation": "r", "tail": f"iso{i}b", "weight": 1.0,
         "severity": 1.0, "tokens": 5}
        for i in range(4)
    ]
    result = greedy_budgeted_coverage(records, budget_tokens=20, use_severity=False)
    assert result["tokens_used"] <= 20
    # 4 edges at 5 tokens; isolated pairs cover 2 new entities each, hub edges only 1
    assert result["entities_covered"] >= 7, result["entities_covered"]
    print(f"  4 edges within 20 tokens cover {result['entities_covered']} entities (hub-avoiding)")


def _mundane_and_critical(mundane_weight: float, n_mundane: int = 30, n_critical: int = 3):
    records = [
        {"head": f"part{i}", "relation": "inspected", "tail": f"item{i}",
         "weight": mundane_weight, "severity": 1.0, "tokens": 6}
        for i in range(n_mundane)
    ]
    records += [
        {"head": f"engine{i}", "relation": "suffered", "tail": f"fire{i}",
         "weight": 1.0, "severity": 3.0, "tokens": 6}
        for i in range(n_critical)
    ]
    return records


def test_severity_weighting_changes_what_is_selected():
    """The severity-on/severity-off contrast is the causal test of the risk term."""
    records = _mundane_and_critical(mundane_weight=2.0)  # severity 3.0 outranks it

    budget = 60  # room for 10 of 33 edges
    plain = greedy_budgeted_coverage(records, budget, use_severity=False)
    risky = greedy_budgeted_coverage(records, budget, use_severity=True)

    plain_m = safety_metrics(plain, records, HIGH_SEVERITY_THRESHOLD)
    risky_m = safety_metrics(risky, records, HIGH_SEVERITY_THRESHOLD)
    assert risky_m["high_severity_recall"] > plain_m["high_severity_recall"], (
        plain_m["high_severity_recall"], risky_m["high_severity_recall"]
    )
    assert risky_m["high_severity_recall"] == 1.0, risky_m
    assert plain["tokens_used"] <= budget and risky["tokens_used"] <= budget
    print(
        f"  critical recall {plain_m['high_severity_recall']:.0%} → "
        f"{risky_m['high_severity_recall']:.0%} at equal budget"
    )


def test_multiplier_alone_fails_when_relevance_dominates():
    """
    The limitation the paper must state: severity is a multiplier, not a guarantee.
    With mundane relevance 5.0 and critical relevance 1.0, a 3x severity boost gives
    3.0 < 5.0, so the critical facts are still dropped.
    """
    records = _mundane_and_critical(mundane_weight=5.0)
    risky = greedy_budgeted_coverage(records, 60, use_severity=True)
    recall = safety_metrics(risky, records, HIGH_SEVERITY_THRESHOLD)["high_severity_recall"]
    assert recall == 0.0, recall
    print("  severity 3x loses to relevance 5x: multiplier alone recalls 0% of critical facts")


def test_risk_floor_gives_a_guarantee():
    """The reserved pass turns the soft preference into a floor, and costs coverage."""
    records = _mundane_and_critical(mundane_weight=5.0)
    budget = 60
    no_floor = greedy_budgeted_coverage(records, budget, use_severity=True)
    floored = greedy_budgeted_coverage(records, budget, use_severity=True, risk_floor=0.3)

    no_floor_m = safety_metrics(no_floor, records, HIGH_SEVERITY_THRESHOLD)
    floored_m = safety_metrics(floored, records, HIGH_SEVERITY_THRESHOLD)
    assert floored["tokens_used"] <= budget
    # 30% of 60 tokens = 18, enough for 3 critical edges at 6 tokens each
    assert floored_m["high_severity_recall"] == 1.0, floored_m
    assert floored_m["high_severity_recall"] > no_floor_m["high_severity_recall"]
    assert floored["reserved_tokens_used"] <= int(budget * 0.3)
    # the floor is not free: coverage is spent on critical facts instead of new entities
    assert floored["entities_covered"] <= no_floor["entities_covered"], (
        floored["entities_covered"], no_floor["entities_covered"]
    )
    print(
        f"  floor 0.3 lifts critical recall {no_floor_m['high_severity_recall']:.0%} → "
        f"{floored_m['high_severity_recall']:.0%}, entities covered "
        f"{no_floor['entities_covered']} → {floored['entities_covered']}"
    )


def test_risk_floor_respects_its_reserve():
    """A floor larger than what critical facts need must not waste the budget."""
    records = _mundane_and_critical(mundane_weight=5.0, n_critical=1)
    floored = greedy_budgeted_coverage(records, 120, use_severity=True, risk_floor=0.5)
    assert floored["reserved_tokens_used"] == 6, floored["reserved_tokens_used"]
    assert floored["tokens_used"] > floored["reserved_tokens_used"]  # remainder still spent
    assert floored["tokens_used"] <= 120
    print("  unused reserve returns to general selection rather than being wasted")


def test_fill_budget_spends_the_remainder():
    """
    Coverage saturates: once every entity sits at its best edge, marginal gain is zero
    and the greedy stops early. fill_budget spends the rest so context length can be
    held constant across arms without changing the objective's value.
    """
    # Saturation needs overlapping entities: every pair among 6 entities, so a handful
    # of edges covers all of them and the rest add nothing.
    entities = [f"e{i}" for i in range(6)]
    records = [
        {"head": a, "relation": "related to", "tail": b, "weight": 1.0,
         "severity": 1.0, "tokens": 5}
        for i, a in enumerate(entities) for b in entities[i + 1:]
    ]
    records[0]["severity"] = 3.0  # one critical edge, so recall is defined
    budget = 60  # room for 12 of 15 edges; coverage saturates after 3

    lean = greedy_budgeted_coverage(records, budget, use_severity=True)
    filled = greedy_budgeted_coverage(records, budget, use_severity=True, fill_budget=True)

    assert lean["filler_edges"] == 0
    assert filled["filler_edges"] > 0, "expected leftover budget to be spent"
    assert filled["tokens_used"] > lean["tokens_used"]
    assert filled["tokens_used"] <= budget
    # the objective is already maximal; filling adds edges, not coverage value
    assert abs(filled["objective"] - lean["objective"]) < 1e-9, (
        lean["objective"], filled["objective"]
    )
    lean_recall = safety_metrics(lean, records, HIGH_SEVERITY_THRESHOLD)["high_severity_recall"]
    filled_recall = safety_metrics(filled, records, HIGH_SEVERITY_THRESHOLD)["high_severity_recall"]
    assert filled_recall >= lean_recall
    print(
        f"  saturated at {lean['tokens_used']}/{budget} tokens, filled to "
        f"{filled['tokens_used']} with {filled['filler_edges']} filler edges, "
        f"objective unchanged"
    )


def test_budget_is_never_exceeded():
    """Every policy is a hard-constrained selection; the budget is the contract."""
    graph = nx.Graph()
    for i in range(40):
        graph.add_edge(f"n{i}", f"n{(i * 7) % 40}", relation="related_to", weight=float(i % 5 + 1))
    records = _edge_records(graph)
    for budget in (0, 1, 17, 120, 10_000):
        for name, sel in (
            ("top_weight", top_weight_baseline(records, budget)),
            ("mst", mst_baseline(graph, budget)),
            ("coverage", greedy_budgeted_coverage(records, budget, use_severity=False)),
            ("risk", greedy_budgeted_coverage(records, budget, use_severity=True)),
        ):
            assert sel["tokens_used"] <= budget, (name, budget, sel["tokens_used"])
    print("  no policy exceeds its budget at 0, 1, 17, 120, or 10k tokens")


def test_token_cost_matches_serialisation():
    tokens = edge_tokens("engine", "suffered", "in-flight fire")
    text = "engine suffered in-flight fire"
    assert tokens == int(-(-len(text) // CHARS_PER_TOKEN)) or abs(
        tokens - len(text) / CHARS_PER_TOKEN
    ) < 1.0, tokens
    assert edge_tokens("a", "b", "c") >= 1
    print(f"  '{text}' costs {tokens} tokens at {CHARS_PER_TOKEN} chars/token")


def test_degenerate_weights_are_flagged():
    """Uniform weights make frequency and MST baselines tie-break artefacts."""
    flat = nx.Graph()
    for i in range(10):
        flat.add_edge(f"a{i}", f"b{i}", relation="r", weight=1.0)
    assert weight_degeneracy(_edge_records(flat))["degenerate"] is True

    varied = nx.Graph()
    for i in range(10):
        varied.add_edge(f"a{i}", f"b{i}", relation="r", weight=float(i + 1))
    assert weight_degeneracy(_edge_records(varied))["degenerate"] is False
    print("  uniform-weight graphs flagged; varied-weight graphs are not")


def test_quintile_stat_is_suppressed_when_unidentifiable():
    """Don't report '0% below the top quintile' when the cutoff equals the minimum."""
    tied = [
        {"head": f"h{i}", "relation": "r", "tail": "fire", "weight": 1.0,
         "severity": 3.0, "tokens": 5}
        for i in range(10)
    ]
    stats = rarity_of_high_severity(tied)
    assert stats["quintile_cutoff_identifiable"] is False
    assert stats["share_below_top_quintile"] is None
    assert stats["share_at_min_weight"] == 1.0

    spread = tied + [
        {"head": f"x{i}", "relation": "r", "tail": "panel", "weight": float(i + 2),
         "severity": 1.0, "tokens": 5}
        for i in range(20)
    ]
    stats2 = rarity_of_high_severity(spread)
    assert stats2["quintile_cutoff_identifiable"] is True
    assert stats2["share_below_top_quintile"] == 1.0  # all critical facts are singletons
    print("  quintile stat reported only when the cutoff is identifiable")


def test_mst_can_lose_on_safety_recall():
    """
    The predicted failure mode: when critical facts are rare, the spanning tree keeps the
    frequent backbone and drops them. This is the paper's claim, as a test.

    The comparison has to be against the *risk floor* arm. Severity as a bare multiplier
    does not beat MST here — a severity bonus on a weight-1 edge still loses to a weight-10
    backbone edge — which is exactly why the floor exists (cf.
    test_multiplier_alone_fails_when_relevance_dominates).

    This test previously asserted that the multiplier-only arm beat MST, and passed only
    because MST's recall was being undercounted by an edge-key mismatch. With that fixed,
    multiplier-only and MST tie at 60%, and only the floor reaches 100%. See
    test_mst_recall_is_not_undercounted_by_key_mismatch.
    """
    graph = nx.Graph()
    for i in range(25):  # frequent mundane backbone
        graph.add_edge("hub", f"routine{i}", relation="inspected", weight=10.0)
    for i in range(5):   # rare critical leaves
        graph.add_edge(f"aircraft{i}", "engine fire", relation="experienced", weight=1.0)

    records = _edge_records(graph)
    budget = 180
    mst = safety_metrics(mst_baseline(graph, budget), records, HIGH_SEVERITY_THRESHOLD)
    multiplier_only = safety_metrics(
        greedy_budgeted_coverage(records, budget, use_severity=True),
        records, HIGH_SEVERITY_THRESHOLD,
    )
    floored = safety_metrics(
        greedy_budgeted_coverage(
            records, budget, use_severity=True,
            risk_floor=0.25, high_severity_threshold=HIGH_SEVERITY_THRESHOLD,
        ),
        records, HIGH_SEVERITY_THRESHOLD,
    )

    assert floored["high_severity_recall"] > mst["high_severity_recall"], (mst, floored)
    # A documented limitation, not an aspiration: assert the tie so a regression that
    # "improves" it is noticed rather than quietly believed.
    assert multiplier_only["high_severity_recall"] == mst["high_severity_recall"], \
        (mst, multiplier_only)
    # Length is comparable, so the recall gain is not bought with extra context
    assert abs(floored["tokens_used"] - mst["tokens_used"]) <= 0.05 * budget, (mst, floored)
    print(
        f"  critical recall: MST {mst['high_severity_recall']:.0%}, "
        f"severity-multiplier {multiplier_only['high_severity_recall']:.0%} (ties MST), "
        f"risk floor {floored['high_severity_recall']:.0%} "
        f"at {floored['tokens_used']} vs {mst['tokens_used']} tokens"
    )


def test_mst_recall_is_not_undercounted_by_key_mismatch():
    """
    MST's selected edges must key-match the reference records.

    Two regressions, both of which scored MST's recall as zero on edges it had actually
    selected — and both of which only ever inflated the policy MST is compared against:
    mst_baseline resolved relation labels from a shorter attribute list that omitted
    `relationship`, and `to_undirected()` can return an edge with its endpoints swapped
    while the key was order-sensitive.
    """
    from risk_retrieval import _edge_records, edge_key

    # Directed graph, relation stored under `relationship` — both failure modes at once
    graph = nx.DiGraph()
    graph.add_edge("in-flight fire", "engine", relationship="occurred in", weight=9.0)
    graph.add_edge("corrosion", "spar", relationship="found on", weight=8.0)
    graph.add_edge("bracket", "panel", relationship="attached to", weight=1.0)

    records = _edge_records(graph)
    assert all(r["relation"] != "related_to" for r in records), \
        f"relations lost from `relationship`: {[r['relation'] for r in records]}"

    plural = nx.Graph()
    plural.add_edge("engine", "quit", relations="HAS CAUSE", weight=1.0)
    assert _edge_records(plural)[0]["relation"] == "HAS CAUSE"

    mst = mst_baseline(graph, budget_tokens=10_000)
    assert mst["selected"], "MST selected nothing"
    assert all(r["relation"] != "related_to" for r in mst["selected"]), \
        f"mst_baseline lost the relation labels: {[r['relation'] for r in mst['selected']]}"

    record_keys = {edge_key(r["head"], r["relation"], r["tail"]) for r in records}
    mst_keys = {edge_key(r["head"], r["relation"], r["tail"]) for r in mst["selected"]}
    assert mst_keys <= record_keys, f"MST keys absent from records: {mst_keys - record_keys}"

    # With a budget large enough for everything, MST must recall every critical edge
    metrics = safety_metrics(mst, records)
    assert metrics["high_severity_total"] > 0, "fixture has no critical edges"
    assert metrics["high_severity_recall"] == 1.0, (
        f"MST recall undercounted: {metrics['high_severity_retrieved']}"
        f"/{metrics['high_severity_total']}"
    )
    print(f"  MST recalls {metrics['high_severity_retrieved']}/"
          f"{metrics['high_severity_total']} critical edges at an unlimited budget")


def test_edge_key_ignores_endpoint_order():
    """The identity used for recall must survive to_undirected()'s endpoint swap."""
    from risk_retrieval import edge_key

    assert edge_key("a", "rel", "b") == edge_key("b", "rel", "a")
    assert edge_key("a", "rel", "b") != edge_key("a", "other", "b")
    print("  orientation-insensitive, relation-sensitive")


def test_self_loop_gain_is_not_double_counted():
    """A self-loop covers one entity, so it must be credited once."""
    loop = [{"head": "engine", "relation": "related to", "tail": "engine",
             "weight": 5.0, "severity": 1.0, "tokens": 4}]
    pair = [{"head": "engine", "relation": "related to", "tail": "pump",
             "weight": 5.0, "severity": 1.0, "tokens": 4}]

    loop_result = greedy_budgeted_coverage(loop, budget_tokens=100, use_severity=False)
    pair_result = greedy_budgeted_coverage(pair, budget_tokens=100, use_severity=False)

    assert loop_result["entities_covered"] == 1, loop_result
    assert pair_result["entities_covered"] == 2, pair_result
    assert loop_result["objective"] == pytest_approx(5.0), loop_result["objective"]
    assert pair_result["objective"] == pytest_approx(10.0), pair_result["objective"]
    print(f"  self-loop objective={loop_result['objective']:.1f} (one entity), "
          f"pair={pair_result['objective']:.1f} (two)")


def pytest_approx(value, tol=1e-9):
    """Tiny float comparison helper so this suite stays dependency-free."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

        def __repr__(self):
            return f"≈{value}"

    return _Approx()


def test_compare_policies_end_to_end():
    graph = nx.Graph()
    for i in range(30):
        graph.add_edge(f"part{i}", f"comp{i % 7}", relation="installed on", weight=float(i % 4 + 1))
    graph.add_edge("cylinder 3", "uncontained failure", relation="led to", weight=1.0)
    graph.add_edge("fuel line", "engine fire", relation="caused", weight=1.0)

    result = compare_policies(graph, budget_tokens=200)
    assert set(result["policies"]) == {
        "top_weight", "mst", "budgeted_coverage", "risk_budgeted_coverage", "risk_floor_0.25"
    }, sorted(result["policies"])
    assert result["severity_prevalence"]["high_severity_total"] >= 2
    for name, m in result["policies"].items():
        assert m["tokens_used"] <= 200, name
        assert 0.0 <= m["high_severity_recall"] <= 1.0, name
    print(f"  4 policies scored at 200 tokens on a {graph.number_of_edges()}-edge graph")


def test_adjudication_export_is_annotatable():
    """The label file must be fillable by a human and auditable afterwards."""
    triples = [
        ("engine", "suffered", "in-flight fire"),
        ("bracket", "attached to", "panel"),
        ("cylinder", "showed", "corrosion"),
    ]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "labels.json")
        result = export_for_adjudication(triples, path)
        assert result["n_items"] == 3  # unmatched triples included on purpose
        with open(path) as f:
            payload = json.load(f)
        assert payload["classes"] == list(SEVERITY_CLASSES)
        for item in payload["items"]:
            assert item["human_class"] is None
            assert "prior_class" in item and "matched_terms" in item
        fire = next(i for i in payload["items"] if i["tail"] == "in-flight fire")
        assert fire["prior_class"] == "catastrophic"
        plain = next(i for i in payload["items"] if i["tail"] == "panel")
        assert plain["prior_class"] == "none" and plain["matched_terms"] == []

        only_matched = os.path.join(td, "matched.json")
        trimmed = export_for_adjudication(triples, only_matched, include_unmatched=False)
        assert trimmed["n_items"] == 2, trimmed
    print("  export includes unmatched triples (lexicon false negatives) by default")


def test_stratified_sample_is_reweightable():
    """
    Annotator batches over-sample the safety classes on purpose, so every item must
    carry the weight needed to recover corpus-level rates.
    """
    triples = (
        [(f"engine {i}", "suffered", "in-flight fire") for i in range(20)]
        + [(f"line {i}", "led to", "uncontained failure") for i in range(30)]
        + [(f"part {i}", "attached to", "panel") for i in range(400)]
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "batch.json")
        result = export_for_adjudication(triples, path, n_sample=60, seed=17)
        assert result["n_items"] == 60, result["n_items"]
        with open(path) as f:
            payload = json.load(f)

        assert payload["n_candidates_total"] == 450
        allocation = payload["sampling"]["allocation"]
        # "none" dominates the corpus but must not dominate the batch
        assert allocation["none"]["sampled"] < 450
        assert allocation["catastrophic"]["sampled"] > 0
        assert allocation["hazardous"]["sampled"] > 0

        # weighted counts must recover the corpus composition
        weighted_none = sum(
            i["sampling_weight"] for i in payload["items"] if i["prior_class"] == "none"
        )
        assert abs(weighted_none - 400) < 1.0, weighted_none
        for item in payload["items"]:
            assert item["sampling_weight"] is not None
            assert item["sampling_stratum"] == item["prior_class"]

        # reproducible, and order does not leak the prior
        again = os.path.join(td, "again.json")
        export_for_adjudication(triples, again, n_sample=60, seed=17)
        with open(again) as f:
            assert json.load(f)["items"] == payload["items"], "sampling is not reproducible"

        classes_in_order = [i["prior_class"] for i in payload["items"]]
        assert len(set(classes_in_order[:10])) > 1, "batch appears sorted by class"
    print("  60-item batch: weighted 'none' count recovers 400, sampling reproducible")


def test_census_export_has_no_weights():
    """Without --sample the export is a census, and claiming weights would be wrong."""
    triples = [(f"part {i}", "attached to", "panel") for i in range(10)]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "census.json")
        result = export_for_adjudication(triples, path, n_sample=50)  # larger than corpus
        assert result["allocation"] is None
        with open(path) as f:
            payload = json.load(f)
        assert payload["sampling"]["scheme"] == "census (all candidates)"
        assert all("sampling_weight" not in i for i in payload["items"])
    print("  census export carries no sampling weights")


def test_coverage_report_shape():
    report = coverage_report([
        "engine suffered fire", "bracket attached to panel", "cylinder showed corrosion",
    ])
    assert report["total"] == 3
    assert report["by_class"]["catastrophic"] == 1
    assert report["by_class"]["none"] == 1
    assert abs(report["share_safety_relevant"] - 2 / 3) < 1e-9
    print(f"  coverage report: {report['share_safety_relevant']:.0%} safety-relevant on 3 texts")


def main():
    tests = [
        ("legacy multipliers pinned", test_legacy_multipliers_pinned),
        ("severity is worst-case", test_severity_takes_worst_case),
        ("coverage credits entities once", test_coverage_credits_each_entity_once),
        ("severity changes selection", test_severity_weighting_changes_what_is_selected),
        ("multiplier alone is fragile", test_multiplier_alone_fails_when_relevance_dominates),
        ("risk floor guarantees recall", test_risk_floor_gives_a_guarantee),
        ("risk floor reserve bounded", test_risk_floor_respects_its_reserve),
        ("fill_budget spends remainder", test_fill_budget_spends_the_remainder),
        ("budget respected", test_budget_is_never_exceeded),
        ("token cost", test_token_cost_matches_serialisation),
        ("degenerate weights flagged", test_degenerate_weights_are_flagged),
        ("quintile stat guarded", test_quintile_stat_is_suppressed_when_unidentifiable),
        ("MST loses on safety recall", test_mst_can_lose_on_safety_recall),
        ("MST recall not undercounted", test_mst_recall_is_not_undercounted_by_key_mismatch),
        ("edge key ignores orientation", test_edge_key_ignores_endpoint_order),
        ("self-loop counted once", test_self_loop_gain_is_not_double_counted),
        ("compare_policies end to end", test_compare_policies_end_to_end),
        ("adjudication export", test_adjudication_export_is_annotatable),
        ("stratified sampling", test_stratified_sample_is_reweightable),
        ("census export", test_census_export_has_no_weights),
        ("coverage report", test_coverage_report_shape),
    ]
    run_suite("risk retrieval", tests)


if __name__ == "__main__":
    main()
