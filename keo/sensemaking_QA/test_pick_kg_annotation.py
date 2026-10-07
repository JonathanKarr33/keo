"""Offline checks: mechanism controls must see labels and more than one weight."""

import os
import subprocess
import sys
import tempfile

from offline_harness import run_suite
from pick_kg import annotation_status, is_annotated

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
PICK = os.path.join(ROOT, "pick_kg.py")
CONTROL = os.path.normpath(os.path.join(
    ROOT, "../kg/output/kg_llm/gpt4o_with_nodes_batches/100/"
    "gpt4o_withprevnodes_100_with_entity_mentions_fixed.gml",
))
LABELED = os.path.normpath(os.path.join(
    ROOT, "../kg/output/kg_llm/gs100_total500/gemma3_4b_it/kg_gemma3_4b_it.gml",
))


def _write(body: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".gml", delete=False)
    handle.write(body)
    handle.close()
    return handle.name


def test_bare_edges_are_not_annotated():
    path = _write(
        'graph [\n'
        '  directed 1\n'
        '  node [ id 0 label "ENGINE" ]\n'
        '  node [ id 1 label "QUIT" ]\n'
        '  edge [ source 0 target 1 ]\n'
        ']\n'
    )
    status = annotation_status(path)
    assert status["labeled_edges"] == 0
    assert status["distinct_weights"] == 1
    assert not is_annotated(status)
    os.unlink(path)


def test_labeled_weighted_edges_pass():
    path = _write(
        'graph [\n'
        '  directed 1\n'
        '  node [ id 0 label "ENGINE" ]\n'
        '  node [ id 1 label "QUIT" ]\n'
        '  node [ id 2 label "PILOT" ]\n'
        '  edge [ source 0 target 1 weight 2 relations "HAS CAUSE" ]\n'
        '  edge [ source 0 target 2 weight 1 relations "USED BY" ]\n'
        ']\n'
    )
    status = annotation_status(path)
    assert status["labeled_edges"] == 2
    assert status["distinct_weights"] == 2
    assert is_annotated(status)
    os.unlink(path)


def test_cli_refuses_control_graph_and_accepts_labeled_graph():
    if not os.path.isfile(CONTROL) or not os.path.isfile(LABELED):
        return
    bare = subprocess.run(
        [PYTHON, PICK, "--check", "--require-annotated", CONTROL],
        capture_output=True, text=True,
    )
    assert bare.returncode == 1, bare.stderr
    labeled = subprocess.run(
        [PYTHON, PICK, "--check", "--require-annotated", LABELED],
        capture_output=True, text=True,
    )
    assert labeled.returncode == 0, labeled.stderr


def main():
    run_suite("pick_kg annotation", [
        ("bare edges not annotated", test_bare_edges_are_not_annotated),
        ("labeled weighted edges pass", test_labeled_weighted_edges_pass),
        ("CLI refuses control / accepts labeled", test_cli_refuses_control_graph_and_accepts_labeled_graph),
    ])


if __name__ == "__main__":
    main()
