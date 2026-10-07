#!/usr/bin/env bash
# Plan or launch the seven mechanism controls on the labeled, weighted graph.
#
# The 100-record GPT-4o extract has no relation labels and no edge weights.
# run_ablation_matrix.sh --controls refuses that file. This wrapper pins the
# 1,468-edge gemma3 graph, a fresh run-id, and the full 133-question set.
# It does not start a model unless --execute is passed.
#
#   bash run_labeled_controls.sh                 # print the command and check the graph
#   bash run_labeled_controls.sh --execute       # generate (local answerer must already be up)
#   PROVIDER=openrouter bash run_labeled_controls.sh --execute
#       requires OPENROUTER_API_KEY; see run_openrouter_generator.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

EXECUTE=0
RUN_ID="${RUN_ID:-labeled_graph}"
SAMPLE="${SAMPLE:-133}"
KG="${KG:-../kg/output/kg_llm/gs100_total500/gemma3_4b_it/kg_gemma3_4b_it.gml}"
PROVIDER="${PROVIDER:-local}"
ANSWER_MODEL="${ANSWER_MODEL:-mlx-community/phi-4-4bit}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --plan) EXECUTE=0; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --sample-size) SAMPLE="$2"; shift 2 ;;
    --kg-path) KG="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --answer-model) ANSWER_MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
    PYTHON="$ROOT/../../.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

"$PYTHON" pick_kg.py --check --require-annotated "$KG"

echo "Labeled-graph controls (seven arms + the keo reference)."
echo "  kg:       $KG"
echo "  run-id:   $RUN_ID   (will not overwrite phi4/ or _gemma4e4b/)"
echo "  provider: $PROVIDER"
echo "  model:    $ANSWER_MODEL"
echo "  n:        $SAMPLE"
echo "  command:"
echo "    bash run_ablation_matrix.sh --controls \\"
echo "      --kg-path \"$KG\" --run-id \"$RUN_ID\" --sample-size \"$SAMPLE\" \\"
echo "      --provider \"$PROVIDER\" --answer-model \"$ANSWER_MODEL\""

if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Plan only. Pass --execute to generate. This does not judge."
  exit 0
fi

if [[ "$PROVIDER" == "openrouter" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set. Refusing to start." >&2
  exit 2
fi

exec bash run_ablation_matrix.sh --controls \
  --kg-path "$KG" \
  --run-id "$RUN_ID" \
  --sample-size "$SAMPLE" \
  --provider "$PROVIDER" \
  --answer-model "$ANSWER_MODEL"
