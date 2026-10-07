#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Prefer the repo venv: macOS ships no bare `python`, and system python3 lacks the
# deps. Override with PYTHON=/path/to/python.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
    PYTHON="$ROOT/../../.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi
echo "Python: $PYTHON"

KG="${ROOT}/data/pilot_kg.gml"
SAMPLE=5
MODEL="${ANSWER_MODEL:-mlx-community/phi-4-4bit}"
PROVIDER="${PROVIDER:-local}"
API_PROVIDER="${API_PROVIDER:-nebius}"
BASE_URL="${BASE_URL:-${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8000/v1}}"
EMB="${EMB:-hash}"
CORPUS="${ROOT}/data/sample.csv"
TEXT_COL="${TEXT_COL:-narrative}"
QA_DIR="${ROOT}/../sensemaking_QA"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kg) KG="$2"; shift 2 ;;
    --sample-size) SAMPLE="$2"; shift 2 ;;
    --answer-model) MODEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --corpus-csv) CORPUS="$2"; shift 2 ;;
    --text-column) TEXT_COL="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# generate_ablation_answers.py runs with cwd=$QA_DIR, so relative --kg/--corpus paths break.
KG="$(cd "$(dirname "$KG")" && pwd)/$(basename "$KG")"
CORPUS="$(cd "$(dirname "$CORPUS")" && pwd)/$(basename "$CORPUS")"

if [[ ! -f "$KG" ]]; then
  echo "Missing KG at $KG. Run build_kg_stub.py first (see README)."
  exit 1
fi

OUT="${ROOT}/runs/pilot_keo/answers.json"
mkdir -p "$(dirname "$OUT")"
cd "$QA_DIR"
EXTRA=()
if [[ "$PROVIDER" == "local" ]]; then
  EXTRA+=(--base-url "$BASE_URL")
fi
if [[ "$PROVIDER" == "huggingface" ]]; then
  EXTRA+=(--API-provider "$API_PROVIDER")
fi

# Confirm the model answers before generating: a dead ID would fill the pilot with
# "Error generating answer: ..." and the second-domain result would read as a failure of
# transfer rather than of configuration.
if ! "$PYTHON" preflight_model.py --answer-model "$MODEL" --provider "$PROVIDER" "${EXTRA[@]}"; then
  echo "Aborting pilot: $MODEL did not answer." >&2
  exit 1
fi

"$PYTHON" generate_ablation_answers.py \
  --question-files "${ROOT}/prompts.json" \
  --output-file "$OUT" \
  --kg-path "$KG" \
  --method keo \
  --use-mst \
  --edge-weight freq \
  --answer-model "$MODEL" \
  --provider "$PROVIDER" \
  --embedding-backend "$EMB" \
  --corpus-csv "$CORPUS" \
  --text-column "$TEXT_COL" \
  --sample-size "$SAMPLE" \
  "${EXTRA[@]}"

echo "Pilot answers → $OUT"
