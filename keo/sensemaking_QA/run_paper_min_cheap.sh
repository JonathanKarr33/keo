#!/usr/bin/env bash
# Cheap paper-min answer sweep (Plan A methods).
#
# Default: HuggingFace/Nebius open models + local embeddings (~$12 answers).
# Judges: --judge = paper-parity Claude Code (pinned Haiku, Pro/Max).
#          JUDGE_PROVIDER=anthropic for Messages API twin.
#          --judge-smoke = same stack, smoke grade label.
# Legacy GPT: JUDGE_PROVIDER=openai.
#
# Usage:
#   export HF_TOKEN=...
#   claude auth login
#   bash run_paper_min_cheap.sh --smoke --judge
#   bash run_paper_min_cheap.sh --judge --final
#   JUDGE_PROVIDER=anthropic ANTHROPIC_API_KEY=... bash run_paper_min_cheap.sh --judge
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

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

SMOKE=0
JUDGE=0
JUDGE_SMOKE=0
JUDGE_FINAL=0
PROVIDER="${PROVIDER:-huggingface}"
API_PROVIDER="${API_PROVIDER:-nebius}"
BASE_URL="${BASE_URL:-${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8000/v1}}"
EMB="${EMB:-local}"
EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
QUESTIONS="${QUESTIONS:-output/aviation_sensemaking_questions.json}"
KG_PATH="${KG_PATH:-../kg/output/knowledge_graph.gml}"
TAG="${TAG:-paper_min}"
# Defaults filled after flag parse (paper vs smoke)
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-}"
JUDGE_GRADE="${JUDGE_GRADE:-}"

# Plan A P0 models (override with MODELS="id1 id2")
MODELS="${MODELS:-Qwen/Qwen3.8-27B google/gemma-4-26B-A4B-it mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
METHODS=(vanilla textchunk keo hybrid routed)
EXTRA_WEIGHTS=(rarity)
OUTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --judge) JUDGE=1; shift ;;
    --judge-smoke) JUDGE=1; JUDGE_SMOKE=1; shift ;;
    --final) JUDGE_FINAL=1; shift ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-provider) JUDGE_PROVIDER="$2"; shift 2 ;;
    --judge-grade) JUDGE_GRADE="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# Resolve judge: default paper via Claude Code (pinned); --judge-smoke marks smoke grade
if [[ "$JUDGE" -eq 1 ]]; then
  if [[ "$JUDGE_SMOKE" -eq 1 ]]; then
    JUDGE_PROVIDER="${JUDGE_PROVIDER:-claude_cli}"
    JUDGE_GRADE="${JUDGE_GRADE:-smoke}"
  else
    JUDGE_PROVIDER="${JUDGE_PROVIDER:-claude_cli}"
    JUDGE_GRADE="${JUDGE_GRADE:-paper}"
  fi
  if [[ "$JUDGE_FINAL" -eq 1 ]]; then
    JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-5}"
  else
    JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5-20251001}"
  fi
fi

safe_name() {
  echo "$1" | tr '/:' '__' | tr -cd 'A-Za-z0-9._-'
}

run_one() {
  local model="$1" method="$2" weight="$3"
  local mname
  mname="$(safe_name "$model")"
  local out="evaluation_results/ablations/${TAG}_${mname}_${method}_${weight}/answers.json"
  mkdir -p "$(dirname "$out")"
  local extra=()
  if [[ "$PROVIDER" == "local" ]]; then
    extra+=(--base-url "$BASE_URL")
  fi
  if [[ "$PROVIDER" == "huggingface" ]]; then
    extra+=(--API-provider "$API_PROVIDER")
  fi
  local sample=()
  if [[ "$SMOKE" -eq 1 ]]; then
    sample=(--sample-size 20)
  fi
  echo "=== ${model} | ${method} | ${weight} ==="
  "$PYTHON" generate_ablation_answers.py \
    --question-files "$QUESTIONS" \
    --output-file "$out" \
    --kg-path "$KG_PATH" \
    --provider "$PROVIDER" \
    --answer-model "$model" \
    --method "$method" \
    --use-mst \
    --edge-weight "$weight" \
    --embedding-backend "$EMB" \
    --embedding-model "$EMB_MODEL" \
    --max-answer-tokens 600 \
    "${extra[@]}" \
    "${sample[@]}"
  OUTS+=("$out")
}

echo "provider=$PROVIDER emb=$EMB smoke=$SMOKE"
echo "models: $MODELS"

# Verify each model serves a completion before generating with it. This loop spans
# several models, so a dead ID skips its own arms rather than killing the sweep — but it
# must not proceed silently, because every arm would otherwise record the provider's
# error text as its answer (see test_judge_guards_offline.py).
PREFLIGHT_BASE=(--provider "$PROVIDER")
if [[ "$PROVIDER" == "huggingface" ]]; then
  PREFLIGHT_BASE+=(--API-provider "$API_PROVIDER")
elif [[ "$PROVIDER" == "local" ]]; then
  PREFLIGHT_BASE+=(--base-url "$BASE_URL")
fi

USABLE_MODELS=()
for model in $MODELS; do
  if "$PYTHON" preflight_model.py --answer-model "$model" "${PREFLIGHT_BASE[@]}"; then
    USABLE_MODELS+=("$model")
  else
    echo "SKIP model $model — preflight failed (no arms generated for it)" >&2
  fi
done
if [[ "${#USABLE_MODELS[@]}" -eq 0 ]]; then
  echo "Aborting: none of the requested models answered. Models tried: $MODELS" >&2
  exit 1
fi
echo "usable models: ${USABLE_MODELS[*]}"

for model in ${USABLE_MODELS[@]+"${USABLE_MODELS[@]}"}; do
  if [[ "$SMOKE" -eq 1 ]]; then
    run_one "$model" routed rarity
    break
  fi
  for method in "${METHODS[@]}"; do
    run_one "$model" "$method" freq
  done
  for w in "${EXTRA_WEIGHTS[@]}"; do
    run_one "$model" keo "$w"
  done
done

echo "Done. Outputs under evaluation_results/ablations/${TAG}_*"
if [[ "${#USABLE_MODELS[@]}" -ne "$(echo "$MODELS" | wc -w | tr -d ' ')" ]]; then
  echo "NOTE: some models were skipped by preflight; tables will be missing those rows." >&2
fi

if [[ "$JUDGE" -eq 1 ]]; then
  need_key=""
  if [[ "$JUDGE_PROVIDER" == "anthropic" ]]; then need_key=ANTHROPIC_API_KEY
  elif [[ "$JUDGE_PROVIDER" == "openai" ]]; then need_key=OPENAI_API_KEY
  elif [[ "$JUDGE_PROVIDER" == "huggingface" ]]; then need_key=HF_TOKEN
  fi
  if [[ -n "$need_key" && -z "${!need_key:-}" ]]; then
    echo "WARNING: --judge set but $need_key missing; skipping judge"
  elif [[ "$JUDGE_PROVIDER" == "claude_cli" ]] && ! command -v claude >/dev/null 2>&1 && [[ -z "${CLAUDE_CMD:-}" ]]; then
    echo "WARNING: --judge needs Claude Code ('claude' on PATH); install / claude auth login"
  else
    final_flag=()
    if [[ "$JUDGE_FINAL" -eq 1 ]]; then final_flag=(--final); fi
    grade_flag=()
    if [[ -n "$JUDGE_GRADE" ]]; then grade_flag=(--judge-grade "$JUDGE_GRADE"); fi
    for out in ${OUTS[@]+"${OUTS[@]}"}; do
      eval_dir="$(dirname "$out")/eval_${JUDGE_GRADE:-na}_${JUDGE_PROVIDER}_${JUDGE_MODEL//\//_}"
      echo "=== JUDGE grade=${JUDGE_GRADE:-?} $JUDGE_PROVIDER/$JUDGE_MODEL → $eval_dir ==="
      "$PYTHON" run_evaluation.py \
        --questions-file "$QUESTIONS" \
        --answers-file "$out" \
        --output-dir "$eval_dir" \
        --evaluation-model "$JUDGE_MODEL" \
        --provider "$JUDGE_PROVIDER" \
        ${grade_flag[@]+"${grade_flag[@]}"} \
        ${final_flag[@]+"${final_flag[@]}"}
    done
  fi
else
  echo "Tip: --judge (Claude Code paper-parity, pinned Haiku) | --judge --final (Sonnet). JUDGE_PROVIDER=anthropic for API."
fi
