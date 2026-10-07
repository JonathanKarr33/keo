#!/usr/bin/env bash
# Serve an open-weight model locally (vLLM) for $0 answer generation.
#
# Prereqs: CUDA GPU + `pip install vllm` (or use Ollama — see bottom).
#
# Usage:
#   bash run_local_vllm.sh                          # start server (Qwen3-30B-A3B)
#   bash run_local_vllm.sh --serve-only
#   bash run_local_vllm.sh --smoke                  # start if needed + 20Q smoke
#   bash run_local_vllm.sh --paper-min              # full Plan A answers, $0 LLM
#
# Env overrides:
#   MODEL, PORT, TP (tensor parallel), MAX_MODEL_LEN, GPU_MEM_UTIL
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# vLLM needs CUDA. Say so here rather than failing several minutes later inside an import, which
# is how this last read as a model problem instead of a platform one.
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "ERROR: run_local_vllm.sh needs a CUDA GPU; this is macOS." >&2
  echo "Use the Apple Silicon path instead, which detects the platform for you:" >&2
  echo "  bash serve_local_model.sh" >&2
  exit 1
fi

# Prefer the repo venv: macOS ships no bare `python`, and system python3 lacks the
# deps. Override with PYTHON=/path/to/python (a GPU box may want its own env).
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

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
QUESTIONS="${QUESTIONS:-output/aviation_sensemaking_questions.json}"
KG_PATH="${KG_PATH:-../kg/output/knowledge_graph.gml}"
EMB="${EMB:-local}"
EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

SERVE_ONLY=0
SMOKE=0
PAPER_MIN=0
SKIP_SERVE=0
JUDGE=0
JUDGE_SMOKE=0
JUDGE_FINAL=0
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-}"
JUDGE_GRADE="${JUDGE_GRADE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve-only) SERVE_ONLY=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --paper-min) PAPER_MIN=1; shift ;;
    --skip-serve) SKIP_SERVE=1; shift ;;
    --judge) JUDGE=1; shift ;;
    --judge-smoke) JUDGE=1; JUDGE_SMOKE=1; shift ;;
    --final) JUDGE_FINAL=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; BASE_URL="http://127.0.0.1:${PORT}/v1"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-provider) JUDGE_PROVIDER="$2"; shift 2 ;;
    --judge-grade) JUDGE_GRADE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

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

server_up() {
  curl -sf "${BASE_URL%/v1}/v1/models" >/dev/null 2>&1 \
    || curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

start_vllm() {
  if server_up; then
    echo "vLLM already up at ${BASE_URL}"
    return 0
  fi
  if ! command -v vllm >/dev/null 2>&1 && ! python -c "import vllm" 2>/dev/null; then
    echo "vLLM not installed. Install with: pip install vllm"
    echo "Or point LOCAL_LLM_BASE_URL at Ollama/LM Studio and re-run with --skip-serve"
    exit 1
  fi
  echo "Starting vLLM: model=${MODEL} port=${PORT} tp=${TP}"
  "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --served-model-name "$MODEL" &
  local pid=$!
  echo "vLLM pid=$pid — waiting for readiness..."
  for i in $(seq 1 120); do
    if server_up; then
      echo "vLLM ready."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM process exited early"
      exit 1
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM"
  exit 1
}

run_gen() {
  local name="$1"; shift
  local out="evaluation_results/ablations/${name}/answers.json"
  mkdir -p "$(dirname "$out")"
  echo "=== $name → $out ==="
  "$PYTHON" generate_ablation_answers.py \
    --question-files "$QUESTIONS" \
    --output-file "$out" \
    --kg-path "$KG_PATH" \
    --provider local \
    --base-url "$BASE_URL" \
    --answer-model "$MODEL" \
    --embedding-backend "$EMB" \
    --embedding-model "$EMB_MODEL" \
    --max-answer-tokens 600 \
    "$@"
  LAST_OUT="$out"
}

judge_one() {
  local answers="$1"
  local need_key=""
  if [[ "$JUDGE_PROVIDER" == "anthropic" ]]; then need_key=ANTHROPIC_API_KEY
  elif [[ "$JUDGE_PROVIDER" == "openai" ]]; then need_key=OPENAI_API_KEY
  elif [[ "$JUDGE_PROVIDER" == "huggingface" ]]; then need_key=HF_TOKEN
  fi
  if [[ -n "$need_key" && -z "${!need_key:-}" ]]; then
    echo "WARNING: --judge set but $need_key missing; skip"
    return 0
  fi
  if [[ "$JUDGE_PROVIDER" == "claude_cli" ]] && ! command -v claude >/dev/null 2>&1 && [[ -z "${CLAUDE_CMD:-}" ]]; then
    echo "WARNING: --judge needs Claude Code ('claude' on PATH); skip"
    return 0
  fi
  local eval_dir
  eval_dir="$(dirname "$answers")/eval_${JUDGE_GRADE:-na}_${JUDGE_PROVIDER}_${JUDGE_MODEL//\//_}"
  echo "=== JUDGE grade=${JUDGE_GRADE:-?} $JUDGE_PROVIDER/$JUDGE_MODEL → $eval_dir ==="
  local extra=()
  if [[ -n "$JUDGE_GRADE" ]]; then extra+=(--judge-grade "$JUDGE_GRADE"); fi
  if [[ "$JUDGE_FINAL" -eq 1 ]]; then extra+=(--final); fi
  "$PYTHON" run_evaluation.py \
    --questions-file "$QUESTIONS" \
    --answers-file "$answers" \
    --output-dir "$eval_dir" \
    --evaluation-model "$JUDGE_MODEL" \
    --provider "$JUDGE_PROVIDER" \
    "${extra[@]}"
}

if [[ "$SKIP_SERVE" -eq 0 ]]; then
  start_vllm
fi

if [[ "$SERVE_ONLY" -eq 1 ]]; then
  echo "Serve-only: leaving vLLM running at ${BASE_URL}"
  wait
  exit 0
fi

# server_up() only proves something answers /v1/models — not that it serves $MODEL. With
# --skip-serve against Ollama/LM Studio a wrong tag reaches generation, which records
# "Error generating answer: ..." per question and (before the loader guard) got judged at
# the 1-5 floor. One uncached request settles it.
if ! "$PYTHON" preflight_model.py \
      --answer-model "$MODEL" --provider local --base-url "$BASE_URL"; then
  echo "Aborting: ${BASE_URL} is not serving ${MODEL}." >&2
  echo "Check the exact id with: curl -s ${BASE_URL}/models | ${PYTHON} -m json.tool" >&2
  exit 1
fi

if [[ "$SMOKE" -eq 1 ]]; then
  run_gen "local_smoke_routed" --method routed --sample-size 20 --edge-weight rarity --use-mst
  echo "Smoke done (local, \$0 LLM)."
  if [[ "$JUDGE" -eq 1 ]]; then judge_one "$LAST_OUT"; fi
  exit 0
fi

if [[ "$PAPER_MIN" -eq 1 ]]; then
  # Plan A methods at \$0 answer cost (local embeddings + local LLM)
  for method in vanilla textchunk keo hybrid routed; do
    run_gen "local_${MODEL##*/}_${method}" --method "$method" --use-mst --edge-weight freq
    if [[ "$JUDGE" -eq 1 ]]; then judge_one "$LAST_OUT"; fi
  done
  run_gen "local_${MODEL##*/}_keo_rarity" --method keo --use-mst --edge-weight rarity
  if [[ "$JUDGE" -eq 1 ]]; then judge_one "$LAST_OUT"; fi
  echo "Paper-min local sweep done."
  exit 0
fi

echo "Server ready at ${BASE_URL}"
echo "Example:"
echo "  python generate_ablation_answers.py \\"
echo "    --question-files $QUESTIONS --output-file evaluation_results/ablations/local_test/answers.json \\"
echo "    --method routed --sample-size 5 --provider local --base-url $BASE_URL \\"
echo "    --answer-model $MODEL --embedding-backend local"
echo ""
echo "Ollama alternative (no vLLM):"
echo "  ollama pull qwen2.5:14b"
echo "  export LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1"
echo "  MODEL=qwen2.5:14b bash run_local_vllm.sh --skip-serve --smoke"
