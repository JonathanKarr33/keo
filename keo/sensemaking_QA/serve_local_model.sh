#!/usr/bin/env bash
# Start the local answer-model server for this machine, whatever it is.
#
#   bash serve_local_model.sh                 # foreground (Ctrl-C to stop)
#   bash serve_local_model.sh --detach        # background, logs to local_server.log
#   bash serve_local_model.sh --status        # is a server answering?
#   MODEL=<id> bash serve_local_model.sh
#
# Apple Silicon serves through mlx-lm, CUDA hosts delegate to vLLM. Picking the wrong one used to
# present as a model error rather than a platform mismatch, so the platform is detected instead of
# assumed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck source=platform.sh
source "$ROOT/platform.sh"

REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
PORT="${PORT:-8000}"
BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
LOG="${LOG:-$ROOT/local_server.log}"
DETACH=0
STATUS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --status) STATUS=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

keo_detect_platform "$REPO_ROOT"
MODEL="${MODEL:-$KEO_DEFAULT_LOCAL_MODEL}"
echo "Platform: ${KEO_OS}/${KEO_ARCH}  accel=${KEO_ACCEL}"

if [[ "$STATUS" == "1" ]]; then
  if curl -s -m 10 "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "✓ server answering at ${BASE_URL}"
    curl -s -m 10 "${BASE_URL}/models" | "$REPO_ROOT/.venv/bin/python" -c \
      'import json,sys; print("  serving:", ", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' \
      2>/dev/null || true
    exit 0
  fi
  echo "✗ nothing answering at ${BASE_URL}. Start it with:"
  keo_serve_hint
  exit 1
fi

if curl -s -m 10 "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "✓ a server is already answering at ${BASE_URL} — leaving it alone"
  exit 0
fi

case "$KEO_ACCEL" in
  mlx)
    if [[ -z "$KEO_SERVE_PYTHON" ]]; then
      echo "ERROR: mlx-lm needs python >=3.10 and none was found." >&2
      keo_serve_hint >&2
      exit 1
    fi
    if ! "$KEO_SERVE_PYTHON" -c 'import mlx_lm' 2>/dev/null; then
      echo "ERROR: $KEO_SERVE_PYTHON has no mlx_lm. Install it there:" >&2
      echo "  $KEO_SERVE_PYTHON -m pip install mlx-lm" >&2
      exit 1
    fi
    echo "Serving $MODEL via mlx-lm on port $PORT"
    if [[ "$DETACH" == "1" ]]; then
      pid="$(keo_detach "$LOG" "$KEO_SERVE_PYTHON" -m mlx_lm server --model "$MODEL" --port "$PORT")"
      echo "  pid $pid, log $LOG"
      for _ in $(seq 1 60); do
        if curl -s -m 5 "${BASE_URL}/models" >/dev/null 2>&1; then
          echo "✓ answering at ${BASE_URL}"
          exit 0
        fi
        sleep 2
      done
      echo "ERROR: server did not come up within 120s; see $LOG" >&2
      exit 1
    fi
    exec "$KEO_SERVE_PYTHON" -m mlx_lm server --model "$MODEL" --port "$PORT"
    ;;
  cuda)
    echo "Delegating to vLLM"
    exec bash run_local_vllm.sh --serve-only
    ;;
  *)
    echo "ERROR: no Apple Silicon or CUDA GPU detected on ${KEO_OS}/${KEO_ARCH}." >&2
    echo "Local generation on CPU is too slow to be useful for a sweep." >&2
    exit 1
    ;;
esac
