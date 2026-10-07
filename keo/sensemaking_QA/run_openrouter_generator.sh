#!/usr/bin/env bash
# Larger Gemma on OpenRouter, same seven controls, own run-id.
#
# Does not call the network unless --execute is passed, and --execute refuses
# when OPENROUTER_API_KEY is unset. Scores must not be pooled with Gemma-4 E4B
# or Phi-4.
#
# Resume (paid-API safe, same idea as Claude judge checkpoints):
#   - Each answered question is flushed to answers.json with meta.partial=true
#   - Billing / hard rate-limit raises exit code 3; re-run the same command to continue
#   - Completed arms (partial=false, full n) are skipped; pass --force via the matrix to redo
#   - Disk completion cache under evaluation_results/.completion_cache also avoids re-billing
#     identical prompts
#   - Parallelism: KEO_GEN_WORKERS (default 10 here; max 16). Sequential = KEO_GEN_WORKERS=1
#
#   bash run_openrouter_generator.sh
#   # with key in repo-root .env or exported:
#   bash run_openrouter_generator.sh --execute
#
# Recommended: google/gemma-4-31b-it (dense peer to Gemma-3 27B).
# Alt MoE: OPENROUTER_MODEL=google/gemma-4-26b-a4b-it RUN_ID=openrouter_gemma4_26b_a4b
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"

# Load repo-root .env if present (does not override already-exported vars)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

EXECUTE=0
RUN_ID="${RUN_ID:-openrouter_gemma4_31b}"
MODEL="${OPENROUTER_MODEL:-google/gemma-4-31b-it}"
# API-bound; 10 concurrent completions are usually fine on OpenRouter.
export KEO_GEN_WORKERS="${KEO_GEN_WORKERS:-10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --plan) EXECUTE=0; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --answer-model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

_key="${OPENROUTER_API_KEY:-}"
_key_lc="$(printf '%s' "$_key" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$_key" || "$_key_lc" == *your_openrouter* || "$_key_lc" == *placeholder* || "$_key" == "changeme" ]]; then
  echo "OPENROUTER_API_KEY is missing or still a placeholder."
  echo "Copy .env.example → .env at the repo root and paste a real key from https://openrouter.ai/keys"
  echo "Default model: $MODEL"
  echo "Run-id would be: $RUN_ID (separate from _gemma4e4b and phi4)."
  if [[ "$EXECUTE" -eq 1 ]]; then
    exit 2
  fi
  exit 0
fi
unset _key _key_lc

exec bash run_labeled_controls.sh \
  $([[ "$EXECUTE" -eq 1 ]] && echo --execute || echo --plan) \
  --provider openrouter \
  --answer-model "$MODEL" \
  --run-id "$RUN_ID"
