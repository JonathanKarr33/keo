#!/usr/bin/env bash
# Poll a --run-id ablation directory and judge each completed arm.
#
# Safe to run while generation is still writing later arms: an arm is judged only
# when answers.json and answers_meta.json agree on n_questions, and judge_skip
# says the eval does not already match the answerer. Does not start the answerer
# or load MiniLM. Retries when Claude is quota-blocked.
#
#   bash watch_and_judge.sh --run-id phi4
#   bash watch_and_judge.sh --run-id phi4 --answer-model mlx-community/phi-4-4bit
#   bash watch_and_judge.sh --run-id phi4 --once          # single pass, no sleep loop
#   bash watch_and_judge.sh --run-id phi4 --poll 180
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# shellcheck source=platform.sh
source "$ROOT/platform.sh"
keo_detect_platform "$(cd "$ROOT/../.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
    PYTHON="$ROOT/../../.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi

RUN_ID=""
ANSWER_MODEL="${ANSWER_MODEL:-${KEO_DEFAULT_LOCAL_MODEL:-mlx-community/phi-4-4bit}}"
QUESTIONS="${QUESTIONS:-output/aviation_sensemaking_questions.json}"
POLL=120
ONCE=0
EXPECTED_N=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --answer-model) ANSWER_MODEL="$2"; shift 2 ;;
    --questions) QUESTIONS="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --expected-n) EXPECTED_N="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "keo: --run-id is required" >&2
  exit 2
fi
case "$RUN_ID" in
  *[!A-Za-z0-9_-]*|"")
    echo "keo: --run-id must be [A-Za-z0-9_-], got: $RUN_ID" >&2
    exit 2
    ;;
esac

ARM_ROOT="evaluation_results/ablations/${RUN_ID}"
LOG="evaluation_results/ablations/${RUN_ID}/watch_and_judge.log"
mkdir -p "$ARM_ROOT"

echo "watch_and_judge: ARM_ROOT=$ARM_ROOT  model=$ANSWER_MODEL  poll=${POLL}s"
echo "  log → $LOG"

arm_ready() {
  # Exit 0 if answers.json looks finished for this arm.
  local dir="$1"
  local answers="${dir}/answers.json"
  local meta="${dir}/answers_meta.json"
  [[ -f "$answers" && -f "$meta" ]] || return 1
  "$PYTHON" - "$answers" "$meta" "$EXPECTED_N" <<'PY'
import json, sys
answers_path, meta_path, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
meta = json.load(open(meta_path))
answers = json.load(open(answers_path))
n_meta = int(meta.get("n_questions") or 0)
n_ans = len(answers) if isinstance(answers, list) else 0
if n_meta <= 0 or n_ans <= 0:
    sys.exit(1)
if n_ans < n_meta:
    sys.exit(1)
if expected > 0 and n_ans < expected:
    sys.exit(1)
# Prefer a non-empty method field in the first non-null answer blob.
sys.exit(0)
PY
}

judge_one() {
  local name="$1"
  local dir="${ARM_ROOT}/${name}"
  local skip_line=""
  # judge_skip exits 0 to skip, 2 to judge. Keep set -e from aborting.
  if skip_line="$("$PYTHON" judge_skip.py "$dir" --requested-model "$ANSWER_MODEL")"; then
    echo "=== $name — SKIP (${skip_line#*$'\t'}) ===" | tee -a "$LOG"
    return 0
  fi
  if ! arm_ready "$dir"; then
    echo "=== $name — waiting (answers incomplete) ===" | tee -a "$LOG"
    return 0
  fi
  echo "=== Judging $name ===" | tee -a "$LOG"
  if "$PYTHON" run_evaluation.py \
      --questions-file "$QUESTIONS" \
      --answers-file "${dir}/answers.json" \
      --output-dir "$dir" >>"$LOG" 2>&1; then
    echo "=== $name — judged OK ===" | tee -a "$LOG"
    return 0
  fi
  # Quota / auth failures leave the arm unjudged; the next poll retries.
  echo "=== $name — judge failed (will retry); see $LOG ===" | tee -a "$LOG"
  return 1
}

pass_once() {
  local any=0
  local failed=0
  local dir name
  shopt -s nullglob
  for dir in "$ARM_ROOT"/*/; do
    name="$(basename "$dir")"
    # Skip non-arm leftovers
    [[ -f "${dir}/answers.json" ]] || continue
    any=1
    if ! judge_one "$name"; then
      failed=1
    fi
  done
  shopt -u nullglob
  if [[ "$any" -eq 0 ]]; then
    echo "No arm directories under $ARM_ROOT yet." | tee -a "$LOG"
  fi
  return "$failed"
}

while true; do
  pass_once || true
  # Done when every arm with answers is skippable (already judged for this model).
  pending=0
  shopt -s nullglob
  for dir in "$ARM_ROOT"/*/; do
    [[ -f "${dir}/answers.json" ]] || continue
    if arm_ready "$dir"; then
      if ! "$PYTHON" judge_skip.py "$dir" --requested-model "$ANSWER_MODEL" >/dev/null; then
        pending=$((pending + 1))
      fi
    else
      pending=$((pending + 1))
    fi
  done
  shopt -u nullglob
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pending=$pending" | tee -a "$LOG"
  if [[ "$ONCE" -eq 1 ]]; then
    exit 0
  fi
  if [[ "$pending" -eq 0 ]]; then
    # Still generating? If the sweep log shows activity, keep waiting a bit.
    echo "All ready arms judged. Sleeping ${POLL}s in case more arms appear…" | tee -a "$LOG"
  fi
  sleep "$POLL"
done
