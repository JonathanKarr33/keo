#!/usr/bin/env bash
# Re-score existing answer files with Sonnet, into a new run-id.
#
# Haiku evals stay where they are. This copies answers.json and answers_meta.json
# only, then (on --execute) calls run_evaluation.py --final. It does not generate
# answers and it does not call Claude unless --execute is passed.
#
#   bash judge_sonnet_matrix.sh
#   bash judge_sonnet_matrix.sh --stage          # copy answer files, still no judge
#   bash judge_sonnet_matrix.sh --execute        # Sonnet on every existing control arm
#
# Sources default to the two judged generators. Arms are the seven controls.
# A missing answers.json is listed and skipped, not invented.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="plan"
QUESTIONS="${QUESTIONS:-output/aviation_sensemaking_questions.json}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
    PYTHON="$ROOT/../../.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) MODE="plan"; shift ;;
    --stage) MODE="stage"; shift ;;
    --execute) MODE="execute"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# source_run:dest_run  (dest is a new tree; Haiku files are not overwritten)
PAIRS="phi4:phi4_sonnet_all _gemma4e4b:gemma4e4b_sonnet_all"
ARMS="mst_on_freq ctrl_chunk_tok ctrl_chunk_sum ctrl_chunk_dedup ctrl_keo_norel ctrl_keo_shuf_s13 textchunk"

ready=0
missing=0
for pair in $PAIRS; do
  src="${pair%%:*}"
  dest="${pair##*:}"
  echo "=== $src  ->  evaluation_results/ablations/$dest  (Sonnet, new tree) ==="
  for arm in $ARMS; do
    answers="evaluation_results/ablations/${src}/${arm}/answers.json"
    meta="evaluation_results/ablations/${src}/${arm}/answers_meta.json"
    if [[ ! -f "$answers" || ! -f "$meta" ]]; then
      echo "  missing  $arm"
      missing=$((missing + 1))
      continue
    fi
    ready=$((ready + 1))
    outdir="evaluation_results/ablations/${dest}/${arm}"
    echo "  ready    $arm"
    if [[ "$MODE" == "plan" ]]; then
      continue
    fi
    mkdir -p "$outdir"
    cp -f "$answers" "$outdir/answers.json"
    cp -f "$meta" "$outdir/answers_meta.json"
    if [[ "$MODE" == "execute" ]]; then
      echo "  judging  $arm"
      "$PYTHON" run_evaluation.py \
        --questions-file "$QUESTIONS" \
        --answers-file "$outdir/answers.json" \
        --output-dir "$outdir" \
        --final
    fi
  done
done

echo "Ready arms: $ready. Missing: $missing."
if [[ "$MODE" == "plan" ]]; then
  echo "Plan only. --stage copies answer files. --execute calls Sonnet. Do not pool with Haiku."
fi
