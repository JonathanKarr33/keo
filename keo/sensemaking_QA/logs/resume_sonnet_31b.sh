#!/usr/bin/env bash
set -euo pipefail
cd /Users/jonathankarr/Documents/keo/keo/sensemaking_QA
LOG="${1:?log path}"
exec >>"$LOG" 2>&1
echo "=== resume script start $(date) ==="
/Users/jonathankarr/Documents/keo/.venv/bin/python - <<'PY'
import time
from datetime import datetime
from zoneinfo import ZoneInfo
tz = ZoneInfo("America/Indianapolis")
now = datetime.now(tz)
target = now.replace(hour=14, minute=35, second=0, microsecond=0)
if now >= target:
    # if already past today's window, start immediately
    wait = 0
else:
    wait = (target - now).total_seconds()
print(f"now={now.isoformat()} target={target.isoformat()} wait_s={wait:.0f}", flush=True)
if wait > 0:
    time.sleep(wait)
print(f"resume start {datetime.now(tz).isoformat()}", flush=True)
PY
export KEO_JUDGE_WORKERS=8
export KEO_ACTION_NLP_ONLY=1
PYTHON=/Users/jonathankarr/Documents/keo/.venv/bin/python
QUESTIONS=output/aviation_sensemaking_questions.json
DEST=evaluation_results/ablations/openrouter_gemma4_31b_sonnet
SRC=evaluation_results/ablations/openrouter_gemma4_31b
ARMS="ctrl_chunk_tok ctrl_chunk_sum ctrl_chunk_dedup ctrl_keo_norel ctrl_keo_shuf_s13 textchunk mst_on_freq"
for arm in $ARMS; do
  mkdir -p "$DEST/$arm"
  if [[ ! -f "$DEST/$arm/answers.json" ]]; then
    cp -f "$SRC/$arm/answers.json" "$DEST/$arm/answers.json"
    cp -f "$SRC/$arm/answers_meta.json" "$DEST/$arm/answers_meta.json"
  fi
  if "$PYTHON" judge_skip.py "$DEST/$arm" >/dev/null 2>&1; then
    echo "SKIP complete $arm"
    continue
  fi
  echo "JUDGE $arm workers=$KEO_JUDGE_WORKERS $(date)"
  set +e
  "$PYTHON" run_evaluation.py \
    --questions-file "$QUESTIONS" \
    --answers-file "$DEST/$arm/answers.json" \
    --output-dir "$DEST/$arm" \
    --final
  ec=$?
  set -e
  echo "DONE $arm exit=$ec $(date)"
done
echo "=== ALL DONE $(date) ==="
