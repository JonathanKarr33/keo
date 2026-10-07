#!/usr/bin/env bash
# Sequential Phi-4 arm judging at KEO_JUDGE_WORKERS=1 (parallel Claude was Jetsam-killed).
# Assumes answers are already on disk under evaluation_results/ablations/phi4/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
REPO="$(cd "$ROOT/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export KEO_ALLOW_MISSING_NLP=1
export KEO_JUDGE_WORKERS="${KEO_JUDGE_WORKERS:-1}"
MODEL="mlx-community/phi-4-4bit"
QUESTIONS="output/aviation_sensemaking_questions.json"
ARM_ROOT="evaluation_results/ablations/phi4"
LOG="$REPO/judge_phi4_seq.log"

ARMS=(mst_on_freq textchunk ctrl_chunk_tok ctrl_chunk_sum ctrl_chunk_dedup ctrl_keo_norel ctrl_keo_shuf_s13)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }

# Free answerer RAM if still loaded
pkill -f "mlx_lm server --model " 2>/dev/null || true
sleep 2

log "start sequential judge workers=$KEO_JUDGE_WORKERS"
for name in "${ARMS[@]}"; do
  dir="${ARM_ROOT}/${name}"
  if skip_line="$("$PYTHON" judge_skip.py "$dir" --requested-model "$MODEL")"; then
    log "SKIP $name (${skip_line#*$'\t'})"
    continue
  fi
  log "JUDGE $name"
  if "$PYTHON" -u run_evaluation.py \
      --questions-file "$QUESTIONS" \
      --answers-file "${dir}/answers.json" \
      --output-dir "$dir" >>"$LOG" 2>&1; then
    log "OK $name"
  else
    log "FAIL $name exit=$?"
  fi
done

log "paired_stats"
mkdir -p evaluation_results/stats
"$PYTHON" paired_stats.py \
  --arm "mst_on_freq=${ARM_ROOT}/mst_on_freq#keo" \
  --arm "textchunk=${ARM_ROOT}/textchunk#textchunkrag" \
  --arm "ctrl_chunk_tok=${ARM_ROOT}/ctrl_chunk_tok#chunk_tok" \
  --arm "ctrl_chunk_sum=${ARM_ROOT}/ctrl_chunk_sum#chunk_sum" \
  --arm "ctrl_chunk_dedup=${ARM_ROOT}/ctrl_chunk_dedup#chunk_dedup" \
  --arm "ctrl_keo_norel=${ARM_ROOT}/ctrl_keo_norel#keo_norel" \
  --arm "ctrl_keo_shuf_s13=${ARM_ROOT}/ctrl_keo_shuf_s13#keo_shuf" \
  --reference mst_on_freq \
  --out "evaluation_results/stats/paired_phi4_$(date +%Y%m%d_%H%M%S).json" \
  --latex evaluation_results/stats/paired_phi4_table.tex >>"$LOG" 2>&1 \
  || log "paired_stats failed"
log "DONE"
