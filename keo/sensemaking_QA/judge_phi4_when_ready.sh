#!/usr/bin/env bash
# Wait for the phi4 controls generation to finish, free the answerer GPU/RAM,
# then judge every arm. Needed because Claude workers + Phi-4 (~8 GB) get the
# judge process Jetsam-killed at ~18 MB free.
#
#   bash judge_phi4_when_ready.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
REPO="$(cd "$ROOT/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
ARM_ROOT="evaluation_results/ablations/phi4"
MODEL="mlx-community/phi-4-4bit"
QUESTIONS="output/aviation_sensemaking_questions.json"
LOG="$REPO/judge_phi4_when_ready.log"
export KEO_ALLOW_MISSING_NLP=1
export KEO_JUDGE_WORKERS="${KEO_JUDGE_WORKERS:-4}"

ARMS=(mst_on_freq textchunk ctrl_chunk_tok ctrl_chunk_sum ctrl_chunk_dedup ctrl_keo_norel ctrl_keo_shuf_s13)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" | tee -a "$LOG"; }

arm_ready() {
  local dir="$1"
  [[ -f "$dir/answers.json" && -f "$dir/answers_meta.json" ]] || return 1
  "$PYTHON" - "$dir/answers.json" "$dir/answers_meta.json" <<'PY'
import json, sys
a=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
n_meta=int(m.get("n_questions") or 0); n_ans=len(a) if isinstance(a, list) else 0
sys.exit(0 if n_meta>0 and n_ans>=n_meta else 1)
PY
}

log "waiting for all ${#ARMS[@]} phi4 arms (PYTHON=$PYTHON workers=$KEO_JUDGE_WORKERS)"
while true; do
  pending=0
  for name in "${ARMS[@]}"; do
    if ! arm_ready "$ARM_ROOT/$name"; then
      pending=$((pending + 1))
    fi
  done
  log "pending_generation=$pending"
  [[ "$pending" -eq 0 ]] && break
  sleep 60
done

log "all arms on disk — stopping local answerer to free RAM for Claude"
# serve_local_model.sh has no --stop; kill the mlx-lm process holding the weights.
pkill -f "mlx_lm server --model .*phi-4" 2>/dev/null || true
pkill -f "mlx_lm server --model " 2>/dev/null || true
sleep 5
# drop page cache pressure a little
vm_stat | awk '/Pages free/ {printf "free_pages %s\n", $3}' | tee -a "$LOG" || true

for name in "${ARMS[@]}"; do
  dir="$ARM_ROOT/$name"
  if skip_line="$("$PYTHON" judge_skip.py "$dir" --requested-model "$MODEL")"; then
    log "SKIP $name (${skip_line#*$'\t'})"
    continue
  fi
  log "JUDGING $name"
  if "$PYTHON" -u run_evaluation.py \
      --questions-file "$QUESTIONS" \
      --answers-file "$dir/answers.json" \
      --output-dir "$dir" >>"$LOG" 2>&1; then
    log "OK $name"
  else
    log "FAIL $name (see $LOG) — will continue"
  fi
done

log "running paired_stats"
mkdir -p evaluation_results/stats
ARM_ARGS=()
ARM_ARGS+=(--arm "mst_on_freq=${ARM_ROOT}/mst_on_freq#keo")
ARM_ARGS+=(--arm "textchunk=${ARM_ROOT}/textchunk#textchunkrag")
ARM_ARGS+=(--arm "ctrl_chunk_tok=${ARM_ROOT}/ctrl_chunk_tok#chunk_tok")
ARM_ARGS+=(--arm "ctrl_chunk_sum=${ARM_ROOT}/ctrl_chunk_sum#chunk_sum")
ARM_ARGS+=(--arm "ctrl_chunk_dedup=${ARM_ROOT}/ctrl_chunk_dedup#chunk_dedup")
ARM_ARGS+=(--arm "ctrl_keo_norel=${ARM_ROOT}/ctrl_keo_norel#keo_norel")
ARM_ARGS+=(--arm "ctrl_keo_shuf_s13=${ARM_ROOT}/ctrl_keo_shuf_s13#keo_shuf")
"$PYTHON" paired_stats.py \
  ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
  --reference mst_on_freq \
  --out "evaluation_results/stats/paired_phi4_$(date +%Y%m%d_%H%M%S).json" \
  --latex evaluation_results/stats/paired_phi4_table.tex \
  >>"$LOG" 2>&1 || log "paired_stats failed"

log "done"
