#!/usr/bin/env bash
# Orchestrator for KEO revision ablation matrix.
# Usage:
#   bash run_ablation_matrix.sh --smoke
#   bash run_ablation_matrix.sh --full --kg-size 500
#   bash run_ablation_matrix.sh --controls --kg-path <labeled.gml>
#     Refuses a graph with no relation labels or a single weight, unless
#     KEO_ALLOW_UNLABELED_KG=1. The 100-record GPT-4o extract is that graph.
#   bash run_ablation_matrix.sh --controls --judge --final --kg-path <labeled.gml>
#     Sonnet-scores every control arm this invocation generates. Do not pool with Haiku.
#   bash run_ablation_matrix.sh --risk --judge --stats  # risk-sensitive selection (Pivot C)
#   CONTROL_SEEDS=13,29,71 bash run_ablation_matrix.sh --controls
#   PROVIDER=local EMB=local bash run_ablation_matrix.sh --smoke
#   PROVIDER=huggingface API_PROVIDER=featherless-ai ANSWER_MODEL=microsoft/phi-4 \
#     bash run_ablation_matrix.sh --full
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Resolve an interpreter that actually has the deps. The repo venv is preferred:
# macOS ships no bare `python`, and system python3 lacks networkx/sentence-transformers.
# Override with PYTHON=/path/to/python.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
    PYTHON="$ROOT/../../.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    echo "No Python interpreter found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi
echo "Python: $PYTHON"

SMOKE=0
FULL=0
CONTROLS=0
RISK=0
JUDGE=0
STATS=0
JUDGE_FINAL=0
CONTROL_SEEDS="${CONTROL_SEEDS:-13}"
RISK_BUDGET="${RISK_BUDGET:-1200}"
RISK_FLOOR="${RISK_FLOOR:-0.25}"
REFERENCE_ARM="${REFERENCE_ARM:-mst_on_freq}"
KG_SIZE=100
SAMPLE=3
# Defaults target the Mac-local path, because it is the one that needs no credentials.
# Qwen/Qwen3.8-27B via huggingface/nebius used to be the default and was doubly dead:
# the model ID was never served, and Hugging Face has since dropped nebius as an inference
# provider entirely (their router now offers featherless-ai, together, groq, deepinfra and
# others). Both failures looked like a model-name typo, which cost real debugging time.
# phi-4-4bit: Microsoft Phi-4 (14B) at 4-bit, ~8.3 GB. Gemma-4 E4B was deleted when this
# replaced it. Serve it from .venv-mlx (py3.12, mlx-lm 0.31.3); the repo venv is 3.9.
ANSWER_MODEL="${ANSWER_MODEL:-mlx-community/phi-4-4bit}"
PROVIDER="${PROVIDER:-local}"
API_PROVIDER="${API_PROVIDER:-together}"
BASE_URL="${BASE_URL:-${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8000/v1}}"
EMB="${EMB:-local}"
EMB_MODEL="${EMB_MODEL:-}"
QUESTIONS="${QUESTIONS:-output/aviation_sensemaking_questions.json}"
KG_PATH=""
FORCE=0
# Empty = legacy layout evaluation_results/ablations/<arm>. A tag writes
# evaluation_results/ablations/<tag>/<arm> so a new answerer cannot overwrite
# an earlier generator's answers and then inherit its judge scores.
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --full) FULL=1; SAMPLE=""; shift ;;
    --controls) CONTROLS=1; shift ;;
    --risk) RISK=1; shift ;;
    --risk-budget) RISK_BUDGET="$2"; shift 2 ;;
    --risk-floor) RISK_FLOOR="$2"; shift 2 ;;
    --control-seeds) CONTROL_SEEDS="$2"; shift 2 ;;
    --judge) JUDGE=1; shift ;;
    --stats) STATS=1; shift ;;
    --final) JUDGE_FINAL=1; shift ;;
    --reference) REFERENCE_ARM="$2"; shift 2 ;;
    --kg-size) KG_SIZE="$2"; shift 2 ;;
    --answer-model) ANSWER_MODEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --API-provider) API_PROVIDER="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --embedding-backend) EMB="$2"; shift 2 ;;
    --embedding-model) EMB_MODEL="$2"; shift 2 ;;
    --kg-path) KG_PATH="$2"; shift 2 ;;
    --sample-size) SAMPLE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$KG_PATH" ]]; then
  CANDIDATE="../kg/output/kg_llm/gpt4o_with_nodes_batches/${KG_SIZE}"
  # Pick the graph with the most nodes, not whatever `find` lists first. These directories
  # are full of 23-byte stubs ("graph [ directed 1 ]") left behind by earlier *_fixed
  # scripts, and a stub used to win: every graph arm then ran against an empty KG.
  if [[ -d "$CANDIDATE" ]]; then
    KG_PATH="$("$PYTHON" pick_kg.py "$CANDIDATE" 2>/dev/null || true)"
  fi
  if [[ -z "$KG_PATH" ]]; then
    KG_PATH="../kg/output/knowledge_graph.gml"
  fi
fi

# Refuse to start on an empty graph. Without this the graph arms silently produced null
# answers, which is indistinguishable from a method that simply performed badly.
if ! "$PYTHON" pick_kg.py --check "$KG_PATH"; then
  echo "Aborting sweep: $KG_PATH has no nodes. Pass --kg-path with a real graph." >&2
  exit 1
fi

# Mechanism controls named keo_norel and "frequency MST" are vacuous on a graph that
# stores neither relation labels nor edge weights (the 100-record GPT-4o extract).
# Smoke runs stay allowed so a plumbing check can still use whatever graph is local.
# KEO_ALLOW_UNLABELED_KG=1 keeps the old file on purpose.
if [[ "$SMOKE" -eq 0 ]] && { [[ "$CONTROLS" -eq 1 ]] || [[ "$FULL" -eq 1 ]]; }; then
  if [[ "${KEO_ALLOW_UNLABELED_KG:-0}" != "1" ]]; then
    if ! "$PYTHON" pick_kg.py --check --require-annotated "$KG_PATH"; then
      echo "Aborting sweep: mechanism controls need a labeled, weighted graph." >&2
      echo "  labeled graph: ../kg/output/kg_llm/gs100_total500/gemma3_4b_it/kg_gemma3_4b_it.gml" >&2
      echo "  override: KEO_ALLOW_UNLABELED_KG=1 (the relation arm will not test labels)" >&2
      exit 1
    fi
  fi
fi

# MODEL= is an alias for ANSWER_MODEL on the local path. The old Qwen3.8-27B / Qwen3-30B
# fallbacks are gone: neither was ever served, and both are far too large for 18 GB anyway.
if [[ "$PROVIDER" == "local" && -n "${MODEL:-}" ]]; then
  ANSWER_MODEL="$MODEL"
fi

# Validate --run-id before any path is built from it. A slash would write outside
# evaluation_results/ablations/; an empty default keeps the legacy layout so the
# gemma-4 arms already on disk stay where skip-on-exists expects them.
if [[ -n "$RUN_ID" ]]; then
  case "$RUN_ID" in
    *[^A-Za-z0-9_-]*)
      echo "keo: --run-id must be [A-Za-z0-9_-], got: $RUN_ID" >&2
      exit 2
      ;;
  esac
  ARM_ROOT="evaluation_results/ablations/${RUN_ID}"
else
  ARM_ROOT="evaluation_results/ablations"
fi

echo "Using KG: $KG_PATH"
echo "Questions: $QUESTIONS"
echo "Model: $ANSWER_MODEL ($PROVIDER) emb=$EMB"
echo "Arm root: $ARM_ROOT"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE}" ]]; then
  SAMPLE_ARGS=(--sample-size "$SAMPLE")
fi

EXTRA=()
if [[ "$PROVIDER" == "local" ]]; then
  EXTRA+=(--base-url "$BASE_URL")
fi
if [[ "$PROVIDER" == "huggingface" ]]; then
  EXTRA+=(--API-provider "$API_PROVIDER")
fi
if [[ -n "$EMB_MODEL" ]]; then
  EXTRA+=(--embedding-model "$EMB_MODEL")
fi
if [[ "$FORCE" -eq 1 ]]; then
  EXTRA+=(--force)
fi

# Arms produced by this invocation, as "dir_name:method" (bash 3.2: no associative arrays)
ARMS=()

run_one() {
  local name="$1"; shift
  local out="${ARM_ROOT}/${name}/answers.json"
  local method="keo" prev=""
  for a in "$@"; do
    [[ "$prev" == "--method" ]] && method="$a"
    prev="$a"
  done
  ARMS+=("${name}:${method}")
  echo "=== Running $name ==="
  "$PYTHON" generate_ablation_answers.py \
    --question-files "$QUESTIONS" \
    --output-file "$out" \
    --kg-path "$KG_PATH" \
    --answer-model "$ANSWER_MODEL" \
    --provider "$PROVIDER" \
    --embedding-backend "$EMB" \
    --max-answer-tokens 600 \
    ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$@"
}

# The offline suites are the guards that keep unusable answers out of the judge and refuse
# imputed scores. Running them here costs ~30s and is the difference between a broken sweep
# failing now and failing after the API spend. Set SKIP_OFFLINE_TESTS=1 to bypass.
if [[ "${SKIP_OFFLINE_TESTS:-0}" != "1" ]]; then
  echo "=== Offline guard suites ==="
  if ! PYTHON="$PYTHON" bash run_offline_tests.sh; then
    echo "Aborting sweep: offline guards are failing, so results from it would not be trustworthy." >&2
    echo "Bypass with SKIP_OFFLINE_TESTS=1 if you know why." >&2
    exit 1
  fi
  echo
fi

# Confirm the model actually answers before spending the sweep on it. Without this, an
# unserved model ID yields one "Error generating answer: ..." per question per arm, and
# --judge then scores those strings at the 1-5 floor (see test_judge_guards_offline.py).
PREFLIGHT_ARGS=(--answer-model "$ANSWER_MODEL" --provider "$PROVIDER")
if [[ "$PROVIDER" == "huggingface" ]]; then
  PREFLIGHT_ARGS+=(--API-provider "$API_PROVIDER")
fi
if [[ "$PROVIDER" == "local" ]]; then
  PREFLIGHT_ARGS+=(--base-url "$BASE_URL")
fi
# When judging, check the judge in the same pass. Unreadable rubric output is refused
# rather than imputed, so a judge whose format has drifted aborts the run — much better to
# find that out now than after generation has been paid for.
if [[ "$JUDGE" -eq 1 ]]; then
  PREFLIGHT_ARGS+=(--judge-model "${JUDGE_MODEL:-haiku}" --judge-provider "${JUDGE_PROVIDER:-claude_cli}")
fi
if ! "$PYTHON" preflight_model.py ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}; then
  echo "Aborting sweep: the answer model or judge is unusable (nothing would be generated or judged)." >&2
  exit 1
fi

# The keo reference arm every control is compared against.
# Always run: generation is skipped when the arm already exists, so this is free on re-runs.
run_one "mst_on_freq"   --method keo --use-mst  --edge-weight freq
if [[ "$CONTROLS" -eq 0 ]] || [[ "$FULL" -eq 1 ]] || [[ "$SMOKE" -eq 1 ]]; then
  run_one "mst_off_freq"  --method keo --no-mst   --edge-weight freq
fi

if [[ "$FULL" -eq 1 ]] || [[ "$SMOKE" -eq 1 ]]; then
  run_one "mst_on_tfidf"  --method keo --use-mst --edge-weight tfidf
  run_one "mst_on_pmi"    --method keo --use-mst --edge-weight pmi
  run_one "mst_on_query"  --method keo --use-mst --edge-weight query
  run_one "mst_on_rarity" --method keo --use-mst --edge-weight rarity
  run_one "routed"        --method routed --use-mst --edge-weight freq
  run_one "vanilla"       --method vanilla
  run_one "textchunk"     --method textchunk
  run_one "community"     --method community --use-mst --edge-weight freq
  run_one "hybrid"        --method hybrid --use-mst --edge-weight freq
fi

# Risk-sensitive budgeted retrieval (papers/ACL_STRATEGY.md §3 Pivot C).
# Same prompt surface as mst_on_freq; only the edge selection differs. The floor=0 arm
# isolates the reserved-budget mechanism from the severity multiplier alone.
if [[ "$RISK" -eq 1 ]] || [[ "$FULL" -eq 1 ]]; then
  echo "=== Risk-sensitive selection (B=$RISK_BUDGET tokens) ==="
  run_one "risk_floor${RISK_FLOOR}" \
    --method risk --risk-budget-tokens "$RISK_BUDGET" --risk-floor "$RISK_FLOOR"
  run_one "risk_floor0" \
    --method risk --risk-budget-tokens "$RISK_BUDGET" --risk-floor 0
  # Unfilled: shows how little context the objective needs, at the cost of a length gap.
  # Keep --no-risk-fill-budget on the same physical line as run_one: a broken backslash
  # continuation once surfaced as `fill-budget: command not found` (exit 127) after a
  # successful controls sweep and skipped the judge phase.
  run_one "risk_nofill" \
    --method risk --risk-budget-tokens "$RISK_BUDGET" --risk-floor "$RISK_FLOOR" --no-risk-fill-budget
fi

# Mechanism controls: isolate *why* graph context helps (papers/ACL_STRATEGY.md §4).
# Compare each against mst_on_freq (keo) and textchunk with paired tests.
if [[ "$CONTROLS" -eq 1 ]] || [[ "$FULL" -eq 1 ]]; then
  echo "=== Mechanism controls (seeds: $CONTROL_SEEDS) ==="
  run_one "textchunk"        --method textchunk                                  # unmatched chunk arm, to size the confound
  run_one "ctrl_chunk_tok"   --method chunk_tok                                  # length-matched chunks
  run_one "ctrl_chunk_sum"   --method chunk_sum                                  # aggregation, no structure
  run_one "ctrl_chunk_dedup" --method chunk_dedup                                # redundancy removed
  run_one "ctrl_keo_norel"   --method keo_norel --use-mst --edge-weight freq      # connectivity, no relations
  for seed in ${CONTROL_SEEDS//,/ }; do
    run_one "ctrl_keo_shuf_s${seed}" \
      --method keo_shuf --use-mst --edge-weight freq --control-seed "$seed"       # entities, false structure
  done
fi

# A sweep that selected no arms has nothing to judge and nothing to compare, so it must say so
# rather than fall through to an empty judge loop and an empty stats table that look like a clean run.
if [[ ${#ARMS[@]} -eq 0 ]]; then
  echo "keo: no arms selected — check the flag combination (--controls/--ablate/--risk)." >&2
  exit 2
fi

# Judge every arm this run produced, each into its own dir (default judge: Claude Code).
if [[ "$JUDGE" -eq 1 ]]; then
  # macOS ships bash 3.2, where `"${ARR[@]}"` on an *empty* array is an unbound-variable error
  # under `set -u`. Every array below is expanded as ${ARR[@]+"${ARR[@]}"} for that reason. This
  # bit once: JUDGE_ARGS is empty for a plain `--judge` run (only --final/--smoke add to it), so
  # judging died after paying for all the generation, while smoke runs passed.
  JUDGE_ARGS=()
  [[ "$JUDGE_FINAL" -eq 1 ]] && JUDGE_ARGS+=(--final)
  [[ "$SMOKE" -eq 1 ]] && JUDGE_ARGS+=(--judge-grade smoke)
  for entry in ${ARMS[@]+"${ARMS[@]}"}; do
    name="${entry%%:*}"
    dir="${ARM_ROOT}/${name}"
    # Resumable only when the eval records the same answerer as answers_meta.json.
    # A leftover eval from a previous model must not be treated as done.
    if [[ "$FORCE" -eq 0 ]]; then
      # judge_skip.py exits 0 to skip, 2 to judge. The `if` keeps set -e from aborting.
      if skip_line="$("$PYTHON" judge_skip.py "$dir" --requested-model "$ANSWER_MODEL")"; then
        echo "=== Judging $name — SKIP (already judged: ${skip_line#*$'\t'}) ==="
        continue
      fi
    fi
    echo "=== Judging $name ==="
    "$PYTHON" run_evaluation.py \
      --questions-file "$QUESTIONS" \
      --answers-file "${dir}/answers.json" \
      --output-dir "$dir" \
      ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"}
  done
fi

# Paired tests of every arm against the reference (default: mst_on_freq = keo).
if [[ "$STATS" -eq 1 ]]; then
  ARM_ARGS=()
  has_ref=0
  for entry in ${ARMS[@]+"${ARMS[@]}"}; do
    name="${entry%%:*}"
    method="${entry##*:}"
    key="$method"
    [[ "$method" == "textchunk" ]] && key="textchunkrag"
    [[ "$name" == "$REFERENCE_ARM" ]] && has_ref=1
    ARM_ARGS+=(--arm "${name}=${ARM_ROOT}/${name}#${key}")
  done
  # Reference may come from an earlier sweep rather than this invocation
  if [[ "$has_ref" -eq 0 ]]; then
    ARM_ARGS+=(--arm "${REFERENCE_ARM}=${ARM_ROOT}/${REFERENCE_ARM}#keo")
  fi
  mkdir -p evaluation_results/stats
  echo "=== Paired tests vs ${REFERENCE_ARM} ==="
  "$PYTHON" paired_stats.py \
    ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
    --reference "$REFERENCE_ARM" \
    --out evaluation_results/stats/paired_$(date +%Y%m%d_%H%M%S).json \
    --latex evaluation_results/stats/paired_table.tex
fi

echo "Ablation matrix complete. See evaluation_results/ablations/ and papers/EXPERIMENT_MATRIX.md"
if [[ "$STATS" -eq 1 ]]; then
  echo "Paired tests + LaTeX table: evaluation_results/stats/"
fi
