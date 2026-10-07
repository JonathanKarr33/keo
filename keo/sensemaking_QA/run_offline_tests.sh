#!/usr/bin/env bash
# Run every offline suite (no API calls, no judged runs, safe to run anywhere).
#
#   bash run_offline_tests.sh                            # all suites
#   bash run_offline_tests.sh test_judge_parse_offline.py  # just one
#   PYTHON=/path/to/python bash run_offline_tests.sh
#
# Exits non-zero if any suite fails, so it is usable as a pre-sweep gate and in CI.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Same resolution as run_ablation_matrix.sh: macOS ships no bare `python`, and system
# python3 lacks networkx/sentence-transformers.
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

# These suites never compute BLEU/ROUGE, so the NLP guard in evaluator.py is not relevant
# to them; a judged run still enforces it.
export KEO_ALLOW_MISSING_NLP=1

ALL_SUITES=(
  test_judge_parse_offline.py
  test_judge_guards_offline.py
  test_paired_stats.py
  test_risk_retrieval_offline.py
  test_risk_arm_offline.py
  test_controls_offline.py
  test_ablation_offline.py
  test_generation_offline.py
  test_generation_resume_offline.py
  test_openrouter_offline.py
  test_judge_parallel_offline.py
  test_shell_safety_offline.py
  test_judge_skip_offline.py
  test_dispersion_offline.py
  test_pick_kg_annotation.py
  test_annotator_sheets_offline.py
)

# Named suites run just those (handy while iterating); no arguments runs everything.
if (( $# )); then
  SUITES=("$@")
else
  SUITES=("${ALL_SUITES[@]}")
fi

echo "Python: $PYTHON"
echo "Suites: ${#SUITES[@]}"
echo

FAILED=()
SUMMARY=()
for suite in "${SUITES[@]}"; do
  if [[ ! -f "$suite" ]]; then
    # A renamed or deleted suite must not vanish from the run silently.
    echo "MISSING  $suite"
    FAILED+=("$suite (missing)")
    SUMMARY+=("MISSING  $suite")
    continue
  fi

  log="$(mktemp)"
  if "$PYTHON" "$suite" >"$log" 2>&1; then
    # run_suite() refuses to print this line for an empty check list, so its absence means
    # the suite exited 0 without reporting what it tested.
    line="$(grep -oE 'OK: .*' "$log" | tail -1)"
    if [[ -z "$line" ]]; then
      echo "NO COUNT $suite  (exited 0 but reported no check count)"
      sed 's/^/    /' "$log" | tail -20
      FAILED+=("$suite (no count)")
      SUMMARY+=("NO COUNT $suite")
    else
      printf 'PASS     %-32s %s\n' "$suite" "$line"
      SUMMARY+=("PASS     $suite  $line")
    fi
  else
    echo "FAIL     $suite"
    sed 's/^/    /' "$log" | tail -30
    FAILED+=("$suite")
    SUMMARY+=("FAIL     $suite")
  fi
  rm -f "$log"
done

echo
echo "===================== summary ====================="
for line in ${SUMMARY[@]+"${SUMMARY[@]}"}; do echo "$line"; done
echo

if (( ${#FAILED[@]} )); then
  echo "FAILED (${#FAILED[@]}/${#SUITES[@]}): ${FAILED[*]}"
  exit 1
fi
echo "All ${#SUITES[@]} offline suites passed."
