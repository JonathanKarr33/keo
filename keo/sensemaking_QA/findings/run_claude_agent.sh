#!/usr/bin/env bash
# Agent-style DoD / findings synthesis via Claude Code CLI
# (same pattern as cst_agents / policymaking_agents).
#
# Uses Pro/Max subscription (`claude auth login`) — NOT Anthropic API credits.
# Bulk ablation judging: use run_evaluation.py --provider claude_cli instead.
#
# Prereq: `claude` on PATH (or CLAUDE_CMD), logged in.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
QA="$(cd "$ROOT/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/out/claude_agent}"
MODEL="${MODEL:-sonnet}"
mkdir -p "$OUT_DIR"

PROMPT_FILE="${1:-}"
if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$OUT_DIR/dod_prompt.md"
  cat > "$PROMPT_FILE" <<'EOF'
You are drafting KEO paper text for a DoD / accredited-settings audience.

Context files (read if present under findings/out/):
- cool_findings.md
- dod_gap_checklist.json

Write concise Markdown with these sections only:
1. One-sentence DoD gap
2. Why local KG-RAG (not public RAG APIs)
3. 3–5 bullet "why use KEO" claims grounded in findings
4. What remains TBD for camera-ready (smoke / paper-min / judges)

Do not invent numeric scores. Prefer citations to findings filenames.
EOF
fi

export PYTHONPATH="$QA:${PYTHONPATH:-}"
python3 - <<PY
from pathlib import Path
import json, sys
sys.path.insert(0, "$QA")
from claude_cli import (
    run_claude,
    claude_code_version_text,
    ClaudeAuthError,
    ClaudeNotFoundError,
)

prompt = Path("$PROMPT_FILE").read_text()
extra = []
for name in ("cool_findings.md", "dod_gap_checklist.json", "cool_findings.json"):
    p = Path("$ROOT/out") / name
    if p.exists():
        extra.append(f"\\n\\n### FILE {name}\\n" + p.read_text()[:12000])
prompt = prompt + "".join(extra)

try:
    # Agent mode: allow default tools so Claude can reason over attached text
    res = run_claude(
        prompt,
        model="$MODEL",
        tools="default",
        effort="medium",
        permission_mode="dontAsk",
        output_format="text",
        timeout_s=600,
    )
except (ClaudeAuthError, ClaudeNotFoundError) as e:
    print(e, file=sys.stderr)
    sys.exit(2)

out = Path("$OUT_DIR")
out.mkdir(parents=True, exist_ok=True)
(out / "agent_stdout.md").write_text(res.stdout or "")
(out / "agent_stderr.txt").write_text(res.stderr or "")
meta = {
    "exit_code": res.exit_code,
    "model": res.model,
    "claude_code_version": claude_code_version_text(),
    "prompt_file": "$PROMPT_FILE",
    "billing": "claude_pro_max_subscription",
}
(out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
print(f"Wrote {out/'agent_stdout.md'} (exit={res.exit_code})")
print(f"Claude Code: {meta['claude_code_version']} model={meta['model']}")
if res.exit_code != 0:
    sys.exit(res.exit_code)
PY
