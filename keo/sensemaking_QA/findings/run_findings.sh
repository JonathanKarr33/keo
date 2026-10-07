#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 mine_cool_findings.py
python3 emit_latex_findings.py
echo "Open findings/out/cool_findings.md"
