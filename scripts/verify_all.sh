#!/usr/bin/env bash
# One command for every gate. Exit non-zero if any gate that RAN failed; skipped gates print NOT RUN.
#   scripts/verify_all.sh [--model artifacts/model/v32] [--prompts 200] [--seeds 2] [--sheets-prompts 20]
set -u
cd "$(dirname "$0")/.."
MODEL=artifacts/model/v32; PROMPTS=200; SEEDS=2; SHEETS_PROMPTS=20
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2;;
    --prompts) PROMPTS=$2; shift 2;;
    --seeds) SEEDS=$2; shift 2;;
    --sheets-prompts) SHEETS_PROMPTS=$2; shift 2;;
    *) echo "unknown arg $1"; exit 64;;
  esac
done
FAIL=0
# clean checkout: install the pinned HyperFormula oracle (tools/package-lock.json) if missing
if [ ! -d tools/node_modules/hyperformula ]; then echo "=== installing tools/ node deps (npm ci)"; npm ci --prefix tools --silent || { echo "npm ci failed"; exit 1; }; fi
run() { local name=$1; shift; echo "=== $name: $*"; "$@"; local rc=$?
        if [ $rc -eq 0 ]; then echo "--- $name PASS"; elif [ $rc -eq 2 ]; then echo "--- $name NOT RUN"; else echo "--- $name FAIL (rc=$rc)"; FAIL=1; fi; }

run "pytest (A1 tokenizer, A2 op parity, model twin, data)" uv run pytest -q
run "A3 end-to-end local (HyperFormula + LibreOffice)" uv run python scripts/verify_e2e.py --model "$MODEL" --prompts "$PROMPTS" --seeds "$SEEDS"
if [ -f "$HOME/.config/spritesheet/sa-key.json" ]; then
  run "A4 Google Sheets" uv run python scripts/verify_sheets.py --model "$MODEL" --prompts "$SHEETS_PROMPTS"
else
  echo "=== A4 Google Sheets: NOT RUN (no service-account key)"
fi
if [ -f scripts/check_claims.py ]; then
  run "H claims check" uv run python scripts/check_claims.py
else
  echo "=== H claims check: NOT RUN (scripts/check_claims.py missing)"
fi
exit $FAIL
