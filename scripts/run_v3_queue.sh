#!/usr/bin/env bash
# Sequential v3 experiment queue (one GPU: parallel runs exhausted unified memory). Usage: scripts/run_v3_queue.sh NAME:ARGS ...
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.5}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.4}
mkdir -p artifacts/runs/logs
for spec in "$@"; do
  name=${spec%%:*}; args=${spec#*:}
  echo "=== $(date +%H:%M) start $name"
  eval uv run python -u -m spritesheet.train_v3 $args --out artifacts/model/$name > artifacts/runs/logs/$name.txt 2>&1
  echo "=== $(date +%H:%M) done $name rc=$?"
done
