#!/bin/bash
# usage: tools/vlm/run_judges.sh name1 name2 ...   (sequential; resumable; detached-safe)
cd "$(dirname "$0")"
for n in "$@"; do
  [ -f ../../artifacts/eval/e1/$n/vlm.json ] && continue
  uv run python vlm_judge.py --name $n >> ../../artifacts/runs/logs/vlm_e1_$n.txt 2>&1
done
