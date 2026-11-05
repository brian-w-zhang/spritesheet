#!/bin/bash
# Queue 6 (sequential GPU): after v3p_a, the grouped PatchGPT (8 rounds in the sheet, D-53) and its unconditional prior.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
until [ -f artifacts/model/v3p_a/config.json ]; do sleep 60; done
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 60000 --p-short 0.3 \
  --cfg '{"dm":96,"H":4,"mlp":192,"layers":3,"G":8}' --out artifacts/model/v3pg_a \
  --note "grouped PatchGPT: 8 rounds of 8 patches (query/context rows, D-53), dm96 L3, head-noun caption dropout 0.3" \
  > artifacts/runs/logs/v3pg_a.txt 2>&1
uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 30000 \
  --cfg '{"dm":48,"H":2,"mlp":96,"layers":2,"G":8}' --out artifacts/model/v3pg_u \
  --note "unconditional grouped prior for guidance (same order), dm48 L2" > artifacts/runs/logs/v3pg_u.txt 2>&1
