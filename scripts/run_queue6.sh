#!/bin/bash
# Queue 6 (sequential GPU): after v3p_a, two grouped PatchGPTs at the ~250k-formula budget (D-53) + a guidance prior.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
until [ -f artifacts/model/v3p_a/config.json ]; do sleep 60; done
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 60000 --p-short 0.3 \
  --cfg '{"dm":96,"H":4,"mlp":192,"layers":2,"G":8}' --out artifacts/model/v3pg_a \
  --note "grouped PatchGPT (D-53): 3x3 patches, 8 rounds of 8; dm96 L2 (250k formulas); head-noun caption dropout 0.3" \
  > artifacts/runs/logs/v3pg_a.txt 2>&1
uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 20000 \
  --cfg '{"dm":48,"H":2,"mlp":96,"layers":2,"G":8}' --out artifacts/model/v3pg_u \
  --note "unconditional grouped prior for guidance (3x3, same order), dm48 L2" > artifacts/runs/logs/v3pg_u.txt 2>&1
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p4 --steps 60000 --p-short 0.3 \
  --cfg '{"dm":128,"H":4,"mlp":256,"layers":3,"G":6}' --out artifacts/model/v3pg_b \
  --note "grouped PatchGPT (D-53): 4x4 patches, 6 rounds of 6; dm128 L3 (249k formulas); head-noun caption dropout 0.3" \
  > artifacts/runs/logs/v3pg_b.txt 2>&1
