#!/bin/bash
# Queue 5 (sequential GPU): after v3p_a, the small unconditional prior for guidance (v3p_u).
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
until [ -f artifacts/model/v3p_a/config.json ]; do sleep 60; done
uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 30000 \
  --cfg '{"dm":64,"H":2,"mlp":128,"layers":2}' --out artifacts/model/v3p_u \
  --note "unconditional PatchGPT prior (no prompt) for classifier-free guidance in the sheet: dm64 L2" > artifacts/runs/logs/v3p_u.txt 2>&1
