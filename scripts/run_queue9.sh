#!/bin/bash
# Queue 9 (GPU, after queue 8): causal unconditional prior (guidance for teachers), then a stronger causal teacher.
# Neither is deployed: they produce guided distillation data for the grouped student (guidance distillation).
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
until [ -f artifacts/model/v3pg_u/config.json ]; do sleep 30; done
uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 12000 \
  --cfg '{"dm":96,"H":4,"mlp":192,"layers":3}' --out artifacts/model/v3p_u \
  --note "causal unconditional prior (raster order) for classifier-free guidance of the causal teachers" > artifacts/runs/logs/v3p_u.txt 2>&1
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 30000 --lr 5e-4 --p-short 0.3 \
  --cfg '{"dm":192,"H":6,"mlp":576,"layers":6}' --out artifacts/model/v3p_b \
  --note "stronger causal teacher (dm192 L6), distillation only, never deployed" > artifacts/runs/logs/v3p_b.txt 2>&1
