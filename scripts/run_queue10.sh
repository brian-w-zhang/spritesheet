#!/bin/bash
# Queue 10 (GPU, sequential), replaces queues 8/9 after v3pg_kd:
#  1. short cosine-schedule student (same distillation data) to compare with v3pg_kd at 15k steps;
#  2. causal unconditional prior (guidance for the teachers); 3. stronger causal teacher dm192 L6.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
until [ -f artifacts/model/v3pg_kd/config.json ]; do sleep 30; done
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 15000 --p-short 0.3 \
  --kd data/build/v3/kd_v3p_a.npz --kd-mix 0.75 \
  --cfg '{"dm":96,"H":4,"mlp":192,"layers":2,"G":8,"sizes":"cosine"}' --out artifacts/model/v3pg_kdcos15k \
  --note "as v3pg_kd but MaskGIT-style cosine round sizes [1,4,6,8,9,12,12,12]; short run (schedule ablation)" \
  > artifacts/runs/logs/v3pg_kdcos15k.txt 2>&1
uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 12000 \
  --cfg '{"dm":96,"H":4,"mlp":192,"layers":3}' --out artifacts/model/v3p_u \
  --note "causal unconditional prior (raster order) for classifier-free guidance of the causal teachers" > artifacts/runs/logs/v3p_u.txt 2>&1
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 30000 --lr 5e-4 --p-short 0.3 \
  --cfg '{"dm":192,"H":6,"mlp":576,"layers":6}' --out artifacts/model/v3p_b \
  --note "stronger causal teacher (dm192 L6), distillation only, never deployed" > artifacts/runs/logs/v3p_b.txt 2>&1
