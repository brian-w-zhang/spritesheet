#!/bin/bash
# Queue 7 (sequential GPU): after v3pg_a, the distilled grouped student (teacher = causal v3p_a), then the guidance prior.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
until [ -f artifacts/model/v3pg_a/config.json ] && [ -f data/build/v3/kd_v3p_a.npz ]; do sleep 60; done
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 60000 --p-short 0.3 \
  --kd data/build/v3/kd_v3p_a.npz --kd-mix 0.75 \
  --cfg '{"dm":96,"H":4,"mlp":192,"layers":2,"G":8}' --out artifacts/model/v3pg_kd \
  --note "grouped PatchGPT (dm96 L2, 8 rounds) distilled from the causal v3p_a: 75% of each batch = teacher samples (t=0.8), 25% real" \
  > artifacts/runs/logs/v3pg_kd.txt 2>&1
uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 20000 \
  --kd data/build/v3/kd_v3p_a.npz --kd-mix 0.75 \
  --cfg '{"dm":48,"H":2,"mlp":96,"layers":2,"G":8}' --out artifacts/model/v3pg_u \
  --note "unconditional grouped prior for guidance (same order and data mix as v3pg_kd), dm48 L2" > artifacts/runs/logs/v3pg_u.txt 2>&1
