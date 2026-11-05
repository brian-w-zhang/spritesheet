#!/bin/bash
# Queue 11 (GPU, after v3p_b): guidance distillation into the deployable causal size (dm128 L4 = 431,836 formulas).
#  1. guided samples from the big teacher v3p_b (g=1 with the causal unconditional prior v3p_u, t=0.8);
#  2. causal student initialised from v3p_a, trained on 75% guided teacher samples + 25% real data.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
until [ -f artifacts/model/v3p_b/config.json ]; do sleep 30; done
uv run python scripts/gen_teacher_codes.py --teacher artifacts/model/v3p_b --unet artifacts/model/v3p_u --guidance 1 \
  --tau 0.8 --n 120000 --batch 512 --out data/build/v3/kd_v3p_b_g1.npz > artifacts/runs/logs/gen_kd_v3p_b_g1.txt 2>&1
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 25000 --lr 3e-4 --p-short 0.3 \
  --init artifacts/model/v3p_a/model.pt --kd data/build/v3/kd_v3p_b_g1.npz --kd-mix 0.75 \
  --cfg '{"dm":128,"H":4,"mlp":256,"layers":4}' --out artifacts/model/v3p_c \
  --note "deployable causal (dm128 L4) from v3p_a, guidance-distilled: 75% guided samples of v3p_b (g=1, t=0.8), 25% real" \
  > artifacts/runs/logs/v3p_c.txt 2>&1
