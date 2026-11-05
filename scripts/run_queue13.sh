#!/bin/bash
# Queue 13 (GPU): same-size improvements for the deployed guided model (D-55).
#  1. v3p_a2 = v3p_a fine-tuned with head-noun caption dropout (p_short 0.3; v3p_a predates it): short prompts.
#  2. v3p_u48b = the dm48 unconditional chain trained 20k more steps (its val NLL was still falling).
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 20000 --lr 3e-4 --p-short 0.3 \
  --init artifacts/model/v3p_a/model.pt --cfg '{"dm":128,"H":4,"mlp":256,"layers":4}' --out artifacts/model/v3p_a2 \
  --note "v3p_a fine-tuned 20k steps with head-noun caption dropout (p_short 0.3), same sheet size" > artifacts/runs/logs/v3p_a2.txt 2>&1
uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 20000 --lr 3e-4 \
  --init artifacts/model/v3p_u48/model.pt --cfg '{"dm":48,"H":2,"mlp":96,"layers":2}' --out artifacts/model/v3p_u48b \
  --note "v3p_u48 trained 20k more steps (lr 3e-4)" > artifacts/runs/logs/v3p_u48b.txt 2>&1
