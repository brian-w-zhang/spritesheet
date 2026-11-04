#!/bin/bash
# Queue 4 (sequential GPU): wait for the critic fine-tune, then a wider guided prior (prior only) on v3_vq6's codes.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
# (the critic run was stopped early; no wait)
uv run python -m spritesheet.train_vq --prior-only --init artifacts/model/v3_vq6/model.pt --cond-drop 0.15 \
  --steps 40000 --lr 1e-3 --cfg '{"C0":256,"C1":128,"C2":64,"C3":32,"G":6,"Kc":256,"e":16,"hp":1024}' \
  --out artifacts/model/v3_vq6_pg --note "prior-only retrain on frozen v3_vq6 codes: hp 256->1024, 15% prompt dropout for classifier-free guidance" \
  > artifacts/runs/logs/v3_vq6_pg.txt 2>&1
