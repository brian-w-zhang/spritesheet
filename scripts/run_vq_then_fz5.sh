#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
while kill -0 8058 2>/dev/null; do sleep 60; done
echo "=== $(date +%H:%M) start v3_vq"
uv run python -u -m spritesheet.train_vq --steps 60000 --out artifacts/model/v3_vq --note "VQ latent: 3x3 grid of 64 codes (54 bits) from a training-only encoder; sheet decoder starts from code embeddings; autoregressive 9-step prior over codes samples them from the seed" > artifacts/runs/logs/v3_vq.txt 2>&1
echo "=== $(date +%H:%M) done v3_vq"
scripts/run_v3_queue.sh 'v3_fz5:--steps 60000 --S 16 --cfg "{\"C0\":256,\"C1\":128,\"C2\":64,\"C3\":32,\"film_z\":true,\"k1\":5,\"k2\":5}" --note "v3 variants + variant steers every layer (FiLM on [c, variant]) + 5x5 kernels in blocks 1-2"'
