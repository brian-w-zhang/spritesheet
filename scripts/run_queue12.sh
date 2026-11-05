#!/bin/bash
# Queue 12 (GPU): small causal unconditional priors for in-sheet guidance (D-55): the U chain must be cheap in formulas.
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
for c in 'u48:{"dm":48,"H":2,"mlp":96,"layers":2}' 'u64:{"dm":64,"H":2,"mlp":128,"layers":2}' 'u32:{"dm":32,"H":2,"mlp":64,"layers":2}'; do
  n=${c%%:*}; cfg=${c#*:}
  uv run python -m spritesheet.train_patchgpt --uncond --dict artifacts/model/patchdict_k1024_p3 --steps 20000 \
    --cfg "$cfg" --out artifacts/model/v3p_$n --note "small causal unconditional prior for in-sheet guidance: $cfg" \
    > artifacts/runs/logs/v3p_$n.txt 2>&1
done
