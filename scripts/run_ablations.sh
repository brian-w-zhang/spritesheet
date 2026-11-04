#!/usr/bin/env bash
# C2, C3, C5: short, uncompiled runs of the main architecture (8k steps each), evaluated with the same metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG='{"C0":128,"C1":64,"C2":48,"C3":32}'
S=${STEPS:-8000}
run () { name=$1; shift
  [ -f artifacts/model/abl_$name/model.pt ] || uv run python -m spritesheet.train --tier 1 --steps $S --lr 3e-3 --cfg "$CFG" --out artifacts/model/abl_$name "$@" > artifacts/model_abl_$name.log 2>&1
  uv run python -m spritesheet.evaluate --model artifacts/model/abl_$name --out artifacts/metrics/abl_$name.json --n 200 \
     --sets heldout_combo,heldout_phrase,unseen_words,pixelgpt_test > artifacts/eval_abl_$name.log 2>&1
  echo "done $name"
}
run base
run bow --bow
run nocliptable --learn-wordvec
run pg5k --pixelgpt-n 5000
run pg10k --pixelgpt-n 10000
run emojionly --mix '{"emoji_var":0.6,"emoji_plain":0.4}'
