#!/usr/bin/env bash
# Fetch raw image sources into data/raw/ (gitignored). Idempotent. Sparse/shallow clones only.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
sparse () { # repo dir branch paths...
  local repo=$1 dir=$2 br=$3; shift 3
  if [ -d "$dir/.git" ]; then echo "have $dir"; return; fi
  git clone -q --depth 1 --filter=blob:none --sparse -b "$br" "https://github.com/$repo.git" "$dir"
  (cd "$dir" && git sparse-checkout set --no-cone "$@" && git rev-parse HEAD > ../"$dir".commit)
  echo "fetched $dir"
}
sparse googlefonts/noto-emoji noto-emoji main '/2D/png/128/*' '/LICENSE' '/README.md'
sparse jdecked/twemoji twemoji main '/assets/72x72/*' '/LICENSE' '/LICENSE-GRAPHICS'
sparse microsoft/fluentui-emoji fluentui-emoji main '/assets/**' '/LICENSE'
sparse hfg-gmuend/openmoji openmoji master '/color/72x72/*' '/data/openmoji.json' '/LICENSE.txt' '/README.md'
sparse game-icons/icons game-icons master '/**/*.svg' '/README.md' '/license.txt'
sparse tabler/tabler-icons tabler-icons main '/icons/**' '/LICENSE' '/tags.json'
mkdir -p cldr
[ -f cldr/annotations-en.json ] || curl -sfL -o cldr/annotations-en.json https://raw.githubusercontent.com/unicode-org/cldr-json/main/cldr-json/cldr-annotations-full/annotations/en/annotations.json
[ -f cldr/annotationsDerived-en.json ] || curl -sfL -o cldr/annotationsDerived-en.json https://raw.githubusercontent.com/unicode-org/cldr-json/main/cldr-json/cldr-annotations-derived-full/annotationsDerived/en/annotations.json
[ -f cldr/LICENSE ] || curl -sfL -o cldr/LICENSE https://raw.githubusercontent.com/unicode-org/cldr-json/main/LICENSE
du -sh ./* | sort -h
# PixelGPT 24x24 free 20K edition (HF card: CC BY 4.0; bundled LICENSE.txt is a placeholder, see docs/03)
[ -d pixelgpt-24x24-20k/data ] || uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('unstonio/pixelgpt-24x24-20k', repo_type='dataset', local_dir='pixelgpt-24x24-20k')"
# CLIP ViT-B/32 (MIT) for the offline word table + evaluation
[ -f ../models/clip-vit-base-patch32/pytorch_model.bin ] || uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', local_dir='../models/clip-vit-base-patch32', allow_patterns=['*.json','*.txt','pytorch_model.bin','*.md'])"
