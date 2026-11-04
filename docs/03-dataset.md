# 03 · Data: sources, licences, normalisation, synthetic variations

Sourcing pass done 2026-11-04. Raw files live in `data/raw/` (gitignored), fetched by `scripts/fetch_data.sh` (the PixelGPT download is in the same section below). Each source's commit hash is recorded in `data/raw/<name>.commit`. **Re-confirm each licence from the files inside `data/raw/` before publishing anything derived.** `ATTRIBUTION.md` comes later; see TODO.

The Nouns dataset was dropped (D-10): it is NFT art, and the scope is now "make anything".

## Sources

| Tier | Source | Licence (where checked) | What's in `data/raw/` | Role |
| --- | --- | --- | --- | --- |
| 1 | **Noto Emoji** (Google) | Images **Apache 2.0**; fonts OFL (repo README "License" section, read via GitHub API) | `noto-emoji/2D/png/128/`, **3,787** PNGs `emoji_u<cp>.png` | concept base, style "noto" |
| 1 | **Twemoji** (jdecked fork, original creators) | Graphics **CC-BY 4.0** (`LICENSE-GRAPHICS`); code MIT. A mention in the README or About is accepted as attribution | `twemoji/assets/72x72/`, **4,009** PNGs `<cp>.png` | style "twemoji" |
| 1 | **Fluent Emoji** (Microsoft) | **MIT** (repo licence) | `fluentui-emoji/assets/<Name>/{3D,Color,Flat,High Contrast}/` + `metadata.json` (cldr name, keywords, unicode, group); **1,595** concepts, 1,285 with a 3D PNG. Color/Flat/HC are SVG (rasterise with `resvg-py`) | 4 styles per concept + keywords |
| 1 | **OpenMoji** | **CC BY-SA 4.0** (`LICENSE.txt`, FAQ). *Share-alike:* derived data/weights would need CC BY-SA | `openmoji/color/72x72/`, **4,566** PNGs `<CP>.png` + `data/openmoji.json` | style "openmoji"; **include, but keep it switchable** (`--no-openmoji`) |
| 1 | **Unicode CLDR annotations** | Unicode License v3 (`cldr/LICENSE`) | `cldr/annotations-en.json`, `annotationsDerived-en.json` | official names + keyword synonyms per emoji |
| 2 | **PixelGPT 24×24, free 20K edition** (unstonio) | HF card says **CC BY 4.0**, but the bundled `LICENSE.txt` is an unfilled placeholder ("A license has not yet been selected"). **Treat as CC BY 4.0 per the card; note the discrepancy;** Brian may email the author | `pixelgpt-24x24-20k/data/{train,validation,test}-*.parquet` (18,000 / 1,000 / 1,000); fields `image, caption, family, category, tags, palette, tile_bg, quality_score, sha256` | **native pixel art** with free-text captions; 19 families / 133 categories |
| 2 | **game-icons.net** | **CC BY 3.0** (game-icons.net/about; the GitHub repo shows no SPDX), credit each author | `game-icons/<author>/<name>.svg`, **4,239** SVGs (white on black) | extra concepts; colourised in code |
| 2 | **Tabler Icons** | **MIT** | `tabler-icons/icons/{outline,filled}/*.svg` (**6,257**) + tags | extra concepts (line icons); lowest priority |
| — | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | MIT | `data/models/clip-vit-base-patch32/` (smoke-tested) | offline: word table + evaluation; **never in the sheet** |

| 3 | **GameTileNet** (Chen et al., AIIDE 2025; `RimiChen/2024-GameTileNet`) | Repo **CC BY-SA 4.0**; each source tileset keeps its OpenGameArt licence (CC0 / CC-BY 3.0 / CC BY-SA 3.0), listed in `Attributions-self.txt` | `gametilenet/` (≈2.7k PNGs, 32×32 native, semantic labels in `DataAndAnnotations/`) | **native 32×32 human-made pixel art** with object labels. Optional; **use only CC0/CC-BY tilesets** unless Brian accepts share-alike |

### Evaluated and not used (yet)
- **`bghira/free-to-use-pixelart`** (HF, card says MIT): ~7k user uploads scraped from PixilArt, so the art belongs to individual uploaders; many are large illustrations; images are URLs to download; content-warning fields. Provenance and licence are murky, so **not used**.
- **PixelGPT full edition** (itch.io, $24, 216,838 sprites). The licence summary allows ML training and commercial use of trained models but **forbids redistributing the dataset**. If Brian buys it, it stays local in `data/raw/` (gitignored), and the public repo never contains its images. The page discloses that it is *"a synthetic, AI-assisted pixel-art dataset"*. The free 20K is the same kind of data, so the full set is an optional upgrade for v48/v64.
- **Kaggle `ebrahimelgazar/pixel-art`** (89k; Kaggle metadata read via API 2026-11-04, contradicting claims that it is 64–128 px): 16×16 **JPEG** (lossy, bad for pixel art), integer class labels only (no captions). The page says it was "collected from an online game" and labels it Apache 2.0, so provenance is doubtful. **Not used.**
- **`tkarr/sprite_caption_dataset`** (SpriteX32's training data): Pokémon sprites with Pokédex text and no licence. **Not used.**
- Kenney (CC0) and DawnLike (CC-BY 4.0) 16×16 tilesets: good art, but numbered tiles with no names. Optional style/test material only.
- Real-photo sets (PD12M, CommonCatalog): a model that fits in a sheet would produce colour blobs. Possibly an MP3 experiment slide.

## Joining sources into "concepts"
- **Emoji concept id = Unicode codepoint sequence** (lowercase hex joined by `-`, with `fe0f` stripped). Noto `emoji_u1f34e.png`, Twemoji `1f34e.png`, OpenMoji `1F34E.png` and Fluent `metadata.json: unicode` all map to `1f34e` ("red apple"). One concept then has **up to 7 drawings** (Noto, Twemoji, OpenMoji, Fluent 3D, Color, Flat, High Contrast). This is how the model learns "apple" rather than memorising one picture.
- Exclude flags, skin-tone variants (`1f3fb`–`1f3ff`), keycaps, ZWJ family sequences, and tiny symbols. Use the Fluent `group` field and CLDR for the category.
- **Text per concept:** the CLDR name, CLDR keywords, Fluent keywords, and the OpenMoji annotation and tags. Build a reviewed synonym table `data/concepts.json`.
- **PixelGPT:** each sprite is its own sample with its own caption (often creative: "ghost with a snail shell", "sad penguin beak", "noir drinking cup"). The category gives a coarse concept label. Captions are noisy (e.g. "hollow chisel" filed under bells_and_chimes), so use `quality_score` for filtering.
- **game-icons:** the concept is the file name (`crossed-swords`), with no colour information; colour is assigned by the synthesis step.

## Normalising different native sizes (16 / 24 / 32 / 72 / 128 px) to one canvas

**Rule: never resample native pixel art by a non-integer factor.** Nearest-neighbour 24→32 makes uneven pixel widths, and bilinear blurs. Instead:

| Source kind | To 32×32 | To 48×48 | To 64×64 |
| --- | --- | --- | --- |
| Native pixel art 24×24 (PixelGPT) | **pad** 1:1 into the canvas (4 px margin), no resampling | **2× nearest** = 48 exactly | 2× nearest (48) padded to 64 |
| Native 32×32 (GameTileNet) | as-is | 1.5× is non-integer → pad 32 into 48 *or* skip | 2× = 64 |
| Native 16×16 (Kenney/DawnLike, if used) | 2× nearest = 32 | 3× = 48 | 4× = 64 |
| High-res emoji / SVG icons | **render natively at the target size**: crop to the alpha bbox, area-downsample (`Image.BOX`) to fit `n−2`, alpha threshold 0.5, palette-quantise, optional 1 px dark outline (prototype: `scripts/preview_pixelate.py`) | same, rendered at 48 | same, at 64 |

General rule: integer scale `k = floor((n − margin)/s)`, nearest-neighbour, then centre-pad.

**Pad with *transparent*, not zero/black.** PixelGPT's `tile_bg` (≈ near-black) must first be masked to transparent; otherwise padding produces a dark frame that the model learns as content.

**Backgrounds:** convert every source to *foreground + transparent*. Emoji and icons use their alpha; PixelGPT uses `pixel == tile_bg` → transparent. Backgrounds are then a controlled attribute (see below), not an accident of the source. The model's output classes are **palette colours + 1 background class**.

**Shared palette:** fit one global palette per resolution with k-means over all normalised foreground pixels, K ≈ 32–48 (log the choice; K drives the logit-layer cell count). Snap every training image to it. The palette is part of the model config and of the sheet (`Palette` tab).

Preview figures: `artifacts/figures/preview_pixelate.png` (8 concepts × Noto/Twemoji/Fluent-3D/OpenMoji at 32 and 48) and `artifacts/figures/preview_pixelgpt.png` (PixelGPT at native 24, padded 32, 2× 48). Both look clean.

## Synthetic variations (each sample has an exact target image)
Applied by `src/spritesheet/synth.py` on the normalised foreground. Each transform also edits the caption. Unspecified attributes take a **canonical default** (original colours, centred, count 1, no background), so the target is deterministic given the caption **plus a style id**.

| Control | Caption words (vary phrasing) | Implementation |
| --- | --- | --- |
| colour | red, blue, golden, pastel, dark, neon, monochrome… | hue/value remap of the dominant region, snapped to the palette |
| style / source | "noto style", "3d", "flat", "outlined", "pixel art" (PixelGPT) | source choice; 1 px outline on/off |
| direction | facing left/right, upside down, sideways | flip / rotate 90 / rotate 180 |
| size | tiny, small, big, huge | scale (integer steps for pixel art; re-render for emoji) |
| position | in the top-left / centre / bottom … (3×3) | placement |
| count | two, three, a pair of, a row of | tiling |
| background | on grass, at night, on a blue background, in a circle, in a box | generated backgrounds and frames |
| **relations** (stretch, high value) | "a cat **next to** a cup", "a bird **on top of** a box" | compose two concepts |
| **accessories** (stretch) | "a dog **wearing a hat**", "a robot **on fire**" | overlay an emoji (hat, crown, fire, sparkles) at an anchor |

**Style is ambiguous** when the prompt doesn't name it (an apple could be drawn by 7 artists), so a **style latent is sampled from the seed** (exact LCG; see doc 04). This avoids training a regressor on a blurry average of the 7 drawings.

## Splits (fixed before training; hashes saved in `data/splits.json`)
- **Held-out (concept × attribute) combinations:** e.g. "blue banana" never seen, though bananas and blue were. This is the main generalisation test.
- **Held-out synonyms/phrasings:** keywords and templates never used in training captions.
- **Unseen words:** words in the CLIP word table that never appear in any training caption (e.g. "puppy" if only "dog" is trained).
- **PixelGPT:** use its own train/val/test splits; it keeps near-duplicates within one split.
- **Held-out drawings:** one style of a concept held out, to test style generalisation.

## Scale tiers (capacity is fixed by what fits in Sheets; doc 07)
- **v32 tier 0:** ~300 curated emoji concepts (food, animals, objects, nature, vehicles) × all styles, plus colour/direction/size/position. This is the shippable MVP.
- **v32 tier 1:** + all usable emoji concepts (~1.5–2.5k) + PixelGPT 18k + count/background.
- **v32 tier 2:** + game-icons, + relations/accessories.
- **v48 / v64:** separate models and separate spreadsheets, starting from the tier the v32 results support.
