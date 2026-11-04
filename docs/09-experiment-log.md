# 09 · Experiment log: how we got to the final model

A running, slide-ready record of every iteration: what we tried, the numbers, and why we changed course.
Numbers come from the files cited in brackets under `artifacts/`. Newest entries are at the bottom. Times are EST, Nov 3–4 2026.

---

## Phase 0 · Choosing what to build (Nov 3, evening)

| Step | What happened | Outcome |
| --- | --- | --- |
| Idea | "A text-to-image model that lives in a spreadsheet." Constraint-driven Advanced Technology project, inspired by GPT-2 in Excel and Freak in the Sheets | Named **SpriteSheet**; compiler named **Pixcell** |
| Dataset v1 | Nouns DAO traits (CC0, exact compositor, so ground truth exists for any prompt) | **Dropped** (D-10): it's NFT art, and we wanted "make anything" |
| Dataset search | Evaluated real-photo sets (PD12M, CommonCatalog), Kaggle pixel-art, SpriteX32's data, PixilArt scrape, emoji sets, icon sets, GameTileNet, PixelGPT | Photos → colour blobs at sheet size; Kaggle = 16×16 JPEG of unclear origin; SpriteX32 data = Pokémon, unlicensed; **chose licensed emoji (7 drawings per concept) + PixelGPT 20K** |
| Resolution | 16 / 24 / 32 px sources → one canvas | **32×32 first; never resample pixel art fractionally** (pad 24→32); v48/v64 later as separate models |

**Key insight for the slides:** the spreadsheet, not the dataset, caps quality. SpriteX32 uses ~75M parameters and 50 steps on a GPU; a sheet can hold a few hundred thousand weights and recalculate once per prompt.

## Phase 1 · De-risking before any training (Nov 4, 00:30–00:50)

| Spike | Result | Decision |
| --- | --- | --- |
| Local formula engines | HyperFormula and LibreOffice both match numpy on a test MLP; LibreOffice silently skips recalculation unless forced | Two local oracles plus a stale-value check (D-08) |
| Google Sheets auth | gcloud's OAuth client is blocked for Sheets scopes | Service account + one sheet shared by a human |
| **R1: real Sheets, random-weight v32** | 127k formulas: pixels **matched the reference exactly**; prompt→pixels 11–18 s [spikes.json] | Model budget set: spend channels on the low-resolution layers (D-21) |
| Word-table width | Median synonym rank after PCA: d=32 → 33, d=48 → 11, **d=64 → 4** (e.g. puppy→dog 233 → 37; azure→blue 15 → 1) [wordtable_dim.json] | **d = 64** |

## Phase 2 · Bring-up models (tier 0 = 300 emoji concepts × 6 styles = 1,800 bases)

| Run | Params | Steps | Train fg-pixel acc (stream) | Held-out combo fg-pixel acc* | What we saw | Next |
| --- | --- | --- | --- | --- | --- | --- |
| `v32_t0` (C 64/48/32/24) | 165,216 | 4,000 | 0.497 | 0.427 | right silhouettes and colours, blurry detail; colour/size/position/count/background follow the prompt | more capacity + steps |
| `v32_t0b` (C 96/64/48/32) | 253,840 | 15,000 | 0.678 | 0.506 | sharper; unseen-word prompts still mostly fail (fg acc 0.065) | move to all sources, wider low-res layers |

\*Measured with a *seed-sampled* style, which can differ from the target's style, so this understates decoder quality. From then on the evaluation also reports accuracy **given the target's style** (decoder quality) alongside sampled-style accuracy (what the sheet does). [quality_t0.json, quality_t0b.json]

## Phase 3 · Main model and ablations (in progress)

- **`v32_main`** (tier 1: 1,409 concepts, 8,358 emoji bases + 18,000 PixelGPT; C 128/64/48/32; **309,648 params**; 40,000 steps, 52 min on the M4). Mixture: 45% emoji variations, 30% PixelGPT, 25% plain emoji. Train-stream fg-pixel acc ended at 0.544. [quality.json, n=300 per set]

| Set | fg-pixel acc (given style) | fg-pixel acc (sampled style) | CLIP score ÷ target's | CLIP concept top-1 (ours / target) | attribute agreement |
| --- | --- | --- | --- | --- | --- |
| train_sample | 0.578 | 0.495 | 0.915 | – / – | hue 0.61, size 0.91, pos 0.96, count 0.70, bg 1.00 |
| heldout_combo | 0.625 | 0.525 | 0.922 | 0.000 / 0.103 | hue 0.85 |
| heldout_phrase | 0.555 | 0.453 | 0.909 | 0.020 / 0.180 | hue 0.67, size 0.89, pos 0.95, count 0.70, bg 1.00 |
| unseen_words | 0.145 | 0.073 | 0.936 | 0.000 / 0.203 | hue 0.31 |
| underspecified | 0.618 | 0.320 | 0.908 | 0.043 / 0.187 | hue 0.54 |
| pixelgpt_test | 0.127 | 0.125 | 0.846 | – / – | hue 0.37 |

  **Read:** attributes are the strength. Size, position, count and background follow the prompt; colour on held-out (concept, colour) pairs agrees 82% of the time; named styles are honoured. Concept *detail* is the weakness. CLIP rarely names the concept from our 32×32 output, though it also struggles with the ground-truth targets (top-1 ≈ 0.08–0.25 among 1,409 names). **Unseen words fail** (fg acc ≈ 0.07): the frozen CLIP table alone doesn't carry a held-out keyword to the right concept. PixelGPT's free-text captions are barely learned (fg 0.12): at sheet-sized capacity, the 18k one-off creative captions mostly add noise. The C5 ablation tests exactly this.
  - Eval bug fixed here: evaluation had defaulted to tier-0 test sets; it now follows the model's tier.
- **`v32_long`**: same architecture, 100,000 steps. Launched 01:51 as the final candidate. *Pending.*
- **Ablations** (8k steps each, same architecture, uncompiled): `base`, `bow` (bag-of-words encoder), `nocliptable` (learned embeddings instead of the CLIP table), `pg5k` / `pg10k` (data scaling), `emojionly` (mixture). *Results pending.*

## Bugs found by the verification gates (slide: "checks, not trust")
- **D-30:** the fixed-width word split broke when the first words summed to ≥100 characters (found by A1).
- **D-31:** one LCG step mapped every seed < ~1000 to style 1 (u ≈ seed·2.2e-5). Fixed with two steps.
- **D-32:** HyperFormula treats `\` in string literals as an escape, so a single cell broke the whole sheet.
