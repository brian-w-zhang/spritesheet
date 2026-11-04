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
- **Ablations** (8k steps each, same architecture as main, uncompiled; fg-pixel accuracy given the target's style, n=300 per set) [ablations.json]:

| Run | held-out combo | held-out phrasing | unseen words | PixelGPT test |
| --- | --- | --- | --- | --- |
| all sources (PixelGPT 18k) | 0.548 | 0.527 | 0.154 | 0.125 |
| bag-of-words encoder (C2) | 0.551 | 0.506 | 0.141 | 0.108 |
| learned embeddings, no CLIP table (C3) | 0.604 | 0.415 | 0.060 | 0.143 |
| PixelGPT 5k (C5) | 0.546 | 0.529 | 0.159 | 0.104 |
| PixelGPT 10k (C5) | 0.545 | 0.540 | 0.142 | 0.128 |
| emoji only (C5 mixture) | 0.558 | 0.542 | 0.162 | 0.064 |

  **What we learned:**
  1. **The CLIP word table is what buys generalisation to new wording.** Without it, held-out phrasings drop 0.527 → 0.415 and unseen words 0.154 → 0.060. But learned embeddings fit *trained* vocabulary better (held-out combos 0.604 vs 0.548). A real trade-off; we keep the table because "type anything" is the point.
  2. **Attention barely beats bag-of-words here** (0.548 vs 0.551 on combos; 0.527 vs 0.506 on phrasings). Our captions are short, and binding cases are rare in the test sets. Honest result: the encoder's attention is cheap insurance, not the source of quality.
  3. **More PixelGPT data doesn't help at this capacity.** 5k / 10k / 18k give PixelGPT-test fg 0.104 / 0.128 / 0.125. **So buying the 216k edition is not worth it for v32.** Revisit for bigger models (v48/v64).
  4. **Emoji-only is as good or slightly better on every emoji set** (combo 0.558, phrasing 0.542), and only loses on PixelGPT prompts. The pixel-art data mostly competes for capacity.

## Bugs found by the verification gates (slide: "checks, not trust")
- **D-30:** the fixed-width word split broke when the first words summed to ≥100 characters (found by A1).
- **D-31:** one LCG step mapped every seed < ~1000 to style 1 (u ≈ seed·2.2e-5). Fixed with two steps.
- **D-32:** HyperFormula treats `\` in string literals as an escape, so a single cell broke the whole sheet.

## Phase 4 · Demo surface in real Google Sheets (02:30–02:45)
- The first sheet render used SPARKLINE bars, which leave a gap around every cell. **Switched to conditional formatting**: each display cell holds a class id with its digits hidden, and one rule per palette colour gives a seamless 32×32 image. Background changed to light grey so white sprites stay visible.
- Tried a "Generate" checkbox; Brian preferred plain Enter, so it was removed. Kept a "Showing: …" done-indicator and a 9-stage bar.
- An interrupted push left the sheet half-uploaded (tabs created, formatting missing): an important lesson for the demo. A full push takes ~2.5 min; always let it finish.
- Verified in Chrome: the apple renders seamlessly. Render matched the reference exactly.

## Phase 5 · v2: few-step masked discrete diffusion (03:00–)
**Why:** v1's one-shot decoder averages possible drawings into mud (Brian's "fox" example). Masked diffusion (MaskGIT-style) commits confident pixels step by step, so later steps stay consistent with earlier ones. Chosen over continuous few-step diffusion (needs distillation), GANs (unstable overnight), and retrieval (can't invent). A **patch codebook** was ruled out first: 4×4 patches reconstruct only 74–84% of foreground pixels even with 4,096 codes [codebook_check.json].

- **v2a** (232,312 params, 6 steps, emoji-only, 40k steps): **worse than v1.** Many prompts came out blank or tiny.
  - Diagnosis 1, the sampler: the most confident pixels are background, so deterministic top-confidence commits fill the canvas with background and squeeze the object out. Blank rate was 26% [v2_sampler_sweep_a.json].
  - Sampler sweep (no retraining): "foreground-first" removes blanks but bloats objects (fg area 1.5–1.7× target). Seeded temperature 1.0 removes blanks (0.3%) but gives confetti. No variant beat v1.
  - Hybrid test: v1 keeps its most confident 50–85% of pixels and v2 refines the rest. Foreground accuracy 0.546–0.601 vs v1 alone 0.607 [v2_hybrid_test.json]. **So the v2a denoiser itself is the weak link.**
  - Diagnosis 2, the architecture: v2a received the prompt only as per-channel FiLM shifts, which carry *what* but not *where*. From an empty canvas, a conv net with no positional signal can't place a shape. v1 has a dense prompt→4×4 layout layer, and v2a didn't.
- **v2b**: adds a **prompt→8×8×C3 spatial prior** (dense layer, computed once per prompt and shared by every step, so cheap in the sheet). Training (60k steps). *Pending.*

### Correction (03:50): the "drop PixelGPT" call was based on a biased measurement
Brian asked whether PixelGPT was really useless. It wasn't shown to be. Two problems with the evidence:
1. On PixelGPT test captions (free text like "ghost with a snail shell") we scored **pixel match against one specific sprite**. For an open-ended caption, many drawings are valid, so every model scores ≈0.10–0.14, and the metric can't see improvement.
2. The other evidence came from **emoji-only test sets**, which naturally favour emoji-only training.

**Fix:** generalisation is now measured with **CLIP R-precision** (does CLIP pick the image's own prompt out of the whole set?) on (a) 100 novel prompts from neither dataset and (b) PixelGPT test captions, with ground-truth PixelGPT sprites as an upper reference [generalization.json]. v2b was restarted *with* PixelGPT (30%), and a PixelGPT-majority v2c (70%) plus a matching one-shot ablation were added to test "more semantically diverse captions → better generalisation".

**Generalisation results (CLIP R-precision; 100 novel prompts, chance R@1 = 0.01; 300 PixelGPT test captions, chance 0.003)** [generalization.json]

| Model | novel R@1 | novel R@5 | PixelGPT R@1 |
| --- | --- | --- | --- |
| v1 main (40k steps, 30% PixelGPT) | 0.20 | 0.39 | 0.027 |
| 8k all-sources (`abl_base`) | 0.13 | 0.35 | 0.010 |
| 8k emoji-only | 0.14 | 0.29 | 0.007 |
| 8k no CLIP table | 0.11 | 0.27 | 0.023 |
| 8k bag-of-words | 0.10 | 0.36 | 0.007 |
| v2a diffusion (emoji-only) | 0.08 | 0.20 | 0.003 |
| *ground-truth PixelGPT sprites* | – | – | *0.147* |

**Read:** (1) longer training dominates; (2) PixelGPT *helps* novel-prompt generalisation at R@5 (0.35 vs 0.29 emoji-only), which reverses the earlier call; (3) the CLIP word table helps (0.35 vs 0.27); (4) free-text captions are hard even for the real sprites. With n=100, differences under ~0.05 are noise.

### Visual line-up (04:05): metrics vs eyes [figures/lineup_greybg_*.png]
Same 36 prompts (12 trained emoji concepts, 12 novel, 12 PixelGPT-style) × models, seed 1, reviewed by eye:
- **v1 main (40k steps) is the best row in all three groups.** Recognisable: strawberry, tree, mushroom, pizza, owl (silhouette), sword, *sad penguin* (with face), *pickup truck*, *toad in mud*. Blobs: fox, cat, dog, dragon, knight, "cat wearing a hat".
- **The 8k-step ablations are visibly blobbier than v1.** Differences *between* them are subtle and mixed, so small metric gaps between ablations should not be over-claimed.
- **v2a (diffusion) is clearly worst:** fragments, or nothing.
- **Bug found by looking:** all figures and CLIP scores used a near-white background (#f6f4ef), which made white subjects (ghosts, skeletons, snowmen) invisible. Figures and evaluation now use the sheet's light grey (#d8d4dd). CLIP numbers before 04:05 are biased against white subjects for every model, and will be re-scored.
**Lesson for the slides:** metrics alone misled us twice tonight (the PixelGPT call, the white-background bug). Every claim now gets a visual check.

### Latency correction + v2d (04:20)
- Brian measured v1 at **~3–4 s per prompt in the browser**. Our API-polled latency (11–30 s) includes network writes and repeated 1,024-cell reads, so it overstates what a user sees. That implies ~7–9M multiply-adds/s in Sheets, so a 30–60 s budget is ~250–500M per prompt: room for a bigger diffusion model.
- "Training longer" = more optimizer steps (more passes over the data). More data matters only once a model **memorises** (train improves, held-out stalls). Our models **underfit**, so the trainer now logs a **validation loss** (PixelGPT val + held-out emoji combos) every 2k steps.
- **v2d** (launched 04:20): E=32, C1=32, C2=64, C3=192 + layout map, T=6. ~63M multiply-adds/step ≈ 380M/prompt (est. 45–55 s in the browser). 120k steps, 50% PixelGPT.
- **PixelGPT padding (Brian's question):** padding 24→32 is the correct, lossless choice. A 1.33× resize gives uneven pixel widths (nearest) or blur and speckle (bilinear). The side effect is a size gap vs emoji (24 px vs ~30 px content). Next run: **random placement** of PixelGPT sprites within the 32×32 canvas (augmentation, still pixel-exact).

### v2b / v2c results (04:55–05:10) [figures/lineup_v2bc_*.png, lineup_pixelstyle.png, metrics/v2_sampler_sweep_*.json]
- **v2b** (layout map, 30% PixelGPT, 680,824 params, 60k steps): the layout fix worked, with **no more blanks**, sensible colours and some structure (strawberry with leaves, tree, car colours). But objects are **shrunken**.
- **v2c** (layout map, **70% PixelGPT**): the most *sprite-like* row. Dark outlines, an outlined penguin, a lighthouse (yellow tower on a red base), a small boat for "pirate ship", standing figures for "knight" and "skeleton warrior". Also often too small.
- **Shrinking = the background-first commit again (milder).** Sampler sweep on v2b/v2c: plain confidence → objects ~0.5–0.6× target area; foreground-first → 1.4–1.6×; **background confidence × β** (sheet: `IF(am=0, β·conf, conf)`) gives area ≈ target at **β ≈ 0.6**. On pixel metrics, diffusion still trails v1 (fg 0.47–0.53 vs 0.61), and by eye the corrected sampler gives v1-like blobs or squares. Not a clear win.
- **Honest verdict so far:** at sheet-deployable sizes and ~1 h of training, masked diffusion adds *structure* (outlines, coherent regions) but not better *shapes* than v1. Combined with the Sheets upload ceiling (D-47), the deployable bet for the best model is a **bigger, much longer-trained v1** (v1-XL: C 256/128/64/32, 150k steps, 50% PixelGPT, random placement; launched 05:10). Diffusion stays as the experimental "watch it form" demo.

### Diffusion in the real sheet (05:40)
- **Inert upload (D-48) works.** The full-width T=4 v2 random build (830k formulas): text upload 111 s, tab-by-tab activation 1,803 s. All 3 test prompts matched the reference pixel for pixel; API-measured latency 36–98 s (first consistent read) [spike_inert_upload.json].
- **v2c_T4_b06**: v2c's weights re-calibrated for **4 steps** with background penalty **β = 0.6** (no retraining; tau [0.713, 0.600, 0.492]). By eye: full-size objects again; mushroom (red cap, white stem), slim tree, strawberry, pink potion bottle, lighthouse, knight figure read well; treasure chest (black square), house and sword are misses. **Comparable to v1, not clearly better.** It is the v2 demo model (with filmstrip) [figures/lineup_v2cT4_*.png].

### Trained v2 live in Google Sheets (07:00) [metrics/sheets_v2c_T4_b06.json]
- `v2c_T4_b06` pushed to the diffusion sheet with the inert upload: 836,342 formulas, 3,631 s to upload and activate. **5/5 prompts** (mushroom, red apple, lighthouse, potion bottle, blue fox) matched the numpy reference pixel for pixel. API-measured time to the first consistent read: 119–163 s. The filmstrip (step 1 → 4) renders under the image.

### v1-XL (08:00): longer + wider wins, and becomes the final v1 [figures/lineup_xl_*.png, metrics/quality.json, quality_v1small.json, generalization_v1xl.json]
- **Setup:** C 256/128/64/32 (736,224 params, 2.4× v1), 150k steps at batch 256 (3.75× v1's steps), 50% PixelGPT with random placement (`--pg-shift`). 3 h on the laptop GPU.
- **By eye:** the clearest gain so far on trained concepts. Fox (ears, white muzzle), cat, rocket (body + window + flame), mushroom (spotted cap), penguin (black/white with face) are all sharper than v1. Novel prompts are mixed (castle and potion bottle better; dragon is a green blob; robot dog worse).
- **Numbers (old v1 → v1-XL):** held-out combos, fg pixel acc given style 0.625 → 0.676; train sample 0.578 → 0.639; hue agreement on PixelGPT captions 0.367 → 0.460; CLIP score up on every set (held-out combos 0.267 → 0.274). Novel-prompt CLIP R@5 0.40 → 0.41 (a tie at n=100).
- **Cost:** 220,806 formulas (vs 169,526), still under the ~250k normal-upload ceiling.
- **Decision:** v1-XL becomes `artifacts/model/v32` (deployed v1). The old v1 is kept as `artifacts/model/v1_small`, with its metrics in `*_v1small.json`. A3 and A4 are re-run on the new model.
- **Lesson:** training length and width were the biggest levers again. Changing the architecture (diffusion) helped less than training the simple model longer and wider.

### v3 begins (09:00–09:40): fixing mode averaging [decisions D-51; artifacts/runs/]
- **Why:** even v1-XL draws the per-pixel *average* of all training sprites for a prompt (fuzzy edges, mixed colours). Diffusion avoids this, but at sheet size it was worse and 4× slower (v2). The plan is to keep the one-shot network and make it **commit to one drawing first**.
- **Data:** PixelGPT 216K, native 24×24 (no padding), one style, no emoji. Captions average 3.3 words: specific noun phrases ("rusty shovel blade", "pouncing leopard"); 130k distinct captions; colour words in only 3.1%, so colour adjectives come from recolour augmentation [data/build/v3/report.json, gitignored].
- **Idea (multiple choice learning):** 16 learned variants; each sprite trains only the variant that already fits it best. The seed picks a variant at run time, so a different seed gives a different but whole frog. In the sheet this costs the same formulas as v1's style.
- **Prior art read:** "The Five-Dollar Model" (feed-forward text → pixel art, 8×8 to 16×16, ≤ 900 images). Its outputs look clean mostly because it memorises tiny datasets. It doesn't address averaging. We took its word-vector noise and its "wider kernels matter" result.
- **Baseline on the new 216K test captions (v1-XL):** novel-prompt CLIP R@5 0.41; PixelGPT-test R@5 0.057; colour-adjective accuracy 0.658; mean per-pixel confidence 0.73 [metrics/v3_eval_v1ref.json].
- **Ops lesson:** two trainings in parallel on the laptop GPU exhausted unified memory, and both stalled at step 1 (archived as `*_aborted_*`). Runs now go one at a time through a queue.

### v3_var16 result (11:45): variants split colour, not structure [metrics/v3_eval_r1.json; figures/v3_lineup_r1_*.png, v3_seeds_r1_v3_var16.png; runs/v3_var16/]
- **Setup:** 16 variants, winner-takes-all, 606,593 params, 60k steps (2.0 h). Validation loss: best variant 1.20, random seed 1.78. That gap is expected once variants commit to specific drawings, so cross-entropy no longer measures quality.
- **Better than v1-XL:** colour-adjective accuracy 0.896 (v1-XL 0.658); less speckle (0.015 vs 0.028 on novel prompts); fewer fragments (1.2 vs 1.4 components). Seeds give genuinely different whole drawings (a white, black or tan cat; a green, purple or red dragon).
- **Worse:** by eye, mostly flat silhouettes with little interior detail (white blobs for cat, owl, skeleton; a sword is a dot; the castle is blank). Novel-prompt R@5 0.35 (v1-XL 0.41). PixelGPT-test R@5 0.067 (v1-XL 0.057).
- **Diagnosis:** 16 variants carry only 4 bits of "which drawing". Winner-takes-all spent them on the cheapest split (colour, silhouette), so interior detail still averages within a variant. The 216K captions are specific and rarely repeat, so they are far less predictable than emoji (7 clean renders per concept): training fg accuracy 0.36 vs ~0.57 for v1-XL.
- **Next:** the control (v3_base, no variants) to measure what the variants bought. Then give the latent more bits: a small grid of discrete codes (a VQ-style tokenizer at 3×3) with a cheap prior over 9 codes, so the decoder gets enough information to draw detail.
