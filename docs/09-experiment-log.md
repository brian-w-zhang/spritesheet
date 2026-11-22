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

### v3_base control (12:35): variants bought diversity, not detail [metrics/v3_eval_r2.json; figures/v3_lineup_r2_*.png; runs/v3_base/]
- v3_base (no variants, same data, arch and steps; 605,498 params): **by eye about the same as v3_var16**, mostly flat blobs, with slightly better numbers (novel R@5 0.41 vs 0.35; colour-adjective accuracy 0.933 vs 0.896; PixelGPT-test R@5 0.077 vs 0.067). Seed diversity 0.0 (deterministic) vs 0.293.
- **Conclusion:** 16 winner-takes-all variants are too few bits. They give seed variety (colour, silhouette) but no structure. The remaining blur is averaging *within* each variant. v1-XL (emoji + PixelGPT, 150k steps) still draws the most detail, because emoji give clean, repeated targets per concept.
- **Next:** v3_vq (3×3 grid of 64 codes, 54 bits, autoregressive prior); started 12:34.

### Better evaluation + v3 in the real sheet (13:00) [metrics/v3_eval_r3.json, parity_sheets_v3_var16.json; figures/v3_panel_r3_*.png; figures/v3_groundtruth_r3.png is local only (real PixelGPT sprites)]
- **Evaluation upgrade** (after Brian asked whether ~30 by-eye prompts are enough): 1,000 held-out PixelGPT test captions, bootstrap 95% CIs, and a 64-prompt × 4-seed panel per model (2 held-out captions from each of the 21 PixelGPT families, plus novel and adjective prompts, ≈256 images). Also a ground-truth strip (real sprite vs each model), kept local because of the licence.
- **Results:**
  - Novel prompts R@5 (n = 100): v3_base 0.41 [0.32, 0.51], v3_var16 0.35 [0.26, 0.45], v1-XL 0.41 [0.31, 0.50]. **A statistical tie;** the earlier 0.35-vs-0.41 "gap" was noise.
  - Held-out PixelGPT captions R@5 (n = 1,000): v3_base 0.027 [0.018, 0.038], v3_var16 0.033 [0.023, 0.044], v1-XL 0.013 [0.007, 0.020]. Real sprites: 0.209.
  - By eye, all models get colour and rough silhouette but no outlines or detail.
- **v3_var16 live in the v3 Google Sheet:** 136,872 formulas; 6/6 prompts pixel-identical to the reference, max rel 9.99e-15, median 17.2 s via the API.
- **v3_vq6 (6×6 × 256 codes) at 5k steps:** reconstructions through the codes are much better than with 3×3 (val fg accuracy 0.76 vs 0.57) and keep the shapes. Prompt-driven samples are still blobs, so the text→codes prior is the bottleneck (NLL 3.55 nats/code vs 5.5 for uniform).
- **Size budget** (Brian: use the room): at 24×24, width costs stored weights and multiply-adds more than formulas. 384/192/96/48 with 5×5 kernels = 186k formulas and 158M multiply-adds (2.7× v1-XL); 512/256/128/64 with 5×5 = 236k and 281M (4.8×). The final v3 trains the winning idea at that size for 150k+ steps.
- **Licence hygiene:** figures and snapshots that show real PixelGPT sprites (ground-truth strips, VQ reconstruction snapshots) are gitignored. Only model outputs are published.

### v3_vq6 result (r4): more structure and variety, still mushy and oversized [metrics/v3_eval_r4.json; figures/v3_lineup_r4_*.png, v3_panel_r4_*.png, v3_seeds_r4_*.png; runs/v3_vq6/]
- **Setup:** 6×6 grid of 256 codes (288 bits), autoregressive MLP prior, 60k steps (2.4 h), 1,945,282 params total (encoder is training-only).
- **Validation:** reconstruction through the codes fg accuracy 0.781, all 256 codes used; prior NLL 3.28 nats/code (uniform 5.5).
- **Numbers vs v3_var16 / v3_base / v1-XL:**
  - Novel R@5: 0.30 [0.21, 0.39], vs 0.35 / 0.41 / 0.41.
  - PixelGPT-test R@5: 0.018 [0.010, 0.027], vs 0.033 / 0.027 / 0.013.
  - Colour-adjective accuracy: **0.971**, vs 0.896 / 0.933 / 0.658.
  - Seed diversity: **0.604**, vs 0.293 / 0.0 / 0.291.
  - Mean pixel confidence: 0.866, vs 0.589 / 0.568 / 0.732.
- **By eye:** the first v3 model with multi-colour interiors: the frog gets a mouth line, the dragon is two-tone, the wizard has a hat shape. Seeds give genuinely different drawings. But sprites fill most of the canvas, and regions are mushy with no outlines. CLIP retrieval does not improve.
- **Diagnosis:**
  - The decoder averages *inside* each 4×4 code cell; even reconstructions lose outlines.
  - The prior is weakly text-conditioned (NLL far above what sharp samples need). Sampled code grids mix several sprites' codes, which gives the oversized, patchy look.
- **Next:** a critic (hinge GAN) fine-tune of the decoder for crisp cells (running). Then a classifier-free-guided prior (`--cond-drop`) so prompts steer the codes harder.

### Critic fine-tune of v3_vq6 (adv weight 0.1): no effect, stopped at 7.75k/15k [runs/v3_vq6_gan_adv0.1_stopped7500/]
- **Setup:** hinge-GAN patch critic with projection conditioning, on straight-through hard samples. The decoder was trained on reconstruction CE + 0.1 × adversarial loss; the encoder and prior were frozen.
- **Result:** the critic learned to separate real from fake easily (d_loss 1.25 → 0.22 by 7.5k steps). But the samples at 2.5k, 5k and 7.5k are visually identical to v3_vq6, and reconstruction CE stayed flat (0.52–0.57).
- **Reading:** at weight 0.1 the adversarial gradient is swamped by reconstruction. More importantly, the blob shapes come from the *sampled codes* (prior), which this run cannot change.
- **Decision:** stopped early to free the GPU for the guided prior. A stronger critic (weight ≥ 1) remains an option if the prior fix lands.

### Wider guided prior (v3_vq6_pg) + guidance sweep (r5): guidance is the first real lever [metrics/v3_eval_r5.json; figures/v3_lineup_r5_*.png, v3_panel_r5_*.png]
- **Setup:** prior-only retrain on v3_vq6's frozen codes, hp 256 → 1024, 15% prompt dropout, 40k steps (49 min). Val prior NLL 3.28 → 3.12 nats/code. The decoder is unchanged (reconstruction fg accuracy 0.781).
- **Novel R@5 / PixelGPT-test R@5 / seed diversity:**
  - no guidance: 0.28 / 0.019 / 0.57
  - g=1: 0.37 / 0.037 / 0.48
  - **g=3: 0.45 [0.35, 0.55] / 0.045 [0.033, 0.059] / 0.41.** Best of any model so far: v1-XL is 0.41 / 0.013, v3_var16 0.35 / 0.033. Clip score 0.258.
  - g=3, t=0.6: about the same as g=3.
- **By eye:** objects become recognisable at g=3: a diagonal sword, a long key, a red-capped mushroom with a stem, a tree with a trunk, a lantern with a flame. Creatures are still mush. The conv decoder's within-cell blur remains.
- **Learned:** a strong, well-conditioned prior plus classifier-free guidance is what moves quality. In the sheet, guidance needs an unconditional prior pass (cost: one more prior).

### Patch dictionary (k-modes): crisp patches with no decoder [artifacts/model/patchdict_k*_p*/patchdict.json]
- **Idea:** replace the conv decoder with a fixed dictionary of hard P×P patches, fit by weighted k-modes on augmented training patches (distance = mismatched pixels; update = per-pixel majority). Output is a lookup, so there's no averaging inside a patch. In the sheet it is INDEX formulas.
- **Validation reconstruction fg accuracy** (exact pixel match; v3_vq6's conv decoder is 0.781 with blur):
  - 4×4 patches: K=1024 0.718, K=4096 0.771;
  - **3×3 patches: K=1024 0.799**, K=4096 0.848;
  - 2×2 patches, K=1024: 0.913.
- **By eye** (local-only images: they show real sprites): 3×3 at K=1024 keeps outlines, faces and fish. Chosen: 64 tokens per sprite.
- **Next:** v3p (PatchGPT), a causal transformer over the 64 patch tokens with the prompt as a 12-row prefix. It is designed to compile as one causal grid of formulas, with Gumbel-max seed noise that is exact in float64. Run v3p_a started.

### v3-P (PatchGPT) training + sheet path (Nov 4, 16:30–18:30) [runs/v3p_a/; metrics/parity_ops_v3p.json; D-52]
- **v3p_a:** dm 128, 4 layers, 4 heads, MLP 256, 810,112 params; batch 256, 60k steps (0.2 s/step on MPS; most of it is the 1024-way output). Val NLL per patch: 7.16 (start) → 2.08 (2.5k) → 1.93 (10k) → 1.89 (22.5k); top-1 patch accuracy 0.57.
- **By eye** (snapshots, seed 1 and 16 frog seeds): from 5k steps the samples are crisp pixel art with dark outlines, not blobs: a black cat face with yellow eyes, an owl face, a clean diagonal sword by 10k. Frog seeds give 16 different drawings, several frog-shaped. Weakness: some samples are fragmented (scattered patches at temperature 1).
- **Sheet path:** spec `reference_p.py` (row by row) + emitter `emit_p.py` (one causal grid). Op parity on a random small model: HyperFormula 4.8e-14, LibreOffice 4.2e-14, codes/pixels exact, with and without guidance.
- **Real Google Sheets, full size (random weights, 431,836 formulas, inert upload):**
  - Upload and activation of all tabs up to `Score` worked (15–25 s per tab).
  - Activating the whole `Score` tab (the step that closes the 64-step loop) made the spreadsheet return 404/503 for >15 min, and the request never applied.
  - Activating `Score` one row at a time works: 17–49 s per row, each roughly one full recalc of the sheet.
  - Measuring a real prompt change once every row is live (in progress).
- **Data finding:** of 653 captions containing "frog", only 296 have "frog" as the head noun ("frog foot imprint" and "frog leg skewer rack" are not frogs). So short prompts map to mixtures of frog-related objects. Added head-noun-preserving caption dropout (`data.v3.shorten`, `--p-short`).

### PatchGPT temperature sweep at 30k steps (r6) [metrics/v3_eval_r6.json; figures/v3_lineup_r6_*.png, v3_panel_r6_*.png]
- **Novel R@5 / PixelGPT-test R@5 / speckle / fg components / seed diversity:**
  - t=1.0: 0.35 / 0.028 / 0.051 / 3.89 / 0.48
  - **t=0.8: 0.38 [0.29, 0.47] / 0.027 / 0.025 / 2.02 / 0.39**
  - t=0.6: 0.25 / 0.030 / 0.009 / 0.88 / 0.27
  - References: v3_vq6_pg@g=3 0.45 / 0.045 / 0.018 / 1.52 / 0.41; v1-XL 0.41 / 0.013 / 0.028 / 1.40 / 0.29.
- **By eye, t=0.8 is the best-looking v3 so far:** crisp outlined pixel art. A sword, a corked red potion, a golden key, a spotted mushroom, a rooted tree, a shield ring, a lantern. A black cat silhouette with ears, a black owl with yellow eyes. t=1.0 scatters patches; t=0.6 collapses many creatures to tiny or blank sprites (the background mode wins).
- **Next:** guidance with the small unconditional prior (queued), head-noun caption dropout, and the full 60k steps.

### Sheets latency vs size for v3-P / v3-PG (random weights, real v3 sheet; logs: artifacts/runs/logs/latency_v3p_random.txt, push_v3pg_random*.txt)
- **API prompt-change latency (first consistent read / stable); every render pixel-identical to the reference:**
  - causal, 64 steps, 431,836 formulas: 128 / 148 s;
  - grouped, 8 rounds, 446,452 formulas: 88–97 / 106–115 s;
  - **grouped, pruned, 8 rounds, ~250k formulas: 56–58 / 66–69 s** (3 prompts);
  - v1-XL for reference: 221k formulas, 30–38 s.
- **Reading:** latency follows formula count, with a penalty for sequential depth (64 > 8 rounds > v1's ~20 tab levels). Browser timing was not measurable reliably tonight: the laptop is loaded with GPU training, and the tab showed "Trying to connect…".
- **Decision:** train two grouped models at ~250k formulas (`v3pg_a` 3×3/dm96/L2; `v3pg_b` 4×4/dm128/L3). If the latency is too slow for the demo, there are ~140k-formula variants (4×4, dm 96/L2 or dm 64/L3).

### v3p_a final (60k steps, causal reference) (r7) [metrics/v3_eval_r7.json; figures/v3_lineup_r7_*.png, v3_seeds_r7_*.png, v3_panel_r7_*.png]
- **Final val NLL** 1.86 nats/patch; top-1 patch accuracy 0.57.
- **t=0.8:** novel R@5 0.38 [0.29, 0.47]; **PixelGPT-test R@5 0.042** (30k checkpoint 0.027; v1-XL 0.013); speckle 0.025; fg components 1.82; colour-adjective accuracy 1.00; seed diversity 0.40. At t=0.9: R@5 0.33 / 0.046, more fragments (3.06).
- **By eye: the best v3 so far.** A standing skeleton figure, a fox, a red dragon with a flame, a black owl with yellow eyes, a wizard in a hat. 8 seeds per prompt give whole, distinct drawings (lying foxes, slimes, wizards). Some creatures (cat) still collapse to blobs on some seeds.
- **Role:** this 64-step causal model is too slow for Sheets (D-53), so it is the quality reference for the deployable grouped models.

### Grouped v3pg_a vs causal v3p_a (r8) [metrics/v3_eval_r8.json; figures/v3_lineup_r8_*.png, v3_seeds_r8_*.png]
- `v3pg_a` (grouped, 8 rounds, dm96 L2, 250k formulas) was stopped at 30k of 60k steps, to free the GPU for distillation. The baseline is the 30k checkpoint (`scripts/ckpt_to_model.py`; run dir has STOPPED.json). Val NLL at 30k is 2.18, against 1.88 for the causal model at the same step.
- **Novel R@5:**

  | Model | Novel R@5 | PixelGPT-test R@5 | Fg components |
  |---|---|---|---|
  | `v3pg_a` t=0.8 | 0.25 [0.17, 0.33] | 0.021 | 2.06 |
  | `v3pg_a` t=0.7 | 0.27 | 0.023 | 1.42 |
  | causal `v3p_a` t=0.8 | 0.38 | 0.042 | 1.82 |

- **By eye:** the grouped samples are incoherent. Patches of different "versions" of the object are mixed: an orange fox with a white patch, a frog over a blue water strip that appears in many prompts.
- **Colours per sprite** [metrics/colour_stats_r10.json]: `v3pg_a` gives 3.94 at t=0.8 and 6.50 at t=1. Real sprites average 3.50. This confirms the mixing: parallel picks within a round disagree.
- **Next:** sequence-level distillation from the causal teacher (`v3pg_kd`, training).

### Sampling tricks on the causal teacher (r9, r10) [metrics/v3_eval_r9.json, v3_eval_r10.json, colour_stats_r10.json; figures/v3_lineup_r9_*.png, v3_lineup_r10_*.png]
- **Flatness is caused by the temperature.** Mean distinct colours per sprite: real test sprites 3.50; `v3p_a` 1.87 at t=0.8, 2.47 at t=0.9, 3.42 at t=1.0. Lowering t removes the rarer multi-colour patches, which leaves flat single-colour silhouettes.
- **Clipped Gumbel noise (`MIN(c, g)`, r9) fails:**
  - With c=1, novel R@5 is 0.21 and fg components 0.55: most creature prompts come out blank.
  - In raster order, the empty patch is the most likely code at every early position. Starting the object is a tail event, so cutting the tail empties the image.
- **Min-p truncation (r10, `patchgpt.pick`)** keeps only codes whose logit is within c of the best one:
  - **Applied to all codes:** blank images (0.9 colours at c=3).
  - **Applied among foreground codes only** (the empty patch is always eligible): the foreground share is right (0.38 against 0.40 for real sprites), but the colours drop to 1.88 at c=4.
  - **Best novel R@5 so far:** 0.46 [0.37, 0.56] at t=1 with c=4. Compare 0.38 at t=0.8 and 0.41 at t=1. The samples are still speckled by eye.
- **Lesson:** the prior is genuinely uncertain. Sampling tricks only trade flat against noisy. Quality has to come from a better prior, so the next step is a stronger teacher plus guidance, distilled into the grouped student. Guidance distillation means the sheet does not need a second (unconditional) chain.

### Distilled grouped student v3pg_kd (r11) [metrics/v3_eval_r11.json, colour_stats_r11.json; figures/v3_lineup_r11_*.png, v3_seeds_r11_*.png]
- **Setup:** 40k steps; each batch is 75% samples from `v3p_a` (t=0.8, `data/build/v3/kd_v3p_a.npz`, 200k samples, 47 min to generate) and 25% real data.
- **Novel R@5:**

  | Model | Novel R@5 |
  |---|---|
  | `v3pg_kd` t=0.8 | **0.32** [0.23, 0.41] |
  | `v3pg_kd` t=1 | 0.27 |
  | `v3pg_kd` t=1, m=4 | 0.27 |
  | `v3pg_a` (no distillation, 30k) | 0.25 |
  | causal teacher `v3p_a` t=0.8 | 0.38 |

- **Colours per sprite at t=0.8:** 3.11, against 3.94 without distillation (real sprites: 3.50). Less mixing.
- **By eye:** more unified shapes than `v3pg_a`, but still blobs without the defining parts. The sword is near-empty, the mushroom has no cap pattern, the owl has no eyes. Clearly behind the causal model.
- **The cosine-schedule student** (`v3pg_kdcos15k`, 15k steps) finished but was **NOT evaluated**: the grouped path was deprioritised (below).
- **Decision:** the final model will be causal. Real-Sheets latency follows formula count more than depth (D-53: 432k causal 128 s vs 446k grouped ~90 s), and the grouped models stay blobbier. Quality comes first (Brian: "not blobs").

### Guidance on the causal model (r12, r13) [metrics/v3_eval_r12.json, v3_eval_r13.json, colour_stats_r12.json; figures/v3_lineup_r12_*.png, v3_lineup_r13_*.png, v3_seeds_r13_*.png]
- **r12, looser truncation at t=1:** colours rise with c (m=5: 2.19; m=6: 2.60; m=8: 3.08). Novel R@5 is 0.38 at m=6 and 0.41 at m=8. Objects at t=1 with m=6 look good by eye (chest, potion, castle, crown). The creatures are still the weak part.
- **r13, classifier-free guidance** with the causal unconditional prior `v3p_u` (dm96 L3, 12k steps):

  | Setting | Novel R@5 | PixelGPT-test R@5 | Fg components |
  |---|---|---|---|
  | **g=1, t=0.8** | **0.51** [0.42, 0.61] (best so far) | **0.078** (2× `v3p_a`) | 3.41 |
  | g=1, t=1, m=6 | 0.50 | 0.073 | — |
  | g=2 | 0.35–0.40 | — | ~9 |

  At g=1, colour-adjective accuracy is 0.97 and speckle 0.042. At g=2 the fragments explode and colour-adjective accuracy is 0.80–0.82.
- **By eye at g=1:** prompts are much more specific. Foxes have ears and white chests, frogs have eyes, owls have faces, knights have helmets. But there are more specks and fragments.
- **Plan (guidance distillation):** a guidance chain in the sheet would add ~200k formulas. Instead, generate guided samples from a stronger teacher (`v3p_b`, dm192 L6) at g=1, t=0.8. Then train a causal student at the deployable size (dm128 L4, 431,836 formulas, initialised from `v3p_a`) on 75% of those (`scripts/run_queue11.sh` → `v3p_c`).
- **Meanwhile:** `v3p_a` (t=0.8) is being pushed to the v3 sheet as the working final model. The Score tab is activated one row per request, now built into `sheets.activate(row_tabs=…)`.

### Trained causal v3p_a on the real v3 sheet [log: artifacts/runs/logs/push_v3p_a.txt]
- **Model:** `v3p_a` at t=0.8 (export `artifacts/model/v3p_a_sheet`), 431,836 formulas, inert upload. Score was activated one row per request, all 64 rows in 2,323 s. The whole push took 2,944 s (153 requests, 76.6 MB).
- **Result:** the "a red potion" seed-1 render read back from Sheets **matches the reference pixel for pixel**.
- **Prompt-change latency:** push_sheets reported 45.1 / 60.3 s, but that number is **not a real prompt change**: the push renders the prompt already in the sheet. Real prompt changes on the guided build take ~110–170 s (next entry).
- **Role:** this is the working fallback on the sheet. Next is the same model plus the in-sheet guidance chain (D-55).

### Small unconditional chain for in-sheet guidance (r14) [metrics/v3_eval_r14.json; figures/v3_lineup_r14_*.png, v3_seeds_r14_*.png]
- **Setup:** U chain `v3p_u48` (causal, dm48 L2, 20k steps, 61,120 formulas) against the r13 prior `v3p_u` (dm96 L3, ~180k formulas), both with `v3p_a` at t=0.8.

  | Setting | Novel R@5 | PixelGPT-test R@5 | Colour adj. | Speckle | Fg components |
  |---|---|---|---|---|---|
  | **g=1, u48** | **0.55** [0.45, 0.64] (best so far) | 0.069 | 0.97 | 0.048 | 4.18 |
  | g=1, u96 | 0.51 | 0.078 | — | 0.042 | 3.41 |
  | g=1.5, u48 | 0.42 | — | 0.88 | — | 7.67 |

- **By eye at g=1 with u48:**
  - Objects are clean and specific: a red-and-white spotted mushroom, a corked red potion in a glass flask, a lantern with a handle, a crown, a sword, a key.
  - Creatures: a fox with ears and a white face, a frog on grass, a wizard with a hat.
  - Remaining flaw: stray specks on some creatures (owl, ghost, skeleton).
  - g=1.5 breaks the drawings into multicolour noise.
- **Decision:** push `v3p_a` + u48 at g=1 to the v3 sheet (492,956 formulas). The guidance weight is now a live cell (Prompt!B4), so it can be tuned in the sheet without a re-push.

### Guidance weight, temperature and U-chain size (r15, r16) [metrics/v3_eval_r15.json, v3_eval_r16.json, colour_stats_r16.json; figures/v3_*_r15_*.png, v3_*_r16_*.png]
- **All with `v3p_a`:**

  | g | t | U chain | Novel R@5 | PixelGPT-test R@5 | Speckle | Fg comp. | Colours |
  |---|---|---|---|---|---|---|---|
  | **0.75** | **0.8** | u48 | **0.61** [0.52, 0.70] | 0.073 | 0.037 | 3.20 | **3.07** |
  | 0.75 | 0.7 | u48 | 0.57 | 0.083 | 0.031 | 2.49 | 2.61 |
  | 0.75 | 0.9 | u48 | 0.53 | — | 0.050 | 4.21 | — |
  | 0.5 | 0.8 | u48 | 0.55 | — | 0.035 | — | — |
  | 1 | 0.8 | u48 | 0.55 (r14) | — | — | — | 3.95 |
  | 1 | 0.8 | u64 | 0.54 | — | — | — | — |
  | 1 | 0.8 | u32 | 0.50 | — | — | 4.91 | — |
  | 1 | 0.7 | u48 | 0.57 | 0.084 | 0.035 | — | — |

  Real test sprites have 3.50 colours.
- **By eye at g=0.75, t=0.8:**
  - Frogs with eyes and mouths, and foxes with ears and white faces, on all 8 seeds.
  - Owls with faces; dragons as green winged shapes.
  - Cats are still mostly white silhouettes, and ghosts have a few specks.
  - t=0.7 is cleaner (penguin-like owls, clean skeletons) but flatter.
- **Decision:**
  - **g=0.75, t=0.8, U = u48.** dm64 does not help; dm32 is worse.
  - g is the live cell Prompt!B4, so it is set on the sheet without a re-push.
  - Next, at the same sheet size: `v3p_a2` (v3p_a fine-tuned with head-noun caption dropout, which v3p_a predates) and `v3p_u48b` (U trained longer).

### Guided model live on the v3 sheet (A4 for v3-P) [metrics/parity_sheets_v3p.json; figures/v3_sheet_readback.png; log: artifacts/runs/logs/push_v3p_a_g1.txt]
- **Push:** `v3p_a` + U chain `v3p_u48`, t=0.8 (export `artifacts/model/v3p_a_g1_sheet`), 492,956 formulas. Inert upload, then row-by-row Score activation: 4,079 s total, 179 requests, 90.4 MB.
- **Check:** guidance set to 0.75 in the live cell Prompt!B4. Then 8 prompt/seed pairs were written, Pixels read back from Google Sheets, and compared with `reference_p.forward` at g=0.75.
- **Result: all 8 match pixel for pixel.** Prompts: potion, frog, fox, mushroom, golden crown, owl, treasure chest, blue slime. This also checks that the guidance cell is really live.
- **Real prompt-change latency:**
  - first consistent read: 110–137 s;
  - stable: 136–167 s (one at 356 s).
- **By eye (read-back figure):** clean, recognisable sprites. A corked potion, a frog with a mouth, a fox, a spotted mushroom, a crown, a treasure chest, a blue slime with drips.

### Same-size fine-tunes (r17) [metrics/v3_eval_r17.json; figures/v3_*_r17_*.png]
- **What changed:**
  - `v3p_a2` = `v3p_a` fine-tuned 20k steps with head-noun caption dropout (p_short 0.3).
  - `v3p_u48b` = u48 trained 20k more steps.
- **Novel R@5 at g=0.75, t=0.8:**

  | Main | U chain | Novel R@5 |
  |---|---|---|
  | **`v3p_a`** | **u48 (deployed)** | **0.61** [0.52, 0.70] |
  | `v3p_a` | u48b | 0.52 |
  | `v3p_a2` | u48b | 0.51 |
  | `v3p_a2` | u48 | 0.47 |

  PixelGPT-test R@5 is 0.068–0.074 for all four.
- **By eye:** mixed. `v3p_a2` draws cats with ears on some seeds, cleaner skeletons and clearer owls, but stranger knights and wizards.
- **Decision:** no clear win, so the deployed model stays. The re-push costs ~70 min.

## Generation bank and visual review (2026-11-05)

**What:** `scripts/gen_bank.py` generated 740 prompts × 8 seeds with both sheet models headless (v3: v3p_a + v3p_u48, g 0.75, t 0.8; v1: v1-XL). Prompts are the most frequent words and two-word phrases per dataset category, plus 30 demo prompts. Parity of the batched twins against the float64 specs: 24/24 pixel-exact for each model. Speed: v3 42 ms per sprite, v1 2.3 ms per sprite, against 2–3 min per prompt on the sheet (`artifacts/metrics/bank_summary.json`).

**Visual review:** five agents judged every row by eye on contact sheets, with real training sprites as reference (local only), using one rubric: good / ok / blob for the best seed. v3 had 170 good, 512 ok and 58 blob; v1 had 22 good, 96 ok and 622 blob (`artifacts/metrics/bank_review.json`). Reviewer strictness varied (8 to 47 good v3 rows per 150). A second pass by the main session cut v3's "good" to a 64-sprite slide shortlist (`artifacts/figures/bank_shortlist_v3.png`). Plain shapes (circle, slate, dune) were dropped as too easy.

**Learned:** v3 is good at simple, well-defined objects: bottles, potions, fruit, candles, keys, helmets, flames, drops. It is weak on animals beyond a silhouette, abstract prompts and actions. v1 usually draws one shape for all 8 seeds, and its good rows are emoji-like shapes (face, heart, star, clock).

## Bank round 2: adjective variants, all three sheet models, full visual review (2026-11-05)

**What:** the bank grew to 7,006 prompts × 8 seeds × {v3, v2, v1-XL}, about 168k sprites. It has 744 base prompts and 6,262 adjective variants.
- **Seen with the noun** (2,982 variants): modifiers that appear before the noun in at least 2 training captions ("poison apple", "locked chest").
- **Common caption adjectives** (1,930 variants): about 90 frequent ones (cracked, ghostly, carved), rotated over 180 nouns.
- **Hand axes:** colour, material, state, size, pose.

Batched twins against the float64 specs: 24/24 pixel-exact for each model. Speed per sprite: v3 44 ms, v2 7 ms, v1 2.3 ms (`artifacts/metrics/bank_summary.json`).

**Seed diversity:** v3 gives 8.0 distinct sprites per 8 seeds. v2 and v1 give about 2.0, and for about 5,000 prompts all 8 seeds give the identical sprite. Their seed only picks one of 7 styles, and the drawing is deterministic given the style.

**Visual review:** 28 agents judged every row and model by eye, using one rubric ("good" needs the adjective visible too). The main session reviewed the 4 new demo rows and re-checked all 807 v3 variant "good" picks, which gave a 459-sprite v3 shortlist (`artifacts/metrics/bank_review.json`).

| Model | good | ok | blob |
|---|---|---|---|
| v3 | 980 | 5,336 | 690 |
| v2 | 5 | 235 | 6,766 |
| v1 | 60 | 451 | 6,495 |

**Learned:**
- v3 shows many attributes: colour, burning, frozen, crystal, ice, poison and lava all show.
- Abstract caption adjectives mostly don't show ("silk whale", "folded octopus"), though the noun is drawn.
- v2 and v1 almost never show an adjective, and usually draw the same sprite for every seed.

**Sheets galleries:** each demo sheet now has a "Gallery" tab right after Prompt (v3: 800 sprites, v1: 136, v2: 160 including "ok"), drawn as cells with prompt/seed/verdict labels. The first version painted cells with 49 conditional-format rules. That was slow to render at 800 sprites, so the tabs now use static fills: about 13 min to upload for v3, but they open instantly.

## MP3 push: E1 evaluation, then a better model (from 2026-11-21)

### E1 baseline: deployed v3 vs real sprites [metrics/e1_v3.json, e1_real.json (local only: real sprites); D-56]
- **Setup:** 187 fixed prompts × seeds 1–8, all scored (no picking). The VLM judge (Qwen3-VL-8B, blind to the prompt) sees seeds 1–4. Chance on 5-way naming is 0.20. "real" = real PixelGPT sprites: the test sprite for test captions, otherwise training sprites whose caption contains the phrase (or the noun).

  | metric | real | v3 (sheet: g 0.75, t 0.8) |
  |---|---|---|
  | VLM naming, all | 0.662 | 0.539 |
  | naming, creatures | 0.733 | 0.578 |
  | naming, objects | 0.625 | 0.708 |
  | naming, test captions | 0.542 | 0.396 |
  | VLM "clear, not blob" | 0.424 | 0.223 |
  | VLM eyes visible (creatures) | 0.259 | 0.047 |
  | CLIP R@5 (all) | 0.396 | 0.443 |
  | colours per sprite | 3.55 | 2.74 |
  | distinct sprites per 8 seeds | 5.55 | 8.00 |
  | near-duplicate of a training sprite (< 5% pixels differ) | — | 0.001 |

- **By eye** (`artifacts/eval/e1/sheets/real_vs_v3_creatures_local.png`, local only): real sprites have outlines, eyes, shading and 3–4 colours. v3's cats and dogs are white silhouettes, owls are flat shapes, and birds are often a yellow stick.
- **Temperature** (`v3_vs_v3t1_vs_v3g0_creatures.png`): at t = 1.0, some owls and frogs get eyes and inner detail, but shapes fall apart. At 0.8 they hold together but are flat. The model is not confident enough to be both detailed and coherent.
- **Learned about the judges:**
  - The VLM numbers separate real from generated sprites the same way the eye does (eyes 0.26 vs 0.05, clear 0.42 vs 0.22).
  - CLIP ranks v3 above the real sprites. It rewards a clean coloured silhouette, so it is not a quality judge for pixel art.
  - Colour accuracy and facing on "real" are meaningless: few real captions contain those phrases, so those real rows are noun-only matches. Facing is at chance for both.
- **Near-copies:** v3 does not copy training sprites (0.1% of its sprites are within 5% of one).

### E1 sampling ablations on v3 [metrics/e1_v3g0.json, e1_v3t1.json]
v3 with the sheet settings (g 0.75, t 0.8) against no guidance (g 0) and t = 1.0:

| metric | v3 | v3 g=0 | v3 t=1.0 |
|---|---|---|---|
| VLM naming, all | 0.539 | 0.421 | 0.505 |
| naming, creatures | 0.578 | 0.379 | 0.672 |
| VLM clear | 0.223 | 0.143 | 0.112 |
| VLM eyes | 0.047 | 0.014 | 0.038 |
| colours per sprite | 2.74 | 1.81 | 4.02 |

- **Guidance helps:** without it, naming drops by 0.12 and the sprites are one-colour blobs (`sheets/v3_vs_v3g0_vs_v3t1_creatures.png`).
- **t = 1.0** names creatures better (more texture cues) but is judged blob far more often. By eye it is speckled and fragmented.
- **Neither setting** gives eyes. Detail needs a better token space or model, not a sampling knob.

### Training speed on the M4 [runs/logs/bench_train_speed.txt]
Batch 256, T 64, step time in seconds:

| model | params | fp32 | bf16 |
|---|---|---|---|
| dm128 L4 | 0.81M | 0.140 | 0.131 |
| dm256 L6 | 5.3M | 0.531 | 0.483 |
| dm384 L8 | 15.0M | 1.097 | 0.974 |

- bf16 autocast saves only about 10%.
- dm256 L6 at T = 144 (2×2 tokens) ran out of memory under the 0.45 MPS cap at B = 256, so 2×2 models need a smaller batch.
- **Teacher budget:** a 5M teacher at 60k steps takes about 8 h. That is feasible overnight, but only one at a time.

### v4s_a: slot tokens beat colour patches at the same size [metrics/e1_v4s.json, e1_v4s_g0.json, e1_paired_v3__v4s.json; D-57]
- **Setup:**
  - `v4s_a`: dm128 L4, 823k params, 60k steps, 2.6 h on the M4. Its tokens are 4 palette tokens plus 64 3×3 slot-shape tokens (K=1024).
  - `v4s_u48`: unconditional chain, dm48 L2, 20k steps.
  - Sampled exactly like the sheet v3: g 0.75, t 0.8. Val NLL 2.475, token accuracy 0.488 (not comparable to v3: different tokens).

  | metric | real | v3 | **v4s** | v4s g=0 |
  |---|---|---|---|---|
  | VLM naming, all | 0.662 | 0.539 | **0.596** | 0.453 |
  | naming, creatures | 0.733 | 0.578 | **0.638** | 0.509 |
  | naming, objects | 0.625 | 0.708 | **0.750** | 0.558 |
  | naming, test captions | 0.542 | 0.396 | **0.458** | 0.385 |
  | VLM clear | 0.424 | 0.223 | **0.316** | 0.241 |
  | VLM eyes (creatures) | 0.259 | 0.047 | **0.146** | 0.179 |
  | colours per sprite | 3.55 | 2.74 | 2.56 | 2.39 |
  | giant/tiny area | 1.32 | 1.33 | 1.11 | 1.14 |
  | near-duplicates | — | 0.001 | 0.001 | 0.003 |

- **Paired bootstrap over prompts** (v4s − v3, 10k draws):
  - naming +0.058, 95% CI [0.017, 0.098]: better on 79 prompts, worse on 47;
  - clear +0.092, CI [0.048, 0.136];
  - eyes +0.099, CI [0.061, 0.137]: 22 prompts better, 2 worse.

  All three are clear wins. Facing is not (+0.016, CI [−0.17, 0.20]).
- **By eye** (`sheets/v3_vs_v4s_vs_v4s_g0_creatures.png`, `v3_vs_v4s_objects.png`):
  - **Creatures:** cats have ears and faces (a purple cat with eyes). Owls have faces with eyes on 5 of 8 seeds, against 1 of 8 for v3. Foxes have ears and white muzzles. Fish have outlines and fins. v3's white silhouettes are gone.
  - **Objects:** shields are outlined heater shields with emblems, and axes have real heads. Potions and crowns are about equal. Bows are worse: two seeds drew a chest or frame.
  - **Outlines:** sprites now often have a dark outline. v3 almost never drew one, because each outline colour pairing needed its own dictionary entry.
- **Why it works:** shape and colour are factored, so one dictionary entry ("eye dot on fur", "outline corner") serves every colour. The palette tokens fix the sprite's colours first, so shape tokens can't drift into colour noise.
- **Still weak:**
  - Size control: giant/tiny area ratio 1.11 vs 1.33. It is the weakest group, and slot shapes may lose the scale cue.
  - Colours per sprite: 2.56 vs real 3.55. The guided model stays under-shaded.
  - Facing: only 29 of 217k captions mention facing, so it cannot be learned from these captions. Fixing that needs labels (flip augmentation plus VLM facing labels).
- **Decision:** v4 slot tokens are the new base. Next:
  1. A bigger teacher, `v4s_b` (dm256 L6, 5.3M params, 50k steps, training now), then sequence-level distillation into a sheet-size student.
  2. Pixcell already compiles slot models (Gate A2 passes).
  3. Push v4s_a + u48 to a sheet.

### v4s_a on Google Sheets (A4, probe sheet) [metrics/parity_sheets_probe.json; figures/probe_sheet_readback.png; logs: runs/logs/push_v4s_a_probe.txt, verify_v4s_probe.txt]
- **Push:**
  - What: `v4s_a` + `v4s_u48` at t=0.8 (export `artifacts/model/v4s_a_sheet`), sent to the **probe** scratch sheet. The v3 demo sheet is untouched.
  - Cost: 3,176 s, 184 requests, 93.2 MB, using the inert upload plus row activation.
- **Check:** the guidance cell was set to 0.75, and the same 8 prompt/seed pairs as the v3 A4 were read back. **8/8 match the spec pixel for pixel.**
- **Latency:**
  - first consistent read: 103–143 s;
  - stable: 123–167 s.

  That is the same as v3: the palette tokens add 4 steps, and the masked Score cells cost nothing.
- **By eye**, on these fixed seeds (not picked):
  - Good: potion, frog, mushroom, crown, chest, and a slime with an eye.
  - Weak: the fox (an orange blob with an eye) and the owl (pale scatter).

  E1 is the fair comparison. This check is about parity.

### Why v4s draws too few colours, and temperature on v4 (CPU sampling)
- **Colours per sprite** (share of sprites with 1/2/3/4 colours; E1 grids):

  | | 1 | 2 | 3 | 4 | mean |
  |---|---|---|---|---|---|
  | real | 0.01 | 0.09 | 0.26 | **0.65** | 3.55 |
  | v4s (t 0.8) | 0.13 | 0.35 | 0.35 | 0.17 | 2.56 |
  | v4s t 0.9 | 0.09 | 0.29 | 0.35 | 0.27 | 2.79 |
  | v4s t 1.0 | 0.07 | 0.26 | 0.33 | 0.34 | 2.93 |
  | v4s g 0.5, t 1.0 | 0.06 | 0.25 | 0.36 | 0.33 | 2.97 |

- **Two losses** (spec codes, 40 random prompt/seed pairs at t 0.8):
  1. The model picks a smaller palette: a mean of 3.0 colours, with 35% picking 4. In training, 62% of palettes have 4 colours.
  2. The shape tokens then leave chosen slots unused: 2.48 colours are actually drawn.

  Low temperature favours the commonest patches (solid fills of the main slot), and those are exactly the minority slots: eyes, shading, outlines.
- **By eye** (`sheets/v4s_vs_v4s_t09_vs_v4s_t1_creatures.png`): unlike v3, v4s at t 1.0 mostly keeps the shape and adds detail: fox chests, fish eyes and fins, frog spots. A few seeds get noisier (two owls, one dog). t 0.9 looks like the middle ground.
- **Next:** judge t 0.9 / 1.0 / g 0.5 t 1.0 (queued after the teacher; the judge never runs alongside training). Temperature is a live cell in neither sheet yet; the sheet bakes 1/τ into Score.

### Teacher v4s_b (dm256 L6, 5.3M params): val NLL overfits, samples get much better [runs/v4s_b/train_log.jsonl; metrics/e1_neardup_teacher.json]
- **Training:** 50k steps, 8.0 h on the M4 (0.58 s/step). Train loss 1.94. **Val NLL is lowest at 12.5k (2.456)** and then rises to 2.635, worse than the 0.8M `v4s_a` (2.475). By that number it overfits from step ~12k.
- **By eye it is the biggest jump of the project** (`sheets/v4s_vs_v4sb_vs_v4sb_t1_creatures.png`; same sampling, g 0.75, t 0.8):
  - every frog is front-facing with two eyes and a pale belly;
  - foxes are classic side-view foxes with white chests and tails;
  - owls have eyes and feather detail;
  - birds have beaks and wings, and snakes have patterns and tongues.

  Test captions (`v4sb12k_vs_v4sb_test.png`): moths, coral, copper moulds and crow butlers are all recognisable. The 50k checkpoint has more detail and variety but more stray pixels. The 12.5k one is a bit cleaner.
- **Not memorising** (seeds 1–4 × 187 prompts, nearest training sprite by pixel difference):

  | | < 5% | < 10% | < 20% | median |
  |---|---|---|---|---|
  | v4s_a | 0.13% | 1.7% | 35% | 0.233 |
  | v4s_b 12.5k | 0 | 0.9% | 30% | 0.242 |
  | v4s_b 50k | 0 | 0.9% | 24% | 0.267 |

  The teacher's samples are *further* from the training set than the small model's.
- **Lesson:** rising val NLL here is overconfidence on held-out sprites (a wrong guess costs more), not copying and not worse samples. **Val NLL is not a model-selection metric for sampling quality.** Use E1 plus the eye.
- **Problem:** a 5.3M model in the sheet would take roughly 6× the compute per prompt change (~15 min), so it needs distilling into the dm128 L4 sheet size. That is next, after the judge picks the teacher checkpoint.

### How prompt-change time was measured, and why the "2–3 min" number is suspect (2026-11-21)
- **Where every latency number so far came from:** `sheets.set_prompt_and_read`. It writes Prompt!B2:B3 through the **Sheets API**, then polls Pixels through the API (1 s sleep, plus each read's own latency) until two reads agree. That is server-side recalculation as the API sees it, not what a person sees typing in the browser. MP2's "2–3 min" slide figure came from this, and so did v4's 103–167 s on the probe sheet.
- **Brian reports** that typing a prompt on the v3 sheet in the browser is much faster than that.
- **Browser attempt (probe sheet, v4s_a):** typed "a red potion" in Chrome and watched the sprite with an in-page canvas poller. **Inconclusive:** the laptop was saturated (load ~50, 3% idle) by bank generation, scoring and the GPU queue all at once. The server had the new pixels (read via API), but the browser still showed the old sprite after 7 min.
- **Rule from now on:**
  - Time prompt changes in the browser on an idle machine.
  - Report median and slowest over several prompts.
  - Compare builds side by side under the same conditions.
  - Never quote the API polling time as user latency.
- **Peer feedback on speed** (MP2 critiques, all 11 reviewers mention it):
  - profile which tabs dominate recalculation;
  - measure prompt change separately from load;
  - try array formulas (`MMULT`, `LAMBDA`/`REDUCE`) instead of many `SUMPRODUCT` cells;
  - a smaller teaching build that recalculates in seconds.

  MP3 tracks these as experiments below.

### Speed experiment 1: MMULT vs SUMPRODUCT, timed in the browser [metrics/bench_mmult_browser.json; scripts/bench_mmult_sheets.py]
- **Setup:** the same 6-layer dm128 linear chain over 64 rows, built 3 ways on tabs of the probe sheet:
  - A: one SUMPRODUCT per cell (how Pixcell emits linear layers today), 49,152 formulas;
  - B: one MMULT per row (works for the causal model), 384 formulas;
  - C: one MMULT per layer block (upper bound; would be circular in the causal model), 6 formulas.

  Timed in Chrome from Enter on the input cell until a checksum flag turns green, using an in-page canvas poller. A trial only counts if the flag was seen red first; 2 trials were rejected for a stale green.

  | | formulas | first change after page load | warm change |
  |---|---|---|---|
  | A SUMPRODUCT per cell | 49,152 | 12.9, 19.3 s | 6.0 s |
  | B MMULT per row | 384 | 14.3 s | 2.5, 3.3 s |
  | C MMULT per block | 6 | 15.9, 17.4 s | 0.6 s |

- **Learned:**
  1. Sheets recalculates **in the browser**. The first edit after opening a sheet costs a fixed ~13–19 s warm-up, whatever the formula count.
  2. After that, fewer and larger array formulas are much faster: per-row MMULT is about 2× faster than per-cell SUMPRODUCT, and whole-block MMULT about 10×.
  3. API polling is useless for timing on this spreadsheet: a one-cell API read takes 15–21 s.
- **Caveat:** the laptop was partly loaded (GPU queue running, ~35% CPU idle). These are single trials. Treat them as direction, not final numbers.

### Real prompt change on the v4s probe build, in the browser (inconclusive)
- Typed "a frog" (seed 5) right after opening the probe sheet. The **server** had the correct pixels (read via API: equal to the spec), but the **browser** had still not redrawn the sprite after 8 min (canvas poller, laptop loaded). Not a usable number.
- **Lesson:** the browser computes the 500k-formula model itself, and on a busy laptop that is very slow. Prompt-change time has to be measured on an idle machine, and ideally by a person with a stopwatch on the v3 and v4 sheets.

### E1 for the temperature variants and the teacher checkpoints [metrics/e1_v4s_t09.json, e1_v4s_t1.json, e1_v4s_g5t1.json, e1_v4sb12k.json, e1_v4sb.json, e1_v4sb_t1.json]

| metric | real | v3 | v4s | v4s t0.9 | v4s t1 | v4sb 12.5k | **v4sb 50k** | v4sb t1 |
|---|---|---|---|---|---|---|---|---|
| naming, all | 0.662 | 0.539 | 0.596 | 0.595 | 0.556 | 0.678 | **0.702** | 0.695 |
| naming, creatures | 0.733 | 0.578 | 0.638 | 0.664 | 0.741 | 0.776 | 0.828 | 0.871 |
| naming, objects | 0.625 | 0.708 | 0.750 | 0.708 | 0.725 | 0.783 | 0.833 | 0.825 |
| naming, test captions | 0.542 | 0.396 | 0.458 | 0.453 | 0.391 | **0.505** | 0.422 | 0.427 |
| clear | 0.424 | 0.223 | 0.316 | 0.271 | 0.204 | 0.409 | 0.404 | 0.290 |
| eyes | 0.259 | 0.047 | 0.146 | 0.193 | 0.170 | 0.207 | 0.212 | 0.222 |
| CLIP R@5 | 0.396 | 0.443 | 0.473 | 0.459 | 0.423 | 0.564 | 0.625 | 0.561 |
| colours/sprite | 3.55 | 2.74 | 2.56 | 2.79 | 2.93 | 2.71 | 2.87 | 2.99 |

- **Teacher, 50k vs 12.5k steps:**
  - On common nouns, 50k is best: creatures 0.83, objects 0.83, above real sprites.
  - On held-out test captions, 12.5k is better: 0.505 vs 0.422.

  The rising val NLL was real overfitting, but only for unseen captions. For the next teacher, use a checkpoint in between or more regularisation (dropout, caption augmentation).
- **Temperature:** t = 0.8 keeps the best clarity on both models. Higher t buys colours and eyes at the cost of blobs. It is not a free win.
- Paired v4s → v4sb (50k): naming +0.106, CI [0.063, 0.148]; clear +0.088, CI [0.040, 0.136]; eyes +0.066, CI [0.005, 0.127] (`e1_paired_v4s__v4sb.json`).
