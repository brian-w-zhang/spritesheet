# MP2 slide kit

What goes on each slide, the figure to use, and where every number comes from. The course asks for 5–8 slides and a 5-minute video (`docs/01-course-requirements.md`). Every number here is copied from an `artifacts/metrics/*.json` file named next to it. If a number changes, re-run the script and update it here; don't edit it by hand.

**Never put real PixelGPT sprites on a slide** (licence: weights yes, images no). The `v3_groundtruth_*` figures and the VQ reconstruction snapshots are local only, for that reason. Everything else in `artifacts/figures/` shows only our model outputs, emoji renders or diagrams.

---

## 1. Problem and why it's hard (title + hook)
- **One line:** type a prompt into a Google Sheet and a neural network made only of spreadsheet formulas draws a pixel-art sprite. No scripts, add-ons or external calls.
- **Figure:** `artifacts/figures/hero_grid.png` (v1 outputs), or a screen recording of the live sheet.
- **Related work and how we differ:** `artifacts/metrics/related_work.json` and `docs/02-prior-art.md`.
  - Spreadsheet ML exists only as tiny classifiers or hand-built demos.
  - Text-to-pixel-art models (SpriteX32, the Five-Dollar Model) run on GPUs or in Python.
  - As far as our search found, none runs as native spreadsheet formulas. Phrase it that way; it isn't proven.

## 2. How it works: the Pixcell compiler
- **Figures:**
  - `artifacts/figures/pipeline.svg` (prompt → tokenizer → text encoder → conv decoder → palette argmax);
  - `artifacts/figures/network_tabs.png` (one tab per layer).
- **Numbers** (v1-XL, `systems.json`):
  - 736,224 trained parameters;
  - 220,806 formulas;
  - 1,771,223 cells (8.9% of Google's 20M-cell limit);
  - 45 tabs.
- **Correctness:** float64 numpy `reference.py` is the spec. The sheet matches it pixel for pixel. In real Google Sheets: 20 prompts, 0 mismatches, max relative error 9.3e-14 (`parity_sheets.json`). The same check runs locally in HyperFormula and LibreOffice (`parity_ops.json`, `parity_tokenizer.json`).
- **Speed:** median 38.3 s per prompt through the API (`parity_sheets.json`). About 3–4 s by hand in the browser; that one is not measured by a script, so say "about".

## 3. Results: v1 (the live demo)
- **Figures:**
  - `artifacts/figures/lineup_xl_trained.png`, `lineup_xl_novel.png`;
  - `seeds.png` (seed → style);
  - `unseen_words.png`;
  - `curves_v1.png` (loss curves).
- **Numbers:**
  - Scaling was the biggest lever: fg-pixel accuracy on held-out combinations 0.625 → 0.676, from 40k to 150k steps at 2.4× width (`quality_v1small.json`, `quality.json`).
  - Novel prompts (in neither dataset), CLIP R@5 0.41 (`generalization_v1xl.json`).
  - Against SpriteX32 (an 11,405,827-param diffusion U-Net, 50 steps), our first v1 (309,648 params, 1 pass) has a higher CLIP score on held-out combinations: 0.267 vs 0.249 (`spritex32.json`).
  - Data ablations: `curves_ablation.png`, `data_scaling.png` (`ablations.json`, `data_scaling.json`).

## 4. Iteration: fighting "mode averaging" (v2 → v3)
**The story:** a one-shot network trained per pixel draws the *average* of every sprite that fits the prompt, so you get blobs. We attacked this three ways:
- **v2, masked discrete diffusion** (draw in 4 steps). It runs in Google Sheets: 836,342 formulas, uploaded with an inert-text trick (`sheets_v2c_T4_b06.json`). Comparable to v1 by eye, not better, and about 4× slower.
  - Figures: `compare_v1_v2_a.png`, `v2_steps_a.png`, `curves_v2.png`.
- **v3, learned variants** (24×24, PixelGPT 216K, 16 winner-takes-all variants picked by the seed). Live and exact in the v3 sheet: 136,872 formulas, 0 mismatches (`parity_sheets_v3_var16.json`, `sheets_gallery_v3_var16.png`). Colour adjectives improved from 0.658 (v1-XL) to 0.896, but variants split colour, not structure.
- **v3, a VQ code grid + autoregressive prior** (6×6 grid of 256 codes). The first v3 with multi-colour interiors:
  - colour-adjective accuracy 0.971;
  - seed diversity 0.604, vs 0.291 for v1-XL;
  - by eye still mushy and oversized;
  - all from `v3_eval_r4.json`.
  - Figures: `v3_lineup_r4_creatures.png`, `v3_seeds_r4_v3_vq6.png`, `curves_v3.png`.
- **Honest takeaway:** on CLIP retrieval, novel-prompt R@5 for all v3 models and v1-XL overlaps within 95% CIs (`v3_eval_r4.json`). v1-XL stays the deployed model.
- *(Critic fine-tune and guided-prior results go here when their runs finish; see `docs/09-experiment-log.md`.)*

## 5. How we organized (TODO: team to fill; not inferred by the agent)
- Meeting cadence: ___
- Task management and kanban: ___ (screenshot of the board)
- Who owned what: ___

## 6. Use of AI (from `AI-USE.md`)
- **Tool:** Claude Code. It did the research with source lists, the docs-first design, the overnight implementation, tests, training runs and figures.
- **Review:** checks, not trust. Every claim is a script with pass/fail output (parity gates A1–A4; `check_claims.py` for README numbers; `verify_all.sh` from a clean checkout).
- **What worked:** the parity gates caught three spec bugs that review alone missed (D-30, D-31, D-32).
- **What didn't:** Google auth; running two trainings at once exhausted GPU memory and stalled both. We also had to stop the agent trusting loss values and make it look at real Sheets output.
- **Not delegated:** the project choice, novelty claims, outreach, and the narration.

## 7. Lessons and next step
- **Lessons:**
  - Exactness is cheap once you have a float64 spec and parity gates.
  - Quality is the hard part: the per-pixel loss averages, so you need a latent (variants, codes) that commits first.
  - Browser recalculation speed, not the cell limit, is the real budget.
- **Next:** the VQ model in the sheet (codes grid plus an inverse-CDF sampler in formulas), and the MP3 pivot (TODO: team).

---

**Loss curves for every run:** `artifacts/figures/curves_{v1,v2,v3,ablation}.png`. Per-run metadata and snapshots are in `artifacts/runs/<run>/`, indexed by `scripts/collect_runs.py`.
