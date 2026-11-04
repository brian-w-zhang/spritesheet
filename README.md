# SpriteSheet

**A text-to-pixel-art neural network that runs entirely in Google Sheets.**

Type a prompt into a cell, like *"a tiny blue rocket facing left"*, and a 32×32 sprite appears in the sheet's square cells. Every value is computed by native spreadsheet formulas: no Apps Script, add-ons, or external calls.

**Pixcell** is our compiler from a trained model to the workbook. Verification gates check that the spreadsheet matches a float64 reference (see `docs/06-evaluation-and-benchmarks.md`).

Status: **v1 built and verified in Google Sheets; v3 in progress** (see `PROGRESS.md`, `docs/09-experiment-log.md`). Every number below comes from `artifacts/metrics/` (checked by `scripts/check_claims.py`).

## Results

**v1 (the live demo, `artifacts/model/v32`).**
<!-- claims -->
- 736,224 trained parameters, compiled into 220,806 formulas (1,771,223 cells). [`systems.json`]
- **Exact in real Google Sheets:** 20 prompts, 0 pixel mismatches, median 38.3 s per prompt through the API. [`parity_sheets.json`] (Measured by hand in the browser: about 3–4 s.)
- Training longer and wider was the biggest lever: fg-pixel accuracy on held-out combinations (style given) 0.625 → 0.676 (40k → 150k steps, 2.4× width). [`quality_v1small.json`, `quality.json`]
- Novel prompts (in neither dataset), CLIP R@5: 0.41. [`generalization_v1xl.json`]
- Against SpriteX32 (11,405,827-param diffusion U-Net, 50 steps), our first v1 (309,648 params, one pass) scores a higher CLIP score on held-out combinations: 0.267 vs 0.249. [`spritex32.json`]
<!-- /claims -->

The spreadsheet matches the float64 reference to about 1e-13 relative error in HyperFormula, LibreOffice and Google Sheets (`parity_*.json`).

**v2: masked discrete diffusion (second sheet, with a filmstrip of the steps).**
<!-- claims -->
- The trained 4-step model runs in Google Sheets: 836,342 formulas, uploaded with the inert-text trick (D-48). [`sheets_v2c_T4_b06.json`]
<!-- /claims -->
All 5 test prompts matched the reference pixel for pixel. Comparable to v1 by eye, not better, and slower. Each step must reuse the formula budget.

**v3: 24×24, PixelGPT 216K, no styles (in progress; third sheet).**
<!-- claims -->
- Colour adjectives are much better than v1: 0.933 vs 0.658 dominant-hue accuracy. [`v3_eval_r3.json`]
- Live and exact in Google Sheets: 136,872 formulas, 6 prompts, 0 mismatches. [`parity_sheets_v3_var16.json`]
<!-- /claims -->
By eye, v3 does not yet beat v1: outputs average many sprites into blobs (*mode averaging*). Current work: learned variants, a VQ code grid with an autoregressive prior, guidance and a critic fine-tune. See `docs/09-experiment-log.md`.

- Team Whiplash · UWaterloo SE 390 · Advanced Technology
- Training data: Noto Emoji (Apache-2.0), Twemoji (CC-BY 4.0), Fluent Emoji (MIT), OpenMoji (CC BY-SA 4.0), Unicode CLDR annotations, PixelGPT 24×24 20K (CC BY 4.0), plus optional icon sets. Full credits will be in `ATTRIBUTION.md`. Raw data is not redistributed here; see `scripts/fetch_data.sh`.

Start with `docs/00-README-START-HERE.md`.
