# SpriteSheet

**A text-to-pixel-art neural network that runs entirely in Google Sheets formulas.**

Type a prompt into a cell, like *"a tiny blue rocket facing left"*, and a 32×32 sprite appears in the sheet's square cells. Every value is computed by native spreadsheet formulas: no Apps Script, add-ons, or external calls.

**Pixcell** is our compiler from a trained model to the workbook. Verification gates check that the spreadsheet matches a float64 reference (see `docs/06-evaluation-and-benchmarks.md`).

Status: **setup done; build pending** (see `GOAL.md`, `PROGRESS.md`). Headline results will appear here, generated from `artifacts/metrics/`.

- Team Whiplash · UWaterloo SE 390 · Advanced Technology
- Training data: Noto Emoji (Apache-2.0), Twemoji (CC-BY 4.0), Fluent Emoji (MIT), OpenMoji (CC BY-SA 4.0), Unicode CLDR annotations, PixelGPT 24×24 20K (CC BY 4.0), plus optional icon sets. Full credits will be in `ATTRIBUTION.md`. Raw data is not redistributed here; see `scripts/fetch_data.sh`.

Start with `docs/00-README-START-HERE.md`.
