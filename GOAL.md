# GOAL: build SpriteSheet v32 overnight

**Paste this to start:**
```
/goal Build SpriteSheet v32 as specified in /Users/junzhang/Projects/spritesheet/GOAL.md. Done when PROGRESS.md's "Definition of done" table has every MUST row marked PASS (A4 may be NOT RUN only for a stated external reason), scripts/verify_all.sh exits 0 from a clean checkout, the listed figures exist in artifacts/figures/, and all work is committed and pushed to origin.
```

---

## Mission
Build **SpriteSheet v32**: a text-to-pixel-art neural network that runs **entirely as native Google Sheets formulas**. The user types a prompt such as *"a tiny blue rocket facing left"* and a seed, and a 32×32 sprite appears in square cells.
- Build it with **Pixcell**, our compiler from a trained model to a workbook.
- **Prove** that the spreadsheet computes exactly what the float64 reference computes.
- Measure quality automatically, and benchmark against baselines, SpriteX32 and related work.

This is a UWaterloo SE 390 Advanced Technology project; grading rewards "substantial visible progress on the hard part" and honest reporting. **A smaller model that verifiably matches beats a bigger one that doesn't. Never fabricate.** v48 and v64 are later, separate models and spreadsheets: design for a resolution parameter, but don't build them tonight unless every v32 MUST is green.

Read first: `CLAUDE.md` (hard rules), then `docs/00-README-START-HERE.md`, then docs 03–07. The docs are the spec. Log any deviation in `docs/decisions.md`.

## Environment (prepared in the setup pass; verify, don't redo)
- macOS, Apple M4, 16 GB. `uv sync` → Python 3.12 venv with numpy, torch 2.15 (**MPS available**), transformers, openpyxl, pillow, resvg-py, pyarrow, pandas, pytest. Node 24; `tools/node_modules/hyperformula@3.5.0` (config in doc 05). LibreOffice `/opt/homebrew/bin/soffice`.
- Data already in `data/raw/` (gitignored; `scripts/fetch_data.sh` re-fetches it): noto-emoji (3,787 PNG), twemoji (4,009), fluentui-emoji (1,595 concepts, 4 styles), openmoji (4,566 + json), cldr annotations, pixelgpt-24x24-20k (parquet), game-icons (4,239 SVG), tabler-icons (6,257 SVG), gametilenet. Read doc 03 for licences and normalisation rules.
- CLIP ViT-B/32 in `data/models/clip-vit-base-patch32/` (`pytorch_model.bin`, smoke-tested). It is used offline only, for the word table and evaluation.
- Prototypes: `scripts/preview_pixelate.py` (emoji → 32/48 pixel art, looks good), `tools/verify-prototypes/` (HyperFormula/LibreOffice oracles; fix their paths).
- GitHub: `origin` = the public repo `sprite-sheet`. **Commit and push after every passing gate.** Raw data, models, course files and large workbooks are gitignored. Never commit them, and never commit PixelGPT images (licence).
- Google Sheets (**verified 2026-11-04: probe PASS**): auth is a **service account** whose key is at `~/.config/spritesheet/sa-key.json` (google-auth `service_account.Credentials`, scopes spreadsheets + drive.file; see `scripts/sheets_probe.py`). **The service account cannot create spreadsheets** (no Drive quota; the probe confirmed 403). Build into the spreadsheet Brian shared, whose id is `~/.config/spritesheet/config.json` → `sheet_ids.v32`: add, clear and rename tabs inside it, and rename it to `SpriteSheet-32` if you like. Never put the key or the sheet id in the repo. Never print the key. If the API starts failing, mark A4 NOT RUN with the error and continue locally.

## Definition of done (copy into PROGRESS.md; keep it current, with the command + key output line for every PASS)

| ID | MUST/STRETCH | Item | Pass criterion |
| --- | --- | --- | --- |
| A1 | MUST | Tokenizer parity | sheet = `tokenizer.py` on ≥300 prompts incl. adversarial, 100% |
| A2 | MUST | Op parity | every Pixcell op vs numpy64 on HyperFormula and LibreOffice, ≤1e-12 rel |
| A3 | MUST | End-to-end parity (local) | ≥200 prompts × 2 seeds on both engines: identical grids, logits ≤1e-12 rel, 0 error cells, 0 stale cells |
| A4 | MUST* | End-to-end parity (Google Sheets) | ≥20 prompts identical, or NOT RUN with an external reason |
| A5 | MUST | Data pipeline tests | deterministic normalisation; integer-only scaling for pixel art; transparent padding; palette snap idempotent |
| B | MUST | Quality metrics (doc 06 §B) | pixel acc, attribute acc, CLIP concept top-1/top-5, CLIP score, seed diversity, on all fixed splits → `artifacts/metrics/quality.json` |
| C1–C3 | MUST | Retrieval baseline, BoW ablation, no-CLIP-table ablation | same metrics |
| C5 | MUST | Data-scaling + mixture ablations (numpy/torch eval only, not compiled) | same arch, short runs: PixelGPT 5k / 10k / 18k (does more pixel-art data still help? this decides whether to buy the 216k edition) and emoji-only vs all-sources; curves → `artifacts/metrics/data_scaling.json` + `data_scaling.png` |
| C4 | MUST | SpriteX32 head-to-head | same prompts, CLIP metrics + size/steps table (read its code before running) |
| D | MUST | Systems numbers (doc 06 §D) | `artifacts/metrics/systems.json` |
| E | MUST | Related-work table | `artifacts/metrics/related_work.json`, doc 02 facts only |
| F | MUST | Figures (below) | files exist, ≥1600 px wide |
| G | MUST | `scripts/verify_all.sh` | reruns tests, A1–A3 (+A4 if creds) and metrics; exits 0 |
| H | MUST | `scripts/check_claims.py` | every number in README/PROGRESS is found in `artifacts/metrics/` |
| S1 | STRETCH | Rung 2+ of the ladder (more concepts, PixelGPT, game-icons) | report the numbers either way |
| S2 | STRETCH | Relations/accessories ("a cat wearing a hat") with held-out pairs | B metrics on held-out pairs |
| S3 | STRETCH | Interpolation slider between two prompts | parity holds |
| S4 | STRETCH | 2–4 step distilled refinement, one tab per step | parity + filmstrip figure |
| S5 | STRETCH | v48 model + spreadsheet | same gates as v32 |

## Training strategy (decided 2026-11-04)
- **One main model** trained on **all sources with a fixed sampling mixture** (start: 45% emoji + variations, 30% PixelGPT, 15% game-icons/Tabler colourised, 10% GameTileNet CC0/CC-BY tiles; tune only if a source clearly hurts held-out metrics; log in decisions.md). A **style id** per source keeps the looks separate. This is the only model compiled into the spreadsheet.
- **Bring-up first:** a tier-0 run (~300 emoji concepts) proves the whole pipeline end to end, including the sheet. Then train the main all-sources model with the same code.
- **Ablations are cheap and uncompiled:** C1–C3 and C5 are short runs of the same architecture, evaluated with `reference.py`. Parity is proven once, on the compiled main model.
- **Canvas:** 32×32. PixelGPT is padded 1:1 (after masking `tile_bg` to transparent), emoji/SVG are rendered natively at 32, and GameTileNet is used as-is.

## Phases (time boxes are guides; at each box's end, commit, push and update PROGRESS.md)
- **P0 · Check the environment (≤20 min).** Run `uv sync`, check `uv run python -c "import torch; print(torch.backends.mps.is_available())"` and HyperFormula, and run `pytest` on an empty suite. Create `PROGRESS.md` with the DoD table. Commit and push.
- **P1 · Spikes (≤75 min; doc 07).**
  - Port the oracles into `src/spritesheet/oracles/`; reproduce the 1e-16 MLP result as a test.
  - Tokenizer spike on both engines.
  - If Sheets access works: **R1** (synthetic v32-shaped workbook incl. WordVec; time prompt→pixels at 3 widths) and **R3** (32×32 and 64×64 SPARKLINE square-cell grids). Save `artifacts/metrics/spikes.json`. **R1 sets the model size budget**; log it.
- **P2 · Data (≤75 min).** `sources/`, `normalize.py`, `synth.py`, `data/concepts.json`, `data/splits.json`, the palette fit (K logged), and tests (A5). Start at **tier 0** (~300 emoji concepts × all styles, colour + direction). Render `artifacts/figures/dataset_samples.png`.
- **P3 · Text side (≤45 min).** `vocab.py`: build the vocabulary and the CLIP word table (PCA d = 64 or the R1-informed value) → `data/wordvec_*.npy`, plus `tokenizer.py`.
- **P4 · Model + reference (≤90 min).** `model.py`, `train.py` (MPS), `reference.py` (float64, D-09 rounding), and a test that torch-fp64 ≈ reference. Get rung 0 end to end before tuning. Do not tune for more than 30 min before P5.
- **P5 · Pixcell (≤120 min).** `ir.py`, `layout.py`, `emit.py` (allowed subset only), `xlsx.py`, `sheets.py` (≤2 MB batches, ≤1 req/s, repeatCell/copyPaste, backoff on 429, square display cells, hidden gridlines, SPARKLINE render). Gates in order: A2 → A1 → A3.
- **P6 · Real Sheets (≤60 min).** Write the compiled workbook into the shared v32 spreadsheet (replace tabs from earlier runs; keep the `Prompt` tab first), run A4, measure D (upload, latency). Don't change sharing (it is already view-by-link). Refer to it in PROGRESS.md as "the v32 sheet", not by id. Try a Drive PDF/PNG export of the Prompt tab for figures; if that fails, note that Brian must screenshot it.
- **P7 · Benchmarks (≤75 min).** B, C1–C4, D, E, F. Then climb the ladder: rung 1 → 2 → stretches.
- **P8 · Report (reserve 30 min; start ≥1 h before stopping).** `README.md` (what it is, how to run, headline numbers from artifacts, credits to sources), `PROGRESS.md` (final DoD, failures and why, caveats, next steps), and the `AI-USE.md` "what worked / didn't" section with specifics. `check_claims.py` must pass. Final push.

## Figures (`artifacts/figures/`, PNG ≥1600 px wide, light background, no chart junk)
1. `hero_grid.png`: 4×4 prompts → model sprites, with the prompt text under each.
2. `variations.png`: one concept × colour/direction/size/position/count.
3. `vs_target.png`: model vs exact target on held-out combos, diff pixels highlighted.
4. `unseen_words.png`: prompts using words never in training captions (e.g. "puppy", "kitty").
5. `seeds.png`: one style-less prompt × 8 seeds.
6. `vs_spritex32.png`: same prompts, both models.
7. `network_tabs.png`: activation tabs as heatmaps ("the model is visible").
8. `pipeline.svg`: stages with cell counts.
9. `latency_cells.png`: cells and latency per stage (local; Sheets if A4 ran).

## Rules of engagement
- Work only in this repo. Don't touch global config, other projects, or Drive files you didn't create.
- Commit small, frequent and pushed, under the configured identity, **with no AI attribution lines** (CLAUDE.md).
- Stuck >30 min: write the blocker in PROGRESS.md, take the doc 07 fallback, move on.
- Don't delete or weaken a test to pass. If a tolerance changes, log it in decisions.md.
- Read any third-party code (e.g. SpriteX32's `pipeline.py`) before executing it.
