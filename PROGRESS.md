# PROGRESS: SpriteSheet v32

Started 2026-11-04 ~00:30 EST (overnight `/goal` run). Updated after every milestone.

## Definition of done

| ID | MUST/STRETCH | Item | Status | Evidence (command → key line) |
| --- | --- | --- | --- | --- |
| A1 | MUST | Tokenizer parity | PASS | `uv run pytest -q` + `artifacts/metrics/parity_tokenizer.json` → 320 prompts, 0 mismatches in HyperFormula 3.5.0 and LibreOffice 26.8 |
| A2 | MUST | Op parity (HyperFormula + LibreOffice) | PASS | `artifacts/metrics/parity_ops.json` → pass=true; per-block max_rel ≤ 4.2e-13 in both engines (v1); v2 ops: `parity_ops_mdm*.json` |
| A3 | MUST | End-to-end parity (local) | PASS (v1-XL = artifacts/model/v32) | `scripts/verify_e2e.py --model artifacts/model/v32 --prompts 200 --seeds 2` → `artifacts/metrics/parity_local.json`: HyperFormula 400 cases max_rel 4.74e-13, LibreOffice 400 cases max_rel 3.7e-13, 0 pixel/style mismatches, all 7 styles (from the clean-clone run below) |
| A4 | MUST* | End-to-end parity (Google Sheets) | PASS (v1-XL = artifacts/model/v32) | `scripts/verify_sheets.py --model artifacts/model/v32 --prompts 20 --skip-push` (after a push of the same model) → `A4 sheets[v32]: cases=20 max_rel=9.31e-14 pixel_mismatch=0 style_mismatch=0 latency median=38.3s -> PASS` (`parity_sheets.json`); the checker re-reads once if a value is read mid-recalculation (recorded as `rereads`) |
| A5 | MUST | Data pipeline tests | PASS | `uv run pytest -q tests/test_data.py` → `9 passed` |
| B | MUST | Quality metrics | PASS | `spritesheet.evaluate --model artifacts/model/v32` → `artifacts/metrics/quality.json` (6 sets incl. pixel, attribute, CLIP concept, seed diversity); generalisation R-precision → `generalization.json` |
| C1–C3 | MUST | Retrieval / BoW / no-CLIP-table | PASS | `baseline_retrieval.json`; `abl_bow.json`, `abl_nocliptable.json`, summarised in `ablations.json` + `generalization.json` |
| C4 | MUST | SpriteX32 head-to-head | PASS | `scripts/bench_spritex32.py` → `artifacts/metrics/spritex32.json` (CLIP score: held-out combos 0.267 vs 0.249; figure kept local: SpriteX32 outputs depict Pokémon IP) |
| C5 | MUST | Data-scaling + mixture ablations | PASS | `abl_pg5k/pg10k/base/emojionly/pgmajor` → `ablations.json`, `data_scaling.json`, `generalization.json` (+ correction in docs/09) |
| D | MUST | Systems numbers | PASS | `scripts/systems_metrics.py --model artifacts/model/v32` → `systems.json` (1,282,295 cells, 169,526 formulas, Sheets median 30.6 s via API) |
| E | MUST | Related-work table | PASS | `scripts/related_work.py` → `related_work.json` |
| F | MUST | Figures | PASS | hero_grid, variations, vs_target, unseen_words, seeds, vs_spritex32 (local only), network_tabs, pipeline.svg, latency_cells (+ data_scaling, line-ups) |
| G | MUST | `scripts/verify_all.sh` | PASS | Fresh `git clone` of origin @1c6fe93, `bash scripts/verify_all.sh` → exit 0 in 156 min: pytest PASS (19 passed; 8 data tests skip on a clean clone because raw data is not redistributable, A5 runs on the local build), A3 PASS, A4 PASS (20 prompts, max_rel 9.31e-14), H PASS. Log: `artifacts/runs/logs/verify_all_clean.txt` |
| H | MUST | `scripts/check_claims.py` | PASS | `uv run python scripts/check_claims.py` → `checked 26 numbers in claims blocks … PASS` (README Results section) |
| S1–S5 | STRETCH | see GOAL.md | TODO | |

## Log
- 00:30 · Setup verified: Sheets probe PASS (service account → shared sheet), MPS available, data + CLIP local.
- 00:45 · Data pipeline done (tier 0: 300 concepts / 1,800 bases; tier 1: 1,409 concepts, 8,358 emoji bases + 20,000 PixelGPT; S = 7 styles, K = 32). CLIP word table V = 12,119, d = 64 (d chosen by a synonym-rank check: puppy→dog rank 37 at d=64 vs 233 at d=32).
- 00:50 · torch twin == numpy reference (test). Tier-0 bring-up (165k params, 4k steps): shapes and colours right, details blurry; train fg-pixel acc ≈ 0.50. Training a larger model (C = 96/64/48/32, 15k steps).
- Pixcell compiler + oracles: A1 (320 prompts) and A2 (2 random models × 10 cases) PASS on HyperFormula + LibreOffice; A3 random-v32 100 cases PASS; A4 random-v32 20 prompts PASS on Google Sheets; R1/R3 spikes in artifacts/metrics/spikes.json (random weights; rerun A3/A4 with the trained model).
- 03:50 · A3 PASS on trained v1 (400 cases × 2 engines). A4 PASS earlier (20 prompts, median 30.6 s). v2a (diffusion) underperformed v1; v2b (+ spatial prior, with PixelGPT) and v2c (PixelGPT-majority) training. Switched generalisation measurement to CLIP R-precision (see docs/09).
- 06:00 · C4 SpriteX32 bench, D systems, E related work, F figures done for v1 main. v2c (diffusion, 4 steps) uploading to the diffusion sheet (inert method). v1-XL training (ETA ~08:00).
- 2026-11-05 04:30 · **v3 final model live on the v3 sheet:**
  - **Model:** `v3p_a` (causal PatchGPT, dm128 L4, 3×3 patches from a 1024-entry dictionary) plus the in-sheet classifier-free guidance chain `v3p_u48` (D-55). Total 492,956 formulas, t=0.8. Export: `artifacts/model/v3p_a_g1_sheet`.
  - **Guidance:** a live cell, Prompt!B4, set to 0.75.
  - **Verified:** `scripts/verify_v3p_sheet.py` → `parity_sheets_v3p.json`: 8/8 prompts read back from Google Sheets match the spec pixel for pixel. Figure: `v3_sheet_readback.png`.
  - **Latency:** ~110–137 s to the first consistent image, ~136–167 s until stable.
  - **Quality:** novel R@5 0.61 [0.52, 0.70], against v1-XL 0.41 (r15).
  - **Rejected:** grouped / distilled / fine-tuned variants did not beat it (docs/09 r8–r17).
  - **Next:** slides (`docs/slides/`); team info TODOs are for the team.
