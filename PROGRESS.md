# PROGRESS: SpriteSheet v32

Started 2026-11-04 ~00:30 EST (overnight `/goal` run). Updated after every milestone.

## Definition of done

| ID | MUST/STRETCH | Item | Status | Evidence (command → key line) |
| --- | --- | --- | --- | --- |
| A1 | MUST | Tokenizer parity | TODO | |
| A2 | MUST | Op parity (HyperFormula + LibreOffice) | TODO | |
| A3 | MUST | End-to-end parity (local) | TODO | |
| A4 | MUST* | End-to-end parity (Google Sheets) | TODO | |
| A5 | MUST | Data pipeline tests | PASS | `uv run pytest -q tests/test_data.py` → `9 passed` |
| B | MUST | Quality metrics | TODO | |
| C1–C3 | MUST | Retrieval / BoW / no-CLIP-table | TODO | |
| C4 | MUST | SpriteX32 head-to-head | TODO | |
| C5 | MUST | Data-scaling + mixture ablations | TODO | |
| D | MUST | Systems numbers | TODO | |
| E | MUST | Related-work table | TODO | |
| F | MUST | Figures | TODO | |
| G | MUST | `scripts/verify_all.sh` | TODO | |
| H | MUST | `scripts/check_claims.py` | TODO | |
| S1–S5 | STRETCH | see GOAL.md | TODO | |

## Log
- 00:30 · Setup verified: Sheets probe PASS (service account → shared sheet), MPS available, data + CLIP local.
- 00:45 · Data pipeline done (tier 0: 300 concepts / 1,800 bases; tier 1: 1,409 concepts, 8,358 emoji bases + 20,000 PixelGPT; S = 7 styles, K = 32). CLIP word table V = 12,119, d = 64 (d chosen by a synonym-rank check: puppy→dog rank 37 at d=64 vs 233 at d=32).
- 00:50 · torch twin == numpy reference (test). Tier-0 bring-up (165k params, 4k steps): shapes and colours right, details blurry; train fg-pixel acc ≈ 0.50. Training a larger model (C = 96/64/48/32, 15k steps).
- Pixcell compiler + oracles: A1 (320 prompts) and A2 (2 random models × 10 cases) PASS on HyperFormula + LibreOffice; A3 random-v32 100 cases PASS; A4 random-v32 20 prompts PASS on Google Sheets; R1/R3 spikes in artifacts/metrics/spikes.json (random weights; rerun A3/A4 with the trained model).
