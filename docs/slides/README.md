# MP2 slides

The deck is `deck/index.html` (8 slides, 1600×900), printed to `deck/slides.pdf` with headless Chrome:

```
uv run python scripts/slide_assets.py      # sprites.js (our model's outputs) + artifacts/metrics/systems_v3p.json
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
  --virtual-time-budget=15000 --print-to-pdf=$PWD/docs/slides/deck/slides.pdf file://$PWD/docs/slides/deck/index.html
```

Narration: `script.md`. Abstract draft: `abstract.md`. Fonts (Archivo, JetBrains Mono) load from Google Fonts, so print with a network connection.

**Design.** The slide is a spreadsheet: column letters, row numbers, sheet tabs for navigation, and one formula bar. Sprites are drawn as cells, because on the sheet they are cells. All colours are entries of the model's own 48-colour palette. Blue (`#2d56ce`) marks the selected cell and the final model. Yellow marks only what the team still has to fill in.

**Never put real PixelGPT sprites on a slide** (licence: weights yes, images no). Every sprite in the deck is our model's output, from `scripts/slide_assets.py`.

## Sources for every number

| Slide | Number | Source |
|---|---|---|
| 1, 3 | 493,532 formulas; 84 tabs; 2,551,942 cells; 810,112 + 140,176 params | `artifacts/metrics/systems_v3p.json` |
| 1, 5 | 8 / 8 prompts pixel-exact in Google Sheets; 2–3 min per prompt (first consistent 110–137 s, stable 136–356 s) | `artifacts/metrics/parity_sheets_v3p.json` |
| 3 | 195,154 training sprites | `data/build/v3/report.json` (local), D-51 |
| 4 | novel-prompt CLIP R@5: v1-XL 0.41, PatchGPT 0.38 | `artifacts/metrics/v3_eval_r8.json` |
| 4 | final model 0.61 [0.52, 0.70] (g=0.75, t=0.8, U = dm48) | `artifacts/metrics/v3_eval_r15.json` |
| 4 | 8-round parallel drawing 0.32 | `artifacts/metrics/v3_eval_r11.json` |
| 4 | diffusion 836k formulas | `artifacts/metrics/sheets_v2c_T4_b06.json` |
| 6 | 44 decisions, 17 evaluation rounds | `docs/decisions.md` (44 entries, D-1…D-55 with gaps), `v3_eval_r1…r17.json` |
| 7 | three spec bugs caught by parity gates | D-30, D-31, D-32 |
| 7 | 41 s vs ~2 min | `artifacts/runs/logs/push_v3p_a_g1.txt` vs `parity_sheets_v3p.json` (docs/09) |
| 8 | 61k vs ~180k guidance formulas | `systems_v3p.json` (61,120), D-55 (179,776 for dm96 L3) |

**Team to fill in:** meeting cadence (slide 6, highlighted), who narrates which slide (`script.md`).
