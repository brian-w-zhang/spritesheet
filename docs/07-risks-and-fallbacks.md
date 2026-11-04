# 07 · Risks, spikes first, fallback ladder

Rule for the overnight agent: **spike the riskiest thing first, and keep a working, verified artifact at every rung.** A smaller model that matches exactly beats a bigger one that "almost" matches.

## Risk register (ordered by "would kill the demo")

| # | Risk | Spike (≤45 min each, do first) | Fallback |
| --- | --- | --- | --- |
| R1 | **Google Sheets recalc too slow / too big** at the target size (20M cells / 100 MB; API ~2 MB per request, 60 writes/min/user) | Push a synthetic workbook shaped like v32: WordVec (V×64), 1 attention layer, dense → 4 conv blocks (C = 32–64), 1024×(K+1) logits. Time prompt→pixels. Repeat at half and double width | Shrink V, d, channels or K (in that order); per-layer `MMULT` for hot layers; log it in decisions.md. **The R1 result sets the v32 model size and whether v48/v64 are realistic** |
| R2 | **No unattended API access** | `scripts/sheets_probe.py` | Verify locally, mark A4 NOT RUN, leave a one-command `scripts/push_sheets.py` for Brian |
| R3 | **Rendering:** SPARKLINE per cell too slow, or square cells can't be made small (minimum row height) | a 32×32 and a 64×64 SPARKLINE grid with random colours; try pixelSize 6/8/10/12 | conditional formatting per class; bigger cells; local PNGs |
| R4 | **Engine disagreement** (Sheets vs HyperFormula vs LibreOffice) | A2 op parity on both local engines | avoid the function; restructure; never loosen argmax tolerance |
| R5 | **Capacity:** too many concepts for a sheet-sized decoder (blurry or mixed-up icons) | train tier 0 (~300 concepts) for 15 min; read pixel accuracy + CLIP concept accuracy | stay at tier 0 for v32; spend capacity on concepts, not on rare attributes |
| R6 | **Style averaging** (7 drawings of one concept → blur) | check tier-0 outputs with vs without the style latent | the style latent is mandatory; if still blurry, drop to 3–4 styles |
| R7 | **Word-table cost** in Sheets (≈1M value cells) | included in R1 | d = 32; V = 8k; or store the table as packed rows (fewer cells, longer formulas) |
| R8 | **Data licensing slip** (e.g. publishing PixelGPT full-edition files, or forgetting CC-BY credits) | n/a | `data/raw/` is gitignored; ATTRIBUTION.md before publishing; OpenMoji is switchable |
| R9 | **Agent context loss** in a long run | n/a | `PROGRESS.md` + `decisions.md` updated after every milestone |

## Fallback ladder for v32 (stop climbing when time runs out; every rung must pass A1–A3)
1. **Rung 0:** tier-0 emoji (~300 concepts × all styles), colour + direction variations only, single-pass decoder, word table. End-to-end parity locally. This is the minimum shippable MP2 demo.
2. **Rung 1:** + size/position/count/background; + PixelGPT 18k.
3. **Rung 2:** all usable emoji concepts; game-icons; tune for held-out combos and CLIP concept accuracy.
4. **Rung 3 (stretch):** relations/accessories ("a cat wearing a hat"); interpolation slider between two prompts.
5. **Rung 4 (stretch):** 2–4 step distilled flow/refinement, one tab per step (weights shared across steps), for "diffusion in a spreadsheet".

v48 and v64 (separate models and spreadsheets) come after v32 passes all MUST gates. They are MP3 work unless the night goes very well.

## How it could fail even if everything works, and what we say then
- *"It's a lookup."* Answer with held-out combos and unseen words vs the retrieval baseline (C1).
- *"The text part is trivial."* Answer with BoW (C2) and no-CLIP-table (C3) ablations and the binding examples.
- *"SpriteX32 looks better."* It likely will on its own domain: it is far bigger, takes 50 steps and needs a GPU. Our claim is a different point in the design space, compiled into a spreadsheet and verified exactly. Show both honestly.
- *"Spreadsheet stunts exist."* GPT-2 in Excel and Freak in the Sheets exist. We claim text→image, in Google Sheets, compiled from a trained model, with exact parity.
