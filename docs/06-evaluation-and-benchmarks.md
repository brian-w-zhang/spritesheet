# 06 · Evaluation and benchmarks

Every number on a slide comes from a script that writes `artifacts/metrics/*.json` (with git hash and inputs). `scripts/check_claims.py` fails if a number in README/PROGRESS/slides isn't found in `artifacts/`. **No hand-typed numbers.** A check that didn't run is reported **NOT RUN**. Results are reported per resolution (v32, then v48/v64 if built).

## A. Correctness: "the spreadsheet *is* the model" (pass/fail)

| Gate | Check | Pass |
| --- | --- | --- |
| A1 tokenizer parity | `tokenizer.py` vs the sheet's `Tokens` tab on ≥300 prompts (caps, punctuation, `*?~`, double spaces, unknown words, >16 words, plurals) | 100% identical ids |
| A2 op parity | each Pixcell op alone (dense, conv3×3, upsample, relu, attention, softmax, mean-pool, LCG, style sampling, argmax), random inputs, both local engines vs numpy64 | ≤1e-12 relative |
| A3 end-to-end, local | full workbook, ≥200 prompts × 2 seeds, HyperFormula **and** LibreOffice (forced recalc + stale check) | identical class grids; logits ≤1e-12 rel; 0 error cells; 0 stale cells; report the min top-1/top-2 margin |
| A4 end-to-end, Google Sheets | push via the API, set prompts, read `Pixels` | identical grids on ≥20 prompts, or **NOT RUN** (with reason) |
| A5 data pipeline | `normalize.py` is deterministic (hash of the normalised corpus is stable); integer scaling only for pixel art; palette snap is idempotent | tests pass |

## B. Quality (automatic; no human judgement)

All sets are fixed in `data/splits.json` before training.

| Metric | How | Sets |
| --- | --- | --- |
| **Pixel accuracy** vs exact target | class-id match per pixel (foreground-weighted and plain) | emoji variations: train, **held-out (concept×attribute) combos**, held-out phrasings |
| **Attribute accuracy** | programmatic detectors on the *output*: dominant foreground hue vs the requested colour; bbox size vs tiny/huge; bbox centre vs position; flip (correlation with flipped vs unflipped target); count (connected components); background class | held-out combos |
| **Concept accuracy (CLIP)** | upscale the output ×8 nearest; CLIP zero-shot over all concept names → top-1 / top-5 | all sets incl. **unseen words** and PixelGPT test |
| **CLIP score** | cos(CLIP image emb of output, CLIP text emb of prompt), also relative to the same score for the ground-truth target (a "% of target score" normaliser) | all |
| **Diversity across seeds** | mean pairwise pixel distance between seeds for prompts without a style; style-distribution entropy | under-specified prompts |
| PixelGPT | CLIP score + concept accuracy on its test split (no exact target for free text) | PixelGPT test |

## C. Baselines and ablations

| ID | Baseline | Shows |
| --- | --- | --- |
| C1 | **Retrieval:** return the training image with the most similar caption (CLIP text sim) | a lookup can't produce held-out combos ("blue banana") |
| C2 | **Bag-of-words encoder** (same decoder) | what attention buys (word order, binding: "red cat, blue hat") |
| C3 | **No CLIP table** (learned word embeddings only) | open vocabulary: unseen words collapse to `<unk>` |
| C4 | **SpriteX32 head-to-head:** run `viknesh100/SpriteX32` locally (MPS/CPU) on the same prompts. Same CLIP concept accuracy and CLIP score, plus params / steps / text-encoder size / medium. **Read its `pipeline.py` before running it.** Note honestly that it was trained on Pokémon sprites, so emoji-style prompts are out of its domain | where we stand against a real 32×32 T2I model 1000s of times bigger |
| C5 (stretch) | per-cell vs per-layer `MMULT` formulas | compiler design: cells, bytes, Sheets recalc time |
| C6 (stretch) | capacity curve: 2–3 decoder sizes | params vs quality vs recalc time, which sets the v48/v64 plan |

## D. Systems numbers ("a medium never meant for it")
Params; word-table cells; weight cells; formula cells; total cells vs the **20M-cell / 100 MB** Sheets limit; total formula characters; xlsx bytes; Sheets API upload time and request count; **prompt→sprite latency in Google Sheets** (median/p90 over ≥10 prompts, by polling `Pixels`); local recalc time (HyperFormula, LibreOffice); numpy forward time (for the "spreadsheet is N× slower than numpy" stat).

## E. Related-work table (`artifacts/metrics/related_work.json`)
Columns: project · medium · what runs inside the medium · size · steps · how correctness is shown · licence of the training data.
Rows: SpriteSheet v32 (v48/v64) · SpriteX32 · GPT-2 in Excel · Freak in the Sheets. Use only facts from doc 02's opened sources; "not stated" otherwise.

## F. Expert comments (Brian, not the agent)
2–3 one-paragraph reactions to the demo (an ML TA or professor, a pixel-art or game-dev person, the GPT-2-in-Excel author if reachable). Logged verbatim in `docs/feedback.md` with dates and permission.
