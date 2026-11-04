# 04 · Architecture

**SpriteSheet: a text-to-pixel-art model that runs entirely as native Google Sheets formulas.**
Type a prompt into cell `B2` and a seed into `B3`. Each tab is one stage of a neural network, and an N×N sprite (N = 32 first, then 48 and 64 as separate models and spreadsheets) appears in square cells.
There is no Apps Script, add-on, `IMPORT*` or external call. **Pixcell** is the compiler from the trained model to the workbook.

```
prompt ─► [T] tokenizer ─► [W] CLIP word table ─► [E] attention encoder ─► [Z] seeded style/noise latent ─► [D] conv decoder ─► [R] palette argmax ─► [V] render
 cell      portable          frozen, distilled        1–2 layers, trained       exact LCG in-sheet           upsample+conv        N² px × (K+1)          SPARKLINE in
           formulas          offline from CLIP        end-to-end                                              SUMPRODUCT/cell      MATCH(MAX)             square cells
```

## Where this comes from (and what's different)
SpriteX32 (a public 32×32 text-to-sprite model; doc 02) uses a frozen CLIP text encoder (~63M params) and an 11.4M-param U-Net sampled for 50 flow-matching steps. Neither fits in a spreadsheet: 20M cells total, and recalculation time is the real limit. We keep what gives it generality and reshape each piece for the sheet:

| Generalisation source | SpriteX32 | SpriteSheet |
| --- | --- | --- |
| Open vocabulary | full CLIP text encoder | **CLIP word table**: CLIP run offline once per vocabulary word, PCA-compressed to d≈64, stored in the sheet as values |
| Composition of words | CLIP's pooled vector (its card says compositional prompts are weak) | a small **trained attention encoder** over the word vectors, supervised by synthetic captions with exact targets |
| Image prior | 11.4M U-Net × 50 steps | a **single-pass conv decoder** (v1), plus a seeded **style latent** for ambiguity; a 2–4 step distilled flow version is the stretch |
| Data | Pokémon sprites (no licence) | licensed emoji (7 drawings per concept) + PixelGPT pixel art + programmatic variations |

## Components

### [T] Tokenizer (in-sheet, portable)
Lowercase; strip punctuation with a fixed `SUBSTITUTE` chain (must include `* ? ~`, which are `MATCH` wildcards); split into at most `L = 16` words with the `MID/SUBSTITUTE/REPT/TRIM` trick; `MATCH(word, Vocab, 0)`, with `IFERROR` → `<unk>` and padding with `<pad>`. A simple suffix rule is optional (strip a trailing "s" when the word isn't in the vocabulary), with the *same* rule in Python. `tokenizer.py` must match the formulas exactly (gate A1). See doc 05.

### [W] CLIP word table (offline distillation, frozen)
- **Vocabulary V ≈ 8–20k words:** every word in training captions and concept synonyms, plus common English words taken from CLIP's own BPE vocabulary (whole-word tokens, alphabetic, by merge rank). This gives an open vocabulary at no licence cost (MIT).
- **Vector per word:** the CLIP text-encoder pooled output for the bare word (try "a pixel art {word}" as an alternative, then pick by validation), L2-normalised, then PCA to d = 64 fitted on the vocabulary. Store it as a `Vocab` tab (words) and a `WordVec` tab (V×d values) of ≈0.5–1.3M cells. **Measure the Sheets upload and recalc cost (R1).** If too heavy, use d = 32 or a smaller V.
- Frozen during training, so unseen-but-in-table words ("puppy") land near trained concepts ("dog"). Ablation C3 measures this.
- *Optional auxiliary loss:* the encoder's pooled output should predict the (PCA'd) CLIP embedding of the *whole* prompt. This distils CLIP's phrase understanding into the small encoder.

### [E] Attention encoder (trained)
Word vectors (frozen) + learned positional embedding → 1–2 transformer layers (d = 64, 2 heads, ReLU MLP, no LayerNorm or a simple formula-friendly one) → masked mean-pool → condition vector `c`. Softmax is always `EXP(x − MAX)`. Every op used must exist in both `reference.py` and the Pixcell emitter.

### [Z] Seeded style / noise latent
- **Style id:** if the prompt names a style ("3d", "twemoji", "pixel art"), the encoder's style head picks it. Otherwise the in-sheet **MINSTD LCG** (`x ← MOD(48271·x, 2147483647)`, exact below 2^53) samples from the style prior. Style embedding → concatenated into the decoder input. This replaces the blurry average of 7 drawings with a crisp choice.
- Optional small noise vector `z` (uniforms from the same LCG) for within-style variety. Use it only if it measurably helps PixelGPT-style samples; log the decision.
- The rule ("named" threshold, tie direction) is implemented bit-for-bit in `reference.py`.

### [D] Decoder (per-cell formulas)
`[c, style, z]` → dense → 4×4×C₀ → [nearest-upsample ×2 → 3×3 conv → ReLU] × log2(N/4) → N×N×C → 1×1 conv → **K+1 logits** (K palette colours + background).
- Layout trick: store an H×W×C activation as an `H` × `W·C` block (channels interleaved) inside a zero border, so a 3×3 conv output cell is **one** `SUMPRODUCT` over a 3×(3C) window and a weight block. Formulas are relative and fillable (`repeatCell`/`copyPaste`).
- For N = 48 the upsample chain is 6→12→24→48 (start the dense layer at 6×6). For 64 it is 4→…→64.
- The parameter budget (~1–3M) and channel widths are set by the R1 timing spike.

### [R] Palette argmax
Per pixel: `MATCH(MAX(logits), logits, 0)` gives the class id and `INDEX(PaletteHex, id)` the colour. Cells = N²·(K+1): about 34–50k at 32, 75–110k at 48, 135–200k at 64 for K = 32–48. Ties break to the first max in both the sheet and numpy, and the minimum top-1/top-2 margin is reported.

### [V] Render (Sheets only)
The display tab is N×N **square cells**: `updateDimensionProperties` sets the same `pixelSize` on rows and columns (try 10–14 px; check Sheets' minimum row height in the R3 spike), and `hideGridlines` is on. Each cell is `=SPARKLINE(1,{"charttype","bar";"max",1;"color1",hex})`. Fallback: conditional formatting per class. The prompt box sits above, with the seed and a "style" dropdown (data validation) beside it.

## Workbook layout (one spreadsheet per resolution: `SpriteSheet-32`, `-48`, `-64`)
`Prompt` (input, seed, rendered sprite) · `Vocab` · `WordVec` · `Tokens` · `Encoder*` · `Latent` · `Dec0`…`DecN` · `Logits` · `Pixels` · `Palette` · `W_*` (weights, values only) · `About` (model hash, compiler version, parity results, data attribution).
Colour-scaled activation tabs make zoomed-out "the model is visible" screenshots.

## Training (CPU/MPS on the M4; a GPU is not needed at this size)
- The data stream is generated on the fly by `synth.py` (doc 03), with a balanced mix: emoji base and variations, and PixelGPT.
- Loss: cross-entropy over the K+1 classes per pixel. Weight foreground pixels more (backgrounds dominate). Add a style-head CE and the optional CLIP-distillation loss.
- Train in fp32 on MPS, export to float64, and round weights to 15 significant digits (D-09). `reference.py` (numpy float64) is the spec.

## Code layout
```
src/spritesheet/
  sources/        # loaders: noto, twemoji, fluent, openmoji, pixelgpt, gameicons → (concept, style, RGBA)
  normalize.py    # integer-scale/pad pixel art, render emoji/SVG at N, alpha mask, palette fit + snap
  synth.py        # variations + captions + canonical defaults; split logic
  vocab.py        # vocabulary + CLIP word table (offline) → data/wordvec_{d}.npy
  tokenizer.py    # python twin of the in-sheet tokenizer
  model.py        # PyTorch, Pixcell-supported ops only
  train.py
  reference.py    # numpy float64 forward pass = the spec
  pixcell/        # ir.py, layout.py, emit.py, xlsx.py, sheets.py
  oracles/        # hyperformula + libreoffice runners
  evaluate.py     # metrics → artifacts/metrics/*.json
scripts/ fetch_data.sh, preview_pixelate.py, sheets_probe.py, verify_all.sh, check_claims.py, …
```
