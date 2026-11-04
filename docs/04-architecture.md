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
Lowercase; strip punctuation with a fixed `SUBSTITUTE` chain (must include `* ? ~`, which are `MATCH` wildcards); split into at most `L = 12` words with a per-word `MID/SUBSTITUTE/REPT/TRIM` "first token + remainder" split (D-30); `MATCH(word, Vocab, 0)`, with `IFERROR` → `<unk>` and padding with `<pad>`. A simple suffix rule is optional (strip a trailing "s" when the word isn't in the vocabulary), with the *same* rule in Python. `tokenizer.py` must match the formulas exactly (gate A1). See doc 05.

### [W] CLIP word table (offline distillation, frozen)
- **Vocabulary V ≈ 8–20k words:** every word in training captions and concept synonyms, plus common English words taken from CLIP's own BPE vocabulary (whole-word tokens, alphabetic, by merge rank). This gives an open vocabulary at no licence cost (MIT).
- **Vector per word:** the CLIP text-encoder pooled output for the bare word (try "a pixel art {word}" as an alternative, then pick by validation), L2-normalised, then PCA to d = 64 fitted on the vocabulary. Store it as a `Vocab` tab (words) and a `WordVec` tab (V×d values) of ≈0.5–1.3M cells. **Measure the Sheets upload and recalc cost (R1).** If too heavy, use d = 32 or a smaller V.
- Frozen during training, so unseen-but-in-table words ("puppy") land near trained concepts ("dog"). Ablation C3 measures this.
- *Optional auxiliary loss:* the encoder's pooled output should predict the (PCA'd) CLIP embedding of the *whole* prompt. This distils CLIP's phrase understanding into the small encoder.

### [E] Attention encoder (trained)
Word vectors (frozen) + learned positional embedding → 1–2 transformer layers (d = 64, 2 heads, ReLU MLP, no LayerNorm or a simple formula-friendly one) → masked mean-pool → condition vector `c`. Softmax is always `EXP(x − MAX)`. Every op used must exist in both `reference.py` and the Pixcell emitter.

### [Z] Seeded style / noise latent
- **Style id:** if the prompt names a style ("3d", "twemoji", "pixel art"), the encoder's style head picks it. Otherwise the in-sheet **MINSTD LCG** (`x ← MOD(48271·x, 2147483647)`, exact below 2^53; two steps from the sanitised seed, D-31) samples from the style prior. Style embedding → concatenated into the decoder input. This replaces the blurry average of 7 drawings with a crisp choice.
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

**As emitted by Pixcell** (`src/spritesheet/pixcell/emit.py`; one scalar formula per cell; weight matrices are stored
transposed with the bias as an extra last column, so every linear unit is `SUMPRODUCT(x_row, W_row) + bias`):

| Tab | Contents (1-based) |
| --- | --- |
| `Prompt` | A2/B2 prompt (text), A3/B3 seed, B4 = sampled style. 32×32 display at C6:AH37, square 12 px cells, gridlines hidden; each cell `=SPARKLINE(1,{"charttype","bar";"max",1;"color1",Hex!…})` (Sheets build only; the local oracles compile with `render=False`). |
| `Vocab`, `WordVec` | A1:A{V} words as TEXT (row 1 `<pad>`, row 2 `<unk>`); V×d word vectors. |
| `Tokens` | A1 clean; rows 1..L: B rest_i, C word_i, D id_i, E mask_i; F1 = MAX(1, Σmask); H1:H32 punctuation characters as TEXT (D-32). |
| `W_pos`, `X` | X[t,k] = `INDEX(WordVec,id_t,k)` + pos[t,k] (L×d). |
| `W_q/k/v`, `Q`, `K`, `VT` | Q, K as L×d; V stored transposed (d×L) so every reduction is row·row. |
| `Scores` | S[t,u] = `SUMPRODUCT(Q_t,K_u)*scale + (1-mask_u)*-10000`; scale in Scores!N1. |
| `Softmax` | P in cols 1..L, row max col L+2, row sum col L+3, `EXP(S−max)` cols L+5… |
| `Ctx`, `W_o`, `Resid` | O = P·V; A = X + O·Wo + bo. |
| `W_1`, `MLP`, `W_2`, `H` | hidden = MAX(0, A·W1+b1); H = A + hidden·W2 + b2. |
| `Latent` | row 1 c (masked mean) · row 2 style logits · A3 max, B3 sum · row 4 exp · row 5 p · row 6 cum (cum_i = cum_{i−1}+p_i) · A7 s0, B7 s1, C7 u, D7 style · row 8 z = [c, INDEX(W_emb, style, ·)]. |
| `W_d`, `Dense` | G0 as 4 × (4·C0), channels interleaved: cell (y, x·C+c). |
| `W_f{i}`, `FiLM` | FiLM!row i = c·F_i + bc_i per output channel (one shift per channel, referenced by every conv cell). |
| `Up_{i}` | nearest ×2 of the previous grid as cell references, inside a zero border: (2H+2) × ((2H+2)·Cin). |
| `W_c{i}`, `Conv_{i}` | W_c{i} rows 3o+1..3o+3 = the 3×(3·Cin) kernel of output channel o; `Conv_{i}` cell (y, x·Cout+o) = `MAX(0, SUMPRODUCT(Up_i 3×(3·Cin) window, kernel block) + FiLM)`. |
| `W_out`, `Logits` | N × (N·(K+1)); 1×1 conv = `SUMPRODUCT(Conv_3 Cin-row segment, W_out row k) + b`. |
| `Pixels`, `Palette`, `Hex` | class = `MATCH(MAX(r),r,0)-1` (0 = background); Palette A1 = `#f6f4ef`, A2.. = palette; Hex = `INDEX(Palette, class+1)`. |
| `About` | compiler version, model hash (B2), config, vocab size. |

Size at the v32 shape (V=5000, d=32, h=64, S=8, e=8, C=64/48/32/24, K=32): 405,683 cells, 127,057 formulas,
4.76M formula characters (`artifacts/metrics/spikes.json`).

### v2 workbook layout (masked discrete diffusion; `pixcell/emit_mdm.py`, spec `mdm.ref_generate`)
The front is emitted by the same code as v1 (`emit.emit_front`): Prompt inputs (B2 prompt, B3 seed, B4 style
name), `Vocab`, `WordVec`, `Tokens`, the encoder tabs and `Latent`, whose row 8 is `cond = [c, style_emb[style]]`.
Then one shared set of weight tabs, referenced by every step: `W_tok` ((K+2)×E token embedding, row x+1 = class x,
row K+2 = MASK), `W_c1…W_c6` (kernel layout as v1), `W_f1…W_f6` → `FiLM` row i = `cond·F_i + f_i` (computed once:
it does not depend on the step), `W_p2`, `W_p1` (1×1 projections, transposed), `W_out`. Then, per step t = 1…T,
sixteen tabs `S{t}_…`:

| Tab | Contents |
| --- | --- |
| `S{t}_Emb` | zero-bordered (N+2)×((N+2)·E): `INDEX(W_tok, x+1, k)` with x from `S{t-1}_Canvas` (step 1: the constant MASK row) |
| `S{t}_A1` | N×(N·C1): `MAX(0, SUMPRODUCT(3×3E window, kernel) + FiLM1)` |
| `S{t}_Pool1`, `S{t}_Pool2` | bordered avg2: `(a+b+c+d)/4` |
| `S{t}_A2`, `S{t}_A3`, `S{t}_A4` | convs at 16×16 / 8×8 (A3 is written straight into a bordered layout, because A4 convolves it; v2b adds `+GMap!…` after the ReLU, where `GMap` = cond·W_g + bg is computed once, D-44) |
| `S{t}_P2`, `S{t}_P1` | 1×1 projection per cell before upsampling: `SUMPRODUCT(channel segment, W_p row)` |
| `S{t}_Up2`, `S{t}_Up1` | nearest ×2 as plain references, bordered |
| `S{t}_U2`, `S{t}_U1` | `MAX(0, conv + skip (A2 / A1) + FiLM5/6)` |
| `S{t}_Logits` | N×(N·(K+1)) |
| `S{t}_Conf` | cols 1..N max; N+2.. conf = `1/SUMPRODUCT(EXP(l − max))`; 2N+3.. argmax = `MATCH(max, l, 0)−1` (first max) |
| `S{t}_Canvas` | `IF(old=MASK, IF(conf>=tau_t, argmax, MASK), old)`; t = T: `IF(old=MASK, argmax, old)`; tau inlined from `cfg["tau"]` |

`Pixels` = `S{T}_Canvas`; `Palette`, `Hex` and the display are as in v1. **Filmstrip:** below the display, one 32×32
thumbnail per step canvas (`=S{t}_Canvas!…`), two per row, on the same 12 px grid, each under a "step t" label; MASK
(K+1) gets its own conditional-format colour `#b9b4c2` ("unknown"). The xlsx writer applies the same
colour rules (openpyxl `CellIsRule`, number format `;;;`), so the .xlsx also renders in Excel / LibreOffice.
Size and Sheets cost per step count: `artifacts/metrics/spikes_v2.json`.

### v3 (24×24, learned variants; D-51)
Same network and Pixcell layout as v1, with these differences:
- **Canvas:** `N=24`, so the dense layer outputs a 3×3 grid, upsampled three times to 24×24.
- **Kernels:** optional per-block kernel sizes `k1..k3` (odd; zero border = k//2).
- **Colours:** K=48 palette classes.
- **Variants:** S=16 unnamed variants replace the 7 named styles. The variant head and seeded pick are the same formulas as v1's style; the Prompt tab shows no style row (`cfg.variants`).
- **Training:** `src/spritesheet/train_v3.py`, using winner-takes-all variant assignment and word-vector noise.
- **Data:** `src/spritesheet/data/v3.py`.

### v3-P (PatchGPT; D-52; `pixcell/emit_p.py`, spec `reference_p.forward`)
- **Output:** 24×24 = 8×8 patches of 3×3, each copied from a fixed 1024-entry patch dictionary (`W_D`). No conv decoder.
- **Prior:** causal transformer over 75 rows (12 prompt rows, then 63 image-token rows); row L−1+t predicts patch t. Seeded Gumbel-max picks each patch.
- **Tabs:**
  - `Seed` (s0, s1);
  - `Inputs` (word vectors `WV`, input rows `X0`; image rows read `Out!Codes`);
  - per block `Blk{l}_QKV` (1/rms, Q, K, Vᵀ), `Blk{l}_Att` (one N×N exp-weight block per head, plus row sums), `Blk{l}_Ctx` (context, residual) and `Blk{l}_MLP` (1/rms, hidden, residual);
  - `Out` (Codes, final 1/rms);
  - `Score` (T×K Gumbel scores);
  - `Pixels` (patch lookup), then `Hex` / `Display` as in v1;
  - with guidance, the unconditional chain `U_X0`, `U{l}_*`, `U_Out` (LU).

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
