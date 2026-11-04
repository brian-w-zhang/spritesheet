# 10 · How SpriteSheet works, explained four ways

The same system at four depths. Read top-down and stop when you have enough.
1. **The tweet** (anyone)
2. **The 30-second version** (the professor watching our 5-minute video)
3. **The walkthrough** (teammates who need to answer questions about it)
4. **The builder's view** (whoever extends the code: real formulas, real files)

Exact numbers live in `artifacts/metrics/` and `docs/09-experiment-log.md`; this doc explains ideas.

---

## 1. The tweet

> We put a text-to-image AI inside a Google Sheet. Type "a tiny blue rocket" into a cell, and a 32×32 pixel-art rocket appears in the cells below. No code runs: no scripts, no add-ons, no API. Just ~a hundred thousand spreadsheet formulas doing the neural network's maths, and we checked it matches the real model to 12+ decimal places.

Short form: *A neural network that lives entirely in spreadsheet formulas. Type a prompt, get pixel art.*

---

## 2. The 30-second version (for the professor)

- **What:** a small text-to-image neural network, trained by us, then *compiled* into Google Sheets formulas by our compiler, **Pixcell**. The spreadsheet *is* the model: every number the network computes is a cell, and every operation is a formula.
- **Why it's hard:** spreadsheets were never meant to run neural networks. There is a 20M-cell limit, recalculation is slow, there are no loops, and different spreadsheet engines disagree on edge cases. So the model must be tiny, and every operation must be rewritten as plain cell formulas that compute *exactly* what the original model computes.
- **How we know it's right:** a reference implementation in numpy is the spec. We compute the same prompts in Google Sheets and in two other spreadsheet engines, and compare every intermediate cell: values agree to about 1e-14, and the final pixels are identical.
- **How good is it:** we measure it automatically (pixel accuracy against exact targets, attribute checks, CLIP scores) and compare against baselines and against SpriteX32, a GPU-based 32×32 text-to-sprite model hundreds of times larger.
- **Courses it draws on:** machine learning and deep learning (model design, training), compilers (lowering a computation graph into another "instruction set", here spreadsheet formulas), and algorithms and numerical computing (layouts, exact arithmetic, floating-point agreement).

---

## 3. The walkthrough (for teammates)

### The one-sentence mental model
A neural network is a big pile of multiply-and-add. A spreadsheet is very good at multiply-and-add (`SUMPRODUCT`). So if we lay out the network's numbers as cells and write each multiply-and-add as a formula, the spreadsheet computes the network for us whenever the prompt cell changes.

### What happens when you type a prompt (left to right through the tabs)
1. **Prompt tab.** You type "a blue cat" in B2 and a seed in B3.
2. **Tokens.** Formulas lowercase the text, strip punctuation, split it into words, and look each word up in the `Vocab` tab (12k words). Each word becomes a row number.
3. **WordVec: the "dictionary of meanings."** Each vocabulary word has 64 numbers describing what it means. We got these once, offline, from CLIP (a big OpenAI model that understands text and images), then shrank them to 64 numbers per word. This is why words the model never saw in training, like "puppy", still land near related words like "dog".
4. **Encoder tabs (X, Q, K, Scores, Softmax, Ctx, …).** A tiny *attention* layer, the same mechanism as in GPT, lets words influence each other, so "blue" attaches to "cat". The result is one summary vector of 64 numbers: the meaning of the whole prompt.
5. **Latent: picking a drawing style.** An apple can be drawn by 7 artists (Google, Twitter, Microsoft ×3, OpenMoji, pixel art). If you don't say which, the sheet rolls a die using your seed: an exact integer random-number formula, so the same seed always gives the same style.
6. **Decoder tabs (Dense, Up_1/Conv_1, …, Conv_3).** The meaning + style is expanded into a 4×4 grid of features, then repeatedly doubled (8×8 → 16×16 → 32×32). Each step looks at 3×3 neighbourhoods (a *convolution*), the standard way image models "paint".
7. **Logits → Pixels → Hex → Prompt.** Each of the 1,024 pixels gets a score for each of 33 colours; the highest wins (`MATCH(MAX(...))`). The winning colour's hex code paints a square cell on the Prompt tab.

### Where the "AI" came from
- **Data:** about 1,400 everyday concepts, each drawn by up to 6 emoji artists, plus 18,000 captioned pixel-art sprites (PixelGPT). We also generate variations in code ("a **blue** apple", "a **tiny** apple **in the corner**", "**two** apples", "**upside down**"), each with an exactly known correct image.
- **Training:** on a laptop GPU (Apple M4), in PyTorch, for tens of minutes. The trained numbers (weights) are then written into the `W_*` tabs.
- **Why tiny:** the sheet has to recalculate everything on every prompt. A random-weight test model of this shape took about 11 s per prompt in real Google Sheets, so we chose sizes that keep recalculation in the tens of seconds.

### How we know the sheet is right (the part to emphasise in Q&A)
- **One spec:** `reference.py` (numpy, 64-bit floats) is the definition of the model.
- **Three spreadsheet engines** compute the same prompts: Google Sheets, HyperFormula (an open-source spreadsheet engine), and LibreOffice. We compare every tab to the spec. Numbers agree to about 1e-14 relative; the style choice and all 1,024 pixel colours are identical.
- **Traps the checks caught:** a word-splitting formula that broke on long prompts; a random-number formula that always picked style 1 for small seeds; a backslash that broke one spreadsheet engine; and LibreOffice silently reusing stale values instead of recalculating.

### Honest limitations (say these before someone else does)
- 32×32 only (48 and 64 are next, as separate models and sheets).
- Prompts work best on concepts it trained on. Brand-new objects come out as plausible blends, not real drawings.
- Recalculation takes seconds, not milliseconds; the same model in numpy is about 1000× faster. That's the price of the medium.
- Captions are partly synthetic (generated from emoji names and keywords).

---

## 4. The builder's view

### Files
- `src/spritesheet/reference.py`: the spec (forward pass, LCG, style sampling).
- `src/spritesheet/model.py`: the PyTorch twin used for training; `export()` maps it to reference layout.
- `src/spritesheet/pixcell/emit.py`: `compile_model(w, cfg, vocab, palette, render=…)` → `Workbook` (`{sheet: {(row, col): value | "=formula"}}`).
- `pixcell/xlsx.py` and `pixcell/sheets.py`: writers. `pixcell/lint.py` enforces the allowed function subset.
- `oracles/hyperformula.py` and `oracles/libreoffice.py`: local engines. `pixcell/gates.py` + `scripts/verify_*.py`: the A1–A4 gates.

### Real formulas from a compiled workbook (copied from `compile_model` output)
| Tab | Example cell | What it is |
| --- | --- | --- |
| Tokens | `=TRIM(MID(SUBSTITUTE(MID(B1,1,100)," ",REPT(" ",100)),1,100))` | first word of the remaining text |
| X | `=INDEX(WordVec!$A$1:$BL$12119,Tokens!$D$1,1)+W_pos!A1` | word vector lookup + position |
| Scores | `=SUMPRODUCT(Q!A2:BL2,K!A1:BL1)*$N$1+(1-Tokens!$E$1)*-10000` | attention score, padding masked |
| Softmax | `=Q1/O1` (exp cells / row sum; exps are `EXP(x − rowmax)`) | attention weights |
| Ctx | `=SUMPRODUCT(Softmax!A1:L1,VT!A1:L1)` | weighted sum of values (V stored transposed) |
| Latent | `=SUMPRODUCT(Tokens!$E$1:$E$12,H!A1:A12)/Tokens!$F$1` | masked mean → prompt vector c |
| Dense | `=MAX(0,SUMPRODUCT(Latent!$A$8:$BT$8,W_d!A1:BT1)+W_d!BU1)` | z → 4×4×C₀ grid (ReLU) |
| Up_1 | `=Dense!A1` | nearest-neighbour upsample = plain references |
| Conv_1 | `=MAX(0,SUMPRODUCT(Up_1!A1:KB3,W_c1!A1:KB3)+FiLM!$A$1)` | a whole 3×3×Cin convolution in **one** SUMPRODUCT |
| Logits | `=SUMPRODUCT(Conv_3!A1:AF1,W_out!A1:AF1)+W_out!AG1` | 1×1 conv → 33 colour scores |
| Pixels | `=MATCH(MAX(Logits!A1:AG1),Logits!A1:AG1,0)-1` | argmax (first max) |
| Prompt | `=SPARKLINE(1,{"charttype","bar";"max",1;"color1",Hex!A1})` | paints the cell with the computed colour (Sheets only) |

### The key layout trick (convolution as one SUMPRODUCT)
An H×W×C activation is stored as an `H` × `W·C` block (channels interleaved) inside a one-cell zero border. A 3×3 window over all input channels is then one *contiguous* 3 × (3·Cin) rectangle. Store each output channel's kernel in the same shape, and the convolution becomes `SUMPRODUCT(window, kernel) + bias` in each cell, with short formulas and no helper cells.

### Getting it into Google Sheets
1. **Auth:** a service account (`~/.config/spritesheet/sa-key.json`), and a spreadsheet *owned by Brian and shared with it as Editor* (`~/.config/spritesheet/config.json`). A service account cannot create spreadsheets itself.
2. `sheets.push(wb)` deletes old tabs, creates all tabs, then writes every cell. Text (the vocabulary) is written `RAW`, so "true" or "1/2" are never parsed as booleans or dates. Numbers and formulas go `USER_ENTERED`. Chunks are ≤1.8 MB, at about 1 request/s, with backoff on 429/5xx.
3. It formats the Prompt tab: 12 px square rows and columns, gridlines hidden.
4. `set_prompt_and_read(prompt, seed)` writes B2/B3, then polls `Pixels` until it stops changing. That gives the latency number.
5. Gate A4 compares those pixels to the spec.

### Rules that keep the three engines agreeing
Only `SUMPRODUCT, INDEX, MATCH(…,0), MAX, MIN, SUM, EXP, IF, IFERROR, MOD, INT, ABS`, arithmetic, and the text functions `LOWER/SUBSTITUTE/TRIM/MID/REPT` (tokenizer only). One scalar formula per cell. No LET/LAMBDA/array spill/volatile functions. Weights are rounded to 15 significant digits so they survive file formats exactly. Softmax always subtracts the row max. See doc 05.

---

## Demo controls (Prompt tab)
| Cell | What it does |
| --- | --- |
| **B2** | the prompt: type, then press Enter |
| **B3** | seed: picks the drawing style (Noto / Twemoji / OpenMoji / Fluent 3D / Fluent colour / Fluent flat / pixel art) when the prompt doesn't name one. Any number works; it is reduced to an integer 1…2,147,483,646 (`MOD(INT(ABS(seed)),2147483646)+1`). Same prompt + seed → same image; neighbouring seeds are unrelated. It is the only randomness in the model |
| **B4** | the drawing style the sheet sampled for this seed, by name (read-only; steer the style by writing it in the prompt, e.g. "a cat, 3d") |
| **C5:AH36** | the 32×32 sprite: square 12 px cells holding class ids, coloured by one conditional-format rule per palette colour, digits hidden, so the image is seamless |

**Trigger:** press Enter on the prompt cell. (Clickable buttons in Google Sheets need Apps Script, which would break "no code runs".)
**Latency (honest number for the video):** median 30.6 s per prompt in real Google Sheets for the 309,648-param main model (`artifacts/metrics/parity_sheets.json`). Show it real-time once, then speed up the wait in editing with a label.
