# 08 · Module interfaces (contract between data, model, reference, compiler)

All arrays are numpy. N = 32 (resolution parameter everywhere).

## Data → model (`src/spritesheet/data/`)
- `data/build/palette.json`: `{"n": 32, "colors": ["#rrggbb", ...]}` with K colours. **Class id 0 = background/transparent; class id i (1..K) = colors[i-1].**
- `data/build/concepts.json`: list of `{"id": str, "name": str, "keywords_train": [..], "keywords_test": [..], "group": str, "styles": [style names available]}`.
- `data/build/bases_{N}.npz`: `imgs` uint8 [M, N, N] class ids (canonical size, centred), `concept` int [M], `style` int [M], plus `small` uint8 [M, N, N] (the same base rendered small, for size/position/count variants).
- `data/build/styles.json`: ordered list of style names (the style id is the index). Style phrases are in `synth.STYLE_WORDS`.
- `synth.Sampler(split, seed)` → `.batch(B)` returns `(captions: list[str], targets: uint8 [B,N,N], style_ids: int [B], meta: list[dict])`. Deterministic for the same seed. `meta` holds concept, style and attributes, for the attribute-accuracy metrics.
- `synth.eval_set(name)` returns a fixed list of `(caption, target [N,N], style_id, meta)` for each split: `train_sample`, `heldout_combo`, `heldout_phrase`, `unseen_words`, `pixelgpt_test`, `underspecified`.

## Text
- `tokenizer.py`: `normalize(prompt) -> list[str]` (≤ L = 12 words) and `encode(prompt, vocab_index) -> list[int]` (length L; PAD = 1, UNK = 2, vocab ids are 1-based **row numbers in the Vocab tab**; row 1 = `<pad>`, row 2 = `<unk>`).
- `data/build/vocab.txt` (one word per line, line 1 `<pad>`, line 2 `<unk>`) and `data/build/wordvec.npy` float64 [V, d] (rows aligned with vocab.txt; PAD row = zeros).

## Model weights (`artifacts/model/v32/weights.npz` + `config.json`)
float64, already rounded to 15 significant digits. Names and shapes:
```
wordvec [V,d] (frozen)   pos [L,d]
Wq,Wk,Wv,Wo [d,d]  bq,bk,bv,bo [d]          # x @ W + b convention
W1 [d,h] b1 [h]  W2 [h,d] b2 [d]            # MLP, ReLU
Ws [d,S] bs [S]                             # style head
style_emb [S,e]
Wd [d+e, 4*4*C0] bd [4*4*C0]                # dense → grid, reshaped (y, x, c) row-major
for each conv block i=1..3: Wc{i} [3,3,Cin,Cout] bc{i} [Cout], Fi [d, Cout] (FiLM shift from c, added before ReLU)
Wout [C3, K+1] bout [K+1]                   # 1x1 conv to logits
```
`config.json`: `{N, L, V, d, h, S, e, C0, C1, C2, C3, K, attn_scale}`.

## Reference forward (`reference.py`, the spec)
1. ids → X[t] = wordvec[id_t] + pos[t]; mask m_t = (id_t != PAD).
2. Q = X Wq + bq, K = X Wk + bk, V = X Wv + bv. S[t,u] = (Q_t·K_u)·attn_scale + (m_u ? 0 : −1e4). P = softmax over u (subtract the row max). O = P V. A = X + O Wo + bo.
3. H = A + relu(A W1 + b1) W2 + b2.
4. c = Σ_t m_t H_t / Σ_t m_t (masked mean).
5. style logits = c Ws + bs → p = softmax; u = LCG(seed) (s0 = MOD(INT(ABS(seed)), 2147483646) + 1; u = MOD(48271·s0, 2147483647)/2147483647). The cumulative sum is sequential: cum_i = cum_{i−1} + p_i. style = 1 + #{i : cum_i < u}, clamped to S. **A style word in the prompt is handled by the model itself (peaked p).**
6. z = [c, style_emb[style]] → dense → G0 [4,4,C0] (no activation, then ReLU).
7. For i in 1..3: up = nearest ×2; conv3×3 (zero pad 1) + bc_i + (c · F_i) → ReLU.
8. logits = G3 Wout + bout [N,N,K+1]; class = argmax (first max).

Reductions are dot products. The engines may sum in a different order, so parity is relative ≤ 1e-12, except for discrete outputs (style, class), which must be identical; also report margins.

## Compiler (`pixcell/`)
`compile_model(weights, config, prompt_default) -> Workbook` (an in-memory dict `{sheet: {(row, col): value | "=formula"}}` + layout metadata), then `to_xlsx(wb, path)`, `to_hyperformula_json(wb)` and `to_sheets(wb, spreadsheet_id)`. Inputs: `Prompt!B2` (text), `Prompt!B3` (seed). Output: `Pixels` tab class ids [N×N] and the `Prompt` render area.
