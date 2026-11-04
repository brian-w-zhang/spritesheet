"""Pixcell emitter: SpriteSheet model -> Workbook of per-cell scalar formulas (layout: docs/04 "Workbook layout").

Every step of reference.forward is one formula per cell, using only the allowed subset (docs/05):
SUMPRODUCT, INDEX, MATCH(…,0), MAX, MIN, SUM, EXP, IF, IFERROR, MOD, INT, ABS, + - * /, and for the
tokenizer LOWER, SUBSTITUTE, TRIM, MID, REPT. SPARKLINE appears only in the Prompt display (render=True).
"""
import json
import os

import numpy as np

from .. import tokenizer as T
from ..reference import NEG, M31, LCG_A
from .workbook import Block, Text, Workbook, a1, col_letter, ref, rng

VERSION = "pixcell-0.1"
BACKGROUND = "#f6f4ef"
DISPLAY_R0, DISPLAY_C0 = 6, 3          # Prompt display block top-left (row 6, column C)
GRID0 = 4                               # dense layer output is 4x4xC0 (reference.decode)


# ---------------------------------------------------------------- tokenizer formulas (shared with gate A1)
PUNCT_COL = 8   # Tokens!H1:H<n> hold the punctuation characters as TEXT cells (HyperFormula parses "\" as an
                # escape inside string literals, so literals are not portable; cell references are).


def put_punct(wb: Workbook, sheet="Tokens"):
    for k, p in enumerate(T.PUNCT):
        wb.set(sheet, 1 + k, PUNCT_COL, Text(p))
    wb.add_block("Tokens.punct", sheet, 1, PUNCT_COL, len(T.PUNCT), 1)


def f_clean(prompt_ref: str, sheet_prefix="") -> str:
    f = f"LOWER({prompt_ref})"
    for k in range(len(T.PUNCT)):
        f = f'SUBSTITUTE({f},{sheet_prefix}${col_letter(PUNCT_COL)}${1 + k}," ")'
    return "=" + f


def f_rest(i: int, clean_ref: str, prev_rest: str = "", prev_word: str = "") -> str:
    if i == 0:
        return f"=TRIM({clean_ref})"
    return f'=TRIM(SUBSTITUTE({prev_rest},{prev_word},"",1))'


def f_word(rest_ref: str) -> str:
    W = T.W
    return f'=TRIM(MID(SUBSTITUTE(MID({rest_ref},1,{W})," ",REPT(" ",{W})),1,{W}))'


def f_id(word_ref: str, vocab_range: str) -> str:
    return f'=IF({word_ref}="",{T.PAD},IFERROR(MATCH({word_ref},{vocab_range},0),{T.UNK}))'


# ---------------------------------------------------------------- helpers
def model_hash(w, cfg, vocab_words, palette_hex) -> str:
    import hashlib
    import json
    h = hashlib.sha256()
    for k in sorted(w):
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(w[k], np.float64)).tobytes())
    h.update(json.dumps(cfg, sort_keys=True).encode())
    h.update("\n".join(vocab_words).encode())
    h.update("\n".join(palette_hex).encode())
    return h.hexdigest()[:16]


def _wtab(wb: Workbook, name: str, M, bias=None) -> Block:
    """Weight tab with rows = M rows, plus an optional bias column at the end. Values only."""
    M = np.asarray(M, dtype=np.float64)
    ncol = M.shape[1] + (1 if bias is not None else 0)
    b = wb.add_block(name, name, 1, 1, M.shape[0], ncol)
    full = M if bias is None else np.concatenate([M, np.asarray(bias, np.float64)[:, None]], axis=1)
    wb.put_matrix(b, full)
    return b


def compile_model(w, cfg, vocab_words, palette_hex, default_prompt="a red apple", default_seed=1,
                  render=False, style_names=None) -> Workbook:
    """Compile weights+config into a Workbook. render=True adds the coloured display + stage bar (Google Sheets).
    style_names (optional): list of S style names shown in Prompt!B4; defaults to data/build/styles.json if present."""
    L, d, h, S, e, K = T.L, cfg["d"], cfg["h"], cfg["S"], cfg["e"], cfg["K"]
    V = len(vocab_words)
    C = [cfg["C0"], cfg["C1"], cfg["C2"], cfg["C3"]]
    assert cfg.get("L", L) == L, "config L must match tokenizer.L"
    assert w["wordvec"].shape == (V, d), (w["wordvec"].shape, V, d)
    assert vocab_words[0] == "<pad>" and vocab_words[1] == "<unk>"
    assert len(palette_hex) == K, (len(palette_hex), K)
    N = GRID0 * 8
    wb = Workbook()
    wb.meta.update({"config": dict(cfg), "version": VERSION, "render": render, "N": N, "V": V})

    # ---- Prompt (inputs) first, so it is the first tab
    P = "Prompt"
    wb.set(P, 1, 1, Text("SpriteSheet"))
    wb.set(P, 1, 2, Text("Type a prompt in B2 and press Enter. Change the seed (B3) to re-roll the drawing style."))
    wb.set(P, 2, 1, Text("Prompt"))
    wb.set(P, 2, 2, Text(default_prompt))
    wb.set(P, 3, 1, Text("Seed"))
    wb.set(P, 3, 2, float(default_seed))
    wb.set(P, 4, 1, Text("Style (picked)"))
    PROMPT, SEED = "Prompt!$B$2", "Prompt!$B$3"
    wb.meta["inputs"] = {"prompt": (P, 2, 2), "seed": (P, 3, 2)}

    # ---- Vocab + WordVec (values)
    vb = wb.add_block("Vocab", "Vocab", 1, 1, V, 1)
    for i, word in enumerate(vocab_words):
        wb.set("Vocab", 1 + i, 1, Text(word))
    VOCAB = vb.range("Vocab", absolute=True)
    wv = _wtab(wb, "WordVec", w["wordvec"])
    WV = wv.range("WordVec", absolute=True)

    # ---- Tokens: A1 clean; rows 1..L: B rest, C word, D id, E mask; F1 denom
    Tk = "Tokens"
    wb.add_block("Tokens.clean", Tk, 1, 1, 1, 1)
    rest_b = wb.add_block("Tokens.rest", Tk, 1, 2, L, 1)
    word_b = wb.add_block("Tokens.word", Tk, 1, 3, L, 1)
    ids_b = wb.add_block("Tokens.ids", Tk, 1, 4, L, 1)
    mask_b = wb.add_block("Tokens.mask", Tk, 1, 5, L, 1)
    wb.add_block("Tokens.denom", Tk, 1, 6, 1, 1)
    put_punct(wb, Tk)
    wb.set(Tk, 1, 1, f_clean(PROMPT))
    for t in range(L):
        r = 1 + t
        wb.set(Tk, r, 2, f_rest(t, "$A$1", f"B{r - 1}", f"C{r - 1}"))
        wb.set(Tk, r, 3, f_word(f"B{r}"))
        wb.set(Tk, r, 4, f_id(f"C{r}", VOCAB))
        wb.set(Tk, r, 5, f"=IF(D{r}={T.PAD},0,1)")
    wb.set(Tk, 1, 6, f"=MAX(1,SUM({mask_b.range()}))")
    MASK = mask_b.range(Tk, absolute=True)
    DENOM = "Tokens!$F$1"

    # ---- encoder weights
    pos = _wtab(wb, "W_pos", w["pos"])
    Wq = _wtab(wb, "W_q", w["Wq"].T, w["bq"])
    Wk = _wtab(wb, "W_k", w["Wk"].T, w["bk"])
    Wv = _wtab(wb, "W_v", w["Wv"].T, w["bv"])
    Wo = _wtab(wb, "W_o", w["Wo"].T, w["bo"])
    W1 = _wtab(wb, "W_1", w["W1"].T, w["b1"])
    W2 = _wtab(wb, "W_2", w["W2"].T, w["b2"])

    def lin(b: Block, i: int, xrow: str, n: int) -> str:
        """SUMPRODUCT(x, W^T row i) + b_i for a weight tab b with n inputs and a bias column."""
        return f"SUMPRODUCT({xrow},{b.row_range(i, 0, n - 1, b.sheet)})+{b.ref(i, n, b.sheet)}"

    # X = wordvec[id] + pos
    X = wb.add_block("X", "X", 1, 1, L, d)
    wb.fill(X, lambda t, k: f"=INDEX({WV},Tokens!$D${1 + t},{k + 1})+{pos.ref(t, k, 'W_pos')}")
    Xrow = lambda t: X.row_range(t, sheet="X")
    Q = wb.add_block("Q", "Q", 1, 1, L, d)
    wb.fill(Q, lambda t, j: "=" + lin(Wq, j, Xrow(t), d))
    Kb = wb.add_block("K", "K", 1, 1, L, d)
    wb.fill(Kb, lambda t, j: "=" + lin(Wk, j, Xrow(t), d))
    VT = wb.add_block("VT", "VT", 1, 1, d, L)                     # V transposed: VT[k, u]
    wb.fill(VT, lambda k, u: "=" + lin(Wv, k, Xrow(u), d))

    # scores S[t,u] = (Q_t . K_u) * scale + (1 - m_u) * NEG ; scale in Scores!{L+2}1
    Sc = "Scores"
    scale_c = (1, L + 2)
    wb.set(Sc, *scale_c, float(cfg["attn_scale"]))
    wb.add_block("Scores.scale", Sc, 1, L + 2, 1, 1)
    SCALE = ref(*scale_c, absolute=True)
    Sb = wb.add_block("Scores", Sc, 1, 1, L, L)
    wb.fill(Sb, lambda t, u: f"=SUMPRODUCT({Q.row_range(t, sheet='Q')},{Kb.row_range(u, sheet='K')})*{SCALE}"
                             f"+(1-Tokens!$E${1 + u})*{NEG:g}")

    # softmax: P in cols 1..L, max col L+2, sum col L+3, exp cols L+5..2L+4
    Sm = "Softmax"
    Pb = wb.add_block("P", Sm, 1, 1, L, L)
    Mb = wb.add_block("Softmax.max", Sm, 1, L + 2, L, 1)
    Zb = wb.add_block("Softmax.sum", Sm, 1, L + 3, L, 1)
    Eb = wb.add_block("Softmax.exp", Sm, 1, L + 5, L, L)
    wb.fill(Mb, lambda t, _: f"=MAX({Sb.row_range(t, sheet=Sc)})")
    wb.fill(Eb, lambda t, u: f"=EXP({Sb.ref(t, u, Sc)}-{Mb.ref(t, 0)})")
    wb.fill(Zb, lambda t, _: f"=SUM({Eb.row_range(t)})")
    wb.fill(Pb, lambda t, u: f"={Eb.ref(t, u)}/{Zb.ref(t, 0)}")

    # context O = P V  (row . row against VT)
    O = wb.add_block("Ctx", "Ctx", 1, 1, L, d)
    wb.fill(O, lambda t, k: f"=SUMPRODUCT({Pb.row_range(t, sheet=Sm)},{VT.row_range(k, sheet='VT')})")
    # A = X + O Wo + bo
    A = wb.add_block("A", "Resid", 1, 1, L, d)
    wb.fill(A, lambda t, k: f"={X.ref(t, k, 'X')}+" + lin(Wo, k, O.row_range(t, sheet="Ctx"), d))
    # hidden = relu(A W1 + b1); H = A + hidden W2 + b2
    Hd = wb.add_block("MLP", "MLP", 1, 1, L, h)
    wb.fill(Hd, lambda t, j: f"=MAX(0,{lin(W1, j, A.row_range(t, sheet='Resid'), d)})")
    H = wb.add_block("H", "H", 1, 1, L, d)
    wb.fill(H, lambda t, k: f"={A.ref(t, k, 'Resid')}+" + lin(W2, k, Hd.row_range(t, sheet="MLP"), h))

    # ---- Latent: c, style head, LCG, sampled style, z
    Lt = "Latent"
    Ws = _wtab(wb, "W_s", w["Ws"].T, w["bs"])
    Wemb = _wtab(wb, "W_emb", w["style_emb"])
    cb = wb.add_block("c", Lt, 1, 1, 1, d)
    wb.fill(cb, lambda _, k: f"=SUMPRODUCT({MASK},{H.col_range(k, sheet='H')})/{DENOM}")
    CROW = cb.row_range(0, sheet=Lt, absolute=True)
    sl = wb.add_block("style_logits", Lt, 2, 1, 1, S)
    wb.fill(sl, lambda _, s: "=" + lin(Ws, s, CROW, d))
    smax = wb.add_block("style.max", Lt, 3, 1, 1, 1)
    ssum = wb.add_block("style.sum", Lt, 3, 2, 1, 1)
    sexp = wb.add_block("style.exp", Lt, 4, 1, 1, S)
    pst = wb.add_block("p_style", Lt, 5, 1, 1, S)
    cum = wb.add_block("cum", Lt, 6, 1, 1, S)
    wb.set(Lt, 3, 1, f"=MAX({sl.range()})")
    wb.fill(sexp, lambda _, s: f"=EXP({sl.ref(0, s)}-$A$3)")
    wb.set(Lt, 3, 2, f"=SUM({sexp.range()})")
    wb.fill(pst, lambda _, s: f"={sexp.ref(0, s)}/$B$3")
    wb.fill(cum, lambda _, s: f"={pst.ref(0, 0)}" if s == 0 else f"={cum.ref(0, s - 1)}+{pst.ref(0, s)}")
    wb.add_block("s0", Lt, 7, 1, 1, 1)
    wb.add_block("s1", Lt, 7, 2, 1, 1)
    wb.add_block("u", Lt, 7, 3, 1, 1)
    wb.add_block("style", Lt, 7, 4, 1, 1)
    wb.set(Lt, 7, 1, f"=MOD(INT(ABS({SEED})),{M31 - 1})+1")
    wb.set(Lt, 7, 2, f"=MOD({LCG_A}*A7,{M31})")
    wb.set(Lt, 7, 3, f"=MOD({LCG_A}*B7,{M31})/{M31}")
    wb.set(Lt, 7, 4, f"=MIN({S},1+SUMPRODUCT(({cum.range()}<C7)*1))")
    STYLE = "Latent!$D$7"
    if style_names is None:
        sj = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                          "data", "build", "styles.json")
        style_names = json.load(open(sj)) if os.path.exists(sj) else None
    if style_names and len(style_names) == S:
        SR = 20  # style-name lookup table on the About tab (rows 20..), out of the demo's sight
        wb.set("About", SR - 1, 1, Text("style names"))
        for i, nm in enumerate(style_names):
            wb.set("About", SR + i, 1, Text(str(nm)))
        wb.set(P, 4, 2, f"=INDEX(About!$A${SR}:$A${SR + S - 1},{STYLE})")
    else:
        wb.set(P, 4, 2, f"={STYLE}")
    zb = wb.add_block("z", Lt, 8, 1, 1, d + e)
    wb.fill(zb, lambda _, k: f"={cb.ref(0, k)}" if k < d
            else f"=INDEX({Wemb.range('W_emb', absolute=True)},{STYLE},{k - d + 1})")
    ZROW = zb.row_range(0, sheet=Lt, absolute=True)

    # ---- decoder
    Wd = _wtab(wb, "W_d", w["Wd"].T, w["bd"])                    # row j = output unit j (y, x, c) row-major
    G0 = wb.add_block("G0", "Dense", 1, 1, GRID0, GRID0 * C[0])  # interleaved H x (W*C)
    wb.fill(G0, lambda y, j: f"=MAX(0,{lin(Wd, y * GRID0 * C[0] + j, ZROW, d + e)})")

    prev, Hs = G0, GRID0
    for i in (1, 2, 3):
        Cin, Cout = C[i - 1], C[i]
        Hn = Hs * 2
        # FiLM shift per output channel: c . F_i[:, o] + bc_i[o], in FiLM!row i
        Wf = _wtab(wb, f"W_f{i}", w[f"F{i}"].T, w[f"bc{i}"])
        fb = wb.add_block(f"FiLM{i}", "FiLM", i, 1, 1, Cout)
        wb.fill(fb, lambda _, o, Wf=Wf: "=" + lin(Wf, o, CROW, d))
        # conv weights: block o = rows 3o+1..3o+3, element [ky, kx*Cin+ci] = Wc[ky, kx, ci, o]
        Wc = np.asarray(w[f"Wc{i}"], np.float64)                  # [3,3,Cin,Cout]
        Wm = Wc.transpose(3, 0, 1, 2).reshape(Cout * 3, 3 * Cin)
        wcb = _wtab(wb, f"W_c{i}", Wm)
        # upsample into a zero-bordered (Hn+2) x ((Hn+2)*Cin) grid
        up_name = f"Up_{i}"
        up = wb.add_block(up_name, up_name, 1, 1, Hn + 2, (Hn + 2) * Cin)
        src_sheet = prev.sheet

        def upcell(r, cc, prev=prev, Cin=Cin, Hn=Hn, src_sheet=src_sheet):
            Y, X_ = r - 1, cc // Cin - 1
            if Y < 0 or Y >= Hn or X_ < 0 or X_ >= Hn:
                return 0.0
            ch = cc % Cin
            return "=" + prev.ref(Y // 2, (X_ // 2) * Cin + ch, src_sheet)
        wb.fill(up, upcell)
        cv_name = f"Conv_{i}"
        cv = wb.add_block(f"G{i}", cv_name, 1, 1, Hn, Hn * Cout)

        def convcell(y, j, up=up, Cin=Cin, Cout=Cout, wcb=wcb, fb=fb, up_name=up_name, i=i):
            x, o = divmod(j, Cout)
            win = rng(up.r0 + y, up.c0 + x * Cin, up.r0 + y + 2, up.c0 + x * Cin + 3 * Cin - 1, up_name)
            wr = rng(wcb.r0 + 3 * o, 1, wcb.r0 + 3 * o + 2, 3 * Cin, wcb.sheet)
            return f"=MAX(0,SUMPRODUCT({win},{wr})+{fb.ref(0, o, 'FiLM', absolute=True)})"
        wb.fill(cv, convcell)
        prev, Hs = cv, Hn

    # ---- logits (1x1 conv), pixels, hex
    C3 = C[3]
    Wout = _wtab(wb, "W_out", w["Wout"].T, w["bout"])            # row k = class k
    lg = wb.add_block("Logits", "Logits", 1, 1, N, N * (K + 1))
    wb.fill(lg, lambda y, j: "=" + lin(Wout, j % (K + 1), prev.row_range(y, (j // (K + 1)) * C3,
                                                                          (j // (K + 1)) * C3 + C3 - 1, prev.sheet), C3))
    px = wb.add_block("Pixels", "Pixels", 1, 1, N, N)

    def pixcell(y, x):
        r = lg.row_range(y, x * (K + 1), x * (K + 1) + K, "Logits")
        return f"=MATCH(MAX({r}),{r},0)-1"
    wb.fill(px, pixcell)
    pal = wb.add_block("Palette", "Palette", 1, 1, K + 1, 1)
    for i, hx in enumerate([BACKGROUND] + list(palette_hex)):
        wb.set("Palette", 1 + i, 1, Text(hx))
    PAL = pal.range("Palette", absolute=True)
    hx = wb.add_block("Hex", "Hex", 1, 1, N, N)
    wb.fill(hx, lambda y, x: f"=INDEX({PAL},{px.ref(y, x, 'Pixels')}+1)")

    # ---- "Showing:" status (C2). It depends on every pixel, so it only switches to the new prompt once the whole
    # sprite has been recalculated: a done-indicator for the demo. Portable formula.
    wb.set(P, 2, DISPLAY_C0, f'="Showing: "&IF(SUMPRODUCT(({px.range("Pixels", absolute=True)}>=0)*1)={N * N},{PROMPT},"")')
    # ---- stage bar (C4:K4, Sheets only): one square per pipeline stage, each depending on that stage's last cell.
    stages = ["Tokens.ids", "H", "z", "G0", "G1", "G2", "G3", "Logits", "Pixels"]
    prog = wb.add_block("Progress", P, 5, DISPLAY_C0, 1, len(stages))
    wb.meta["progress"] = {"r0": 5, "c0": DISPLAY_C0, "stages": stages}

    # ---- render (Google Sheets only)
    disp = wb.add_block("Display", P, DISPLAY_R0, DISPLAY_C0, N, N)
    wb.meta["display"] = {"r0": DISPLAY_R0, "c0": DISPLAY_C0, "n": N}
    if render:
        # Display = class ids; the Sheets writer colours them with one conditional-format rule per class and hides
        # the digits (number format ;;;), so the 32x32 block reads as one seamless image. Portable formulas only.
        wb.fill(disp, lambda y, x: "=" + px.ref(y, x, "Pixels"))
        for i, name in enumerate(stages):
            b = wb.blocks[name]
            last = b.ref(b.nrows - 1, b.ncols - 1, b.sheet)
            wb.set(P, 5, DISPLAY_C0 + i, f"=IF(ABS({last})>=0,1,0)")
    wb.meta["palette_hex"] = list(palette_hex)

    # ---- About
    ab = "About"
    mh = model_hash(w, cfg, vocab_words, palette_hex)
    wb.meta["model_hash"] = mh
    rows = [("compiler", VERSION), ("model hash", mh), ("config", str({k: cfg[k] for k in sorted(cfg)})),
            ("vocab size", str(V)),
            ("source", "github.com: SpriteSheet (Pixcell compiler); spec = src/spritesheet/reference.py")]
    for r, (k, v) in enumerate(rows, start=1):
        wb.set(ab, r, 1, Text(k))
        wb.set(ab, r, 2, Text(v))
    return wb


def tokenizer_rows(prompts, vocab_words):
    """Gate A1 workbook: one prompt per row of the 'Tok' tab, the SAME formula generators as compile_model.
    Columns: A prompt | B clean | then for i=1..L: rest_i, word_i | then id_1..id_L."""
    wb = Workbook()
    V = len(vocab_words)
    for i, word in enumerate(vocab_words):
        wb.set("Vocab", 1 + i, 1, Text(word))
    VOCAB = rng(1, 1, V, 1, "Vocab", absolute=True)
    put_punct(wb, "Tokens")
    L = T.L
    id_c0 = 3 + 2 * L
    for p, prompt in enumerate(prompts):
        r = 1 + p
        wb.set("Tok", r, 1, Text(prompt))
        wb.set("Tok", r, 2, f_clean(f"A{r}", "Tokens!"))
        for i in range(L):
            rc, wc = 3 + 2 * i, 4 + 2 * i
            wb.set("Tok", r, rc, f_rest(i, f"B{r}", a1(r, rc - 2), a1(r, wc - 2)))
            wb.set("Tok", r, wc, f_word(a1(r, rc)))
            wb.set("Tok", r, id_c0 + i, f_id(a1(r, wc), VOCAB))
    wb.meta["tok"] = {"n": len(prompts), "id_c0": id_c0, "L": L}
    return wb
