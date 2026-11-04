"""Pixcell emitter for SpriteSheet v2: few-step masked discrete diffusion (spec: mdm.ref_generate / ref_denoise).

Same front as v1 (emit.emit_front: Prompt inputs, Tokens, attention encoder, Latent with z = cond = [c, style_emb]).
Then ONE shared set of weight tabs (W_tok, W_c1..W_c6, W_f1..W_f6 -> FiLM, W_p2, W_p1, W_out) and T denoising
steps, each with its own tabs prefixed S{t}_ (layout: docs/04 "v2 workbook layout"):

  S{t}_Emb     (N+2) x (N+2)E   h0 = INDEX(W_tok, x+1, k) from the previous canvas (step 1: constant MASK = K+1)
  S{t}_A1      N x N C1         MAX(0, SUMPRODUCT(3x3E window, kernel) + FiLM1)
  S{t}_Pool1   bordered avg2    (a+b+c+d)/4
  S{t}_A2      N/2 x N/2 C2
  S{t}_Pool2   bordered avg2
  S{t}_A3      bordered (the conv writes straight into the zero-bordered layout a4 reads); v2b: + GMap after ReLU
  S{t}_A4      N/4 x N/4 C3
  S{t}_P2      a4 @ P2 per cell (1x1, before upsampling); S{t}_Up2 nearest x2 as references, bordered
  S{t}_U2      MAX(0, conv + A2 skip + FiLM5)
  S{t}_P1, S{t}_Up1, S{t}_U1   the same at 32x32 (skip A1, FiLM6)
  S{t}_Logits  N x N(K+1)       SUMPRODUCT(u1 segment, W_out row) + b
  S{t}_Conf    max | conf = 1/SUMPRODUCT(EXP(l - max)) | argmax = MATCH(max, l, 0)-1   (three N x N blocks)
  S{t}_Canvas  IF(old=MASK, IF(conf>=tau_t, argmax, MASK), old); at t = T: IF(old=MASK, argmax, old)

Pixels = S{T}_Canvas; Palette/Hex/display as in v1. Filmstrip: T thumbnails (one per step canvas) below the display.
"""
import numpy as np

from .. import tokenizer as T
from .emit import (BACKGROUND, DISPLAY_C0, DISPLAY_R0, VERSION, _lin, _wtab, emit_about, emit_front,
                   emit_output)
from .workbook import Text, Workbook, rng
from ..mdm import HASH_B
from ..reference import LCG_A, M31

S1_REF = "Latent!$B$7"     # seed state s1 (emit_front), the base of the per-(step, pixel, class) uniforms

MDM_VERSION = VERSION + "+mdm"
MASK_COLOUR = "#b9b4c2"   # filmstrip colour of a still-masked ("unknown") pixel
FILM_PER_ROW = 2          # thumbnails per filmstrip row (same 12 px grid as the display)
STEP_PARTS = ("Emb", "A1", "Pool1", "A2", "Pool2", "A3", "A4", "P2", "Up2", "U2", "P1", "Up1", "U1", "Logits",
              "Conf", "Canvas")


def step_sheet(t: int, part: str) -> str:
    return f"S{t}_{part}"


class Act:
    """An H x H x C activation stored H x (H*C) (channels interleaved), optionally inside a one-cell zero border."""

    def __init__(self, block, H, C, border):
        self.b, self.H, self.C, self.o = block, H, C, (1 if border else 0)

    def ref(self, y, x, ch):
        return self.b.ref(y + self.o, (x + self.o) * self.C + ch, self.b.sheet)

    def seg(self, y, x):
        """The C channels of pixel (y, x) as one row range (for 1x1 projections)."""
        c0 = (x + self.o) * self.C
        return self.b.row_range(y + self.o, c0, c0 + self.C - 1, self.b.sheet)

    def window(self, y, x):
        """3 x 3C window centred on (y, x); requires a bordered activation."""
        assert self.o == 1
        b = self.b
        return rng(b.r0 + y, b.c0 + x * self.C, b.r0 + y + 2, b.c0 + x * self.C + 3 * self.C - 1, b.sheet)


def _act(wb: Workbook, name, sheet, H, C, border, inner) -> Act:
    """inner(y, x, ch) -> cell value for the H x H interior; border cells are 0."""
    n = H + 2 if border else H
    blk = wb.add_block(name, sheet, 1, 1, n, n * C)
    o = 1 if border else 0

    def cell(r, cc):
        y, x = r - o, cc // C - o
        if y < 0 or y >= H or x < 0 or x >= H:
            return 0.0
        return inner(y, x, cc % C)
    wb.fill(blk, cell)
    return Act(blk, H, C, border)


def compile_mdm(w, cfg, vocab_words, palette_hex, default_prompt="a red apple", default_seed=1, render=False,
                style_names=None) -> Workbook:
    d, e, K, E = cfg["d"], cfg["e"], cfg["K"], cfg["E"]
    C1, C2, C3, N, TT = cfg["C1"], cfg["C2"], cfg["C3"], cfg["N"], cfg["T"]
    V, M, dz = len(vocab_words), K + 1, d + e
    tau = [float(x) for x in cfg["tau"]]
    temp = float(cfg.get("temp", 0.0) or 0.0)          # > 0: seeded Gumbel-max sampling (mdm.seeded_uniform)
    assert temp >= 0.0
    assert len(tau) >= TT - 1, ("cfg['tau'] needs a threshold for every step but the last", tau, TT)
    assert cfg.get("L", T.L) == T.L and N % 4 == 0
    assert w["wordvec"].shape == (V, d), (w["wordvec"].shape, V, d)
    assert vocab_words[0] == "<pad>" and vocab_words[1] == "<unk>"
    assert len(palette_hex) == K, (len(palette_hex), K)
    assert w["Emb"].shape == (K + 2, E) and w["Wout"].shape == (C1, K + 1)
    wb = Workbook()
    wb.meta.update({"config": dict(cfg), "version": MDM_VERSION, "render": render, "N": N, "V": V, "model": "mdm",
                    "T": TT, "mask_class": M, "temp": temp})
    intro = (f"Type a prompt in B2 and press Enter. The picture is drawn in {TT} denoising steps (filmstrip below). "
             "Change the seed (B3) to re-roll the drawing style.")
    fr = emit_front(wb, w, cfg, vocab_words, default_prompt, default_seed, style_names, intro=intro)
    ZROW = fr["ZROW"]                                    # cond = [c, style_emb[style]] (Latent row 8)

    # ---------------- shared weights (every step references these)
    Wtok = _wtab(wb, "W_tok", w["Emb"])                 # row x+1 = embedding of class x (row K+2 = MASK)
    WTOK = Wtok.range("W_tok", absolute=True)
    chans = {1: (E, C1), 2: (C1, C2), 3: (C2, C3), 4: (C3, C3), 5: (C2, C2), 6: (C1, C1)}
    Wc, Fb = {}, {}
    for i, (cin, cout) in chans.items():
        Wk = np.asarray(w[f"Wc{i}"], np.float64)        # [3,3,Cin,Cout]; block o = rows 3o+1..3o+3
        assert Wk.shape == (3, 3, cin, cout), (i, Wk.shape)
        Wc[i] = _wtab(wb, f"W_c{i}", Wk.transpose(3, 0, 1, 2).reshape(cout * 3, 3 * cin))
    for i, (cin, cout) in chans.items():                 # FiLM shift = cond . F_i + f_i, computed ONCE
        Wf = _wtab(wb, f"W_f{i}", w[f"F{i}"].T, w[f"f{i}"])
        Fb[i] = wb.add_block(f"FiLM{i}", "FiLM", i, 1, 1, cout)
        wb.fill(Fb[i], lambda _, o, Wf=Wf: "=" + _lin(Wf, o, ZROW, dz))
    Wp2 = _wtab(wb, "W_p2", w["P2"].T)                   # row o = output channel (C2), C3 inputs
    Wp1 = _wtab(wb, "W_p1", w["P1"].T)                   # C1 x C2
    Wout = _wtab(wb, "W_out", w["Wout"].T, w["bout"])    # (K+1) x (C1+1)
    gmap = None
    if "Wg" in w:   # v2b global map: G = cond @ Wg + bg as N/4 x N/4 x C3 (y, x, c row-major), added to a3 every step
        n8 = N // 4
        assert w["Wg"].shape == (dz, n8 * n8 * C3), w["Wg"].shape
        Wg = _wtab(wb, "W_g", w["Wg"].T, w["bg"])          # row j = (y*n8 + x)*C3 + c
        gmap = _act(wb, "GMap", "GMap", n8, C3, False, lambda y, x, c: "=" + _lin(Wg, (y * n8 + x) * C3 + c, ZROW, dz))

    def film(i, o):
        return Fb[i].ref(0, o, "FiLM", absolute=True)

    def conv(i, src: Act, o, y, x, skip=None, post=None):
        cin = src.C
        wr = rng(Wc[i].r0 + 3 * o, 1, Wc[i].r0 + 3 * o + 2, 3 * cin, Wc[i].sheet)
        sk = f"+{skip}" if skip else ""
        po = f"+{post}" if post else ""                  # added after the ReLU (the v2b global map on a3)
        return f"=MAX(0,SUMPRODUCT({src.window(y, x)},{wr}){sk}+{film(i, o)}){po}"

    def proj(Wp, src: Act, o):
        return lambda y, x: f"=SUMPRODUCT({src.seg(y, x)},{Wp.row_range(o, 0, src.C - 1, Wp.sheet)})"

    def pool(a: Act):
        return lambda y, x, ch: (f"=({a.ref(2 * y, 2 * x, ch)}+{a.ref(2 * y, 2 * x + 1, ch)}"
                                 f"+{a.ref(2 * y + 1, 2 * x, ch)}+{a.ref(2 * y + 1, 2 * x + 1, ch)})/4")

    canvas = None
    canvases = []
    for t in range(1, TT + 1):
        S = lambda part, t=t: step_sheet(t, part)
        B = lambda part, t=t: f"S{t}.{part.lower()}"
        if canvas is None:
            emb = lambda y, x, k: f"=INDEX({WTOK},{M + 1},{k + 1})"
        else:
            emb = lambda y, x, k, cv=canvas: f"=INDEX({WTOK},{cv.ref(y, x, cv.sheet)}+1,{k + 1})"
        h0 = _act(wb, B("Emb"), S("Emb"), N, E, True, emb)
        a1 = _act(wb, B("A1"), S("A1"), N, C1, False, lambda y, x, o: conv(1, h0, o, y, x))
        p1 = _act(wb, B("Pool1"), S("Pool1"), N // 2, C1, True, pool(a1))
        a2 = _act(wb, B("A2"), S("A2"), N // 2, C2, False, lambda y, x, o: conv(2, p1, o, y, x))
        p2 = _act(wb, B("Pool2"), S("Pool2"), N // 4, C2, True, pool(a2))
        a3 = _act(wb, B("A3"), S("A3"), N // 4, C3, True, lambda y, x, o: conv(3, p2, o, y, x, post=gmap and gmap.ref(y, x, o)))
        a4 = _act(wb, B("A4"), S("A4"), N // 4, C3, False, lambda y, x, o: conv(4, a3, o, y, x))
        q2 = _act(wb, B("P2"), S("P2"), N // 4, C2, False, lambda y, x, o: proj(Wp2, a4, o)(y, x))
        up2 = _act(wb, B("Up2"), S("Up2"), N // 2, C2, True, lambda y, x, ch: "=" + q2.ref(y // 2, x // 2, ch))
        u2 = _act(wb, B("U2"), S("U2"), N // 2, C2, False, lambda y, x, o: conv(5, up2, o, y, x, a2.ref(y, x, o)))
        q1 = _act(wb, B("P1"), S("P1"), N // 2, C1, False, lambda y, x, o: proj(Wp1, u2, o)(y, x))
        up1 = _act(wb, B("Up1"), S("Up1"), N, C1, True, lambda y, x, ch: "=" + q1.ref(y // 2, x // 2, ch))
        u1 = _act(wb, B("U1"), S("U1"), N, C1, False, lambda y, x, o: conv(6, up1, o, y, x, a1.ref(y, x, o)))
        lg = _act(wb, B("Logits"), S("Logits"), N, K + 1, False,
                  lambda y, x, k: "=" + _lin(Wout, k, u1.seg(y, x), C1))
        cs = S("Conf")
        mxb = wb.add_block(B("max"), cs, 1, 1, N, N)
        cfb = wb.add_block(B("conf"), cs, 1, N + 2, N, N)
        amb = wb.add_block(B("argmax"), cs, 1, 2 * N + 3, N, N)
        row = lambda y, x, lg=lg: lg.b.row_range(y, x * (K + 1), x * (K + 1) + K, lg.b.sheet)
        wb.fill(mxb, lambda y, x: f"=MAX({row(y, x)})")
        if temp == 0.0:
            wb.fill(cfb, lambda y, x: f"=1/SUMPRODUCT(EXP({row(y, x)}-{mxb.ref(y, x)}))")
            wb.fill(amb, lambda y, x: f"=MATCH({mxb.ref(y, x)},{row(y, x)},0)-1")
        else:
            # score = logit + temp * Gumbel(u), u from the seed state s1 = Latent!B7 and the (step, pixel, class)
            # index (exact integer hash, mdm.seeded_uniform); chosen = first argmax of the score row;
            # conf = p(chosen) = EXP(l_chosen - max) * (1/sumexp).
            K1 = K + 1

            def score(y, x, k, t=t, lg=lg):
                idx = ((t - 1) * N * N + y * N + x) * K1 + k + 1
                u = f"(MOD({LCG_A}*MOD({S1_REF}+{HASH_B}*{idx},{M31}),{M31})+0.5)/{M31}"
                return f"={lg.ref(y, x, k)}+{temp!r}*(-LN(-LN({u})))"
            sc = _act(wb, B("Score"), S("Score"), N, K1, False, score)
            srow = lambda y, x, sc=sc: sc.b.row_range(y, x * K1, x * K1 + K, sc.b.sheet)
            wb.fill(amb, lambda y, x: f"=MATCH(MAX({srow(y, x)}),{srow(y, x)},0)-1")
            wb.fill(cfb, lambda y, x: (f"=EXP(INDEX({row(y, x)},1,{amb.ref(y, x)}+1)-{mxb.ref(y, x)})"
                                       f"*(1/SUMPRODUCT(EXP({row(y, x)}-{mxb.ref(y, x)})))"))
        cvb = wb.add_block(B("Canvas"), S("Canvas"), 1, 1, N, N)
        conf_ref = lambda y, x: cfb.ref(y, x, cs)
        am_ref = lambda y, x: amb.ref(y, x, cs)
        if t == TT and canvas is None:            # T = 1: commit everything at once
            fn = lambda y, x: f"={am_ref(y, x)}"
        elif t == TT:                              # last step: commit every pixel still masked
            fn = lambda y, x, cv=canvas: (f"=IF({cv.ref(y, x, cv.sheet)}={M},{am_ref(y, x)},"
                                          f"{cv.ref(y, x, cv.sheet)})")
        elif canvas is None:                       # step 1: the old canvas is all MASK (a constant)
            fn = lambda y, x, th=tau[t - 1]: f"=IF({conf_ref(y, x)}>={th!r},{am_ref(y, x)},{M})"
        else:
            fn = lambda y, x, cv=canvas, th=tau[t - 1]: (
                f"=IF({cv.ref(y, x, cv.sheet)}={M},IF({conf_ref(y, x)}>={th!r},{am_ref(y, x)},{M}),"
                f"{cv.ref(y, x, cv.sheet)})")
        wb.fill(cvb, fn)
        canvas = cvb
        canvases.append(cvb)

    # ---------------- output: Pixels = final canvas; Palette, Hex, Showing, display (as v1)
    px = wb.add_block("Pixels", "Pixels", 1, 1, N, N)
    wb.fill(px, lambda y, x: "=" + canvas.ref(y, x, canvas.sheet))
    emit_output(wb, px, palette_hex, render, fr["PROMPT"])
    P = "Prompt"
    # ---------------- filmstrip: one thumbnail per step canvas, FILM_PER_ROW per row below the display, same grid
    film_meta = []
    top = DISPLAY_R0 + N + 1                    # one blank row under the display, then label row + thumbnail
    for t, cv in enumerate(canvases, start=1):
        band, col = divmod(t - 1, FILM_PER_ROW)
        lab_r = top + band * (N + 2)
        r0, c0 = lab_r + 1, DISPLAY_C0 + col * (N + 1)
        wb.set(P, lab_r, c0, Text(f"step {t}" + (" (final)" if t == TT else "")))
        fb = wb.add_block(f"Film{t}", P, r0, c0, N, N)
        if render:
            wb.fill(fb, lambda y, x, cv=cv: "=" + cv.ref(y, x, cv.sheet))
        film_meta.append({"step": t, "r0": r0, "c0": c0, "n": N, "label_row": lab_r})
    wb.meta["filmstrip"] = film_meta
    wb.meta["mask_colour"] = MASK_COLOUR
    emit_about(wb, w, cfg, vocab_words, palette_hex, spec="src/spritesheet/mdm.py (ref_generate)",
               extra=(("model", f"v2 masked discrete diffusion, T={TT} steps"
                       + (f", Gumbel sampling temp={temp!r}" if temp else ", argmax")),
                      ("tau", ", ".join(repr(x) for x in tau[: TT - 1]) + " (last step commits all)")))
    return wb


__all__ = ["compile_mdm", "step_sheet", "STEP_PARTS", "MASK_COLOUR", "BACKGROUND"]
