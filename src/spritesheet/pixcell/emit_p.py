"""Pixcell emitter for SpriteSheet v3-P (PatchGPT; spec: reference_p.forward).

The whole autoregressive sampler is ONE causal grid of formulas: sequence row n only reads rows <= n, and image row
L+t reads the code picked from row L-1+t, so the spreadsheet's dependency graph runs the 64 sampling steps by itself.
One formula per cell, portable subset (docs/05) plus SQRT (RMSNorm) and LN (Gumbel noise, Score tabs only; D-52).
"""
import math

import numpy as np

from .. import tokenizer as T
from ..reference import M31, LCG_A
from ..reference_p import HASH_B, EPS
from .emit import _wtab, emit_tokens, emit_output, emit_about
from .workbook import Block, Text, Workbook, rng


def _lin(Wb: Block, j: int, xrow: str, n: int, scale_ref=None) -> str:
    """(x . W[:, j]) [* r] + b_j for a weight tab whose row j holds W[:, j] then the bias."""
    s = f"SUMPRODUCT({xrow},{Wb.row_range(j, 0, n - 1, Wb.sheet)})"
    if scale_ref:
        s += f"*{scale_ref}"
    return s + f"+{Wb.ref(j, n, Wb.sheet)}"


def emit_stack(wb: Workbook, w, pre: str, X0: Block, N: int, dm: int, H: int, mlp: int, layers: int,
               keymask=lambda m: None, tab=lambda l, part: f"Blk{l + 1}_{part}", is_q=None) -> Block:
    """Transformer blocks over the N rows of X0 (causal). keymask(m) → cell ref of key m's 0/1 mask, None (always a
    key) or False (never a key for other rows: constant 0, no formula; D-53 query rows).
    Returns the final x block. Block names: f"{pre}b{l}.<part>" (trace_p uses the same names)."""
    dh = dm // H
    scale = 1.0 / math.sqrt(dh)
    x = X0
    # v3-PG pruning (D-53): query rows are never keys (no K/V cells, no self weight); in the last layer only query rows
    # need outputs, so the other rows keep just 1/rms, K and V there. Pruned cells are constant 0 (no formula).
    isq = is_q or (lambda n: False)
    pruned = is_q is not None
    for l in range(layers):
        last = pruned and l == layers - 1
        need_out = lambda n, last=last: not last or isq(n)
        p, wp = f"{pre}b{l}.", f"{pre}l{l}."
        Wq = _wtab(wb, tab(l, "Wq"), w[wp + "Wq"].T, w[wp + "bq"])
        Wk = _wtab(wb, tab(l, "Wk"), w[wp + "Wk"].T, w[wp + "bk"])
        Wv = _wtab(wb, tab(l, "Wv"), w[wp + "Wv"].T, w[wp + "bv"])
        Wo = _wtab(wb, tab(l, "Wo"), w[wp + "Wo"].T, w[wp + "bo"])
        W1 = _wtab(wb, tab(l, "W1"), w[wp + "W1"].T, w[wp + "b1"])
        W2 = _wtab(wb, tab(l, "W2"), w[wp + "W2"].T, w[wp + "b2"])
        sq = tab(l, "QKV")
        xs = x.sheet
        xrow = lambda n, x=x, xs=xs: x.row_range(n, sheet=xs)
        rb = wb.add_block(p + "r", sq, 1, 1, N, 1)
        wb.fill(rb, lambda n, _, xrow=xrow: f"=1/SQRT(SUMPRODUCT({xrow(n)},{xrow(n)})/{dm}+{EPS!r})")
        Q = wb.add_block(p + "Q", sq, 1, 3, N, dm)
        K = wb.add_block(p + "K", sq, 1, 4 + dm, N, dm)
        VT = wb.add_block(p + "VT", sq, N + 2, 3, dm, N)
        wb.fill(Q, lambda n, j, xrow=xrow, rb=rb, no=need_out: "=" + _lin(Wq, j, xrow(n), dm, rb.ref(n, 0)) if no(n) else 0.0)
        wb.fill(K, lambda n, j, xrow=xrow, rb=rb: 0.0 if isq(n) else "=" + _lin(Wk, j, xrow(n), dm, rb.ref(n, 0)))
        wb.fill(VT, lambda j, m, xrow=xrow, rb=rb: 0.0 if isq(m) else "=" + _lin(Wv, j, xrow(m), dm, rb.ref(m, 0)))
        sa = tab(l, "Att")
        Es, Zs = [], []
        for h in range(H):
            Eb = wb.add_block(p + f"E{h}", sa, 1, 1 + h * (N + 1), N, N)

            def ecell(n, m, h=h, Q=Q, K=K, sq=sq, no=need_out):
                km = keymask(m) if m < n else None
                if m > n or km is False or not no(n) or (m == n and isq(n)):
                    return 0.0
                s = (f"EXP(SUMPRODUCT({Q.row_range(n, h * dh, h * dh + dh - 1, sq)},"
                     f"{K.row_range(m, h * dh, h * dh + dh - 1, sq)})*{scale!r})")
                return f"={s}*{km}" if km else f"={s}"
            wb.fill(Eb, ecell)
            Es.append(Eb)
        Zb = wb.add_block(p + "Z", sa, N + 2, 1, N, H)
        wb.fill(Zb, lambda n, h, no=need_out: f"=SUM({Es[h].row_range(n, 0, n)})" if no(n) else 0.0)
        sc = tab(l, "Ctx")
        C = wb.add_block(p + "ctx", sc, 1, 1, N, dm)
        wb.fill(C, lambda n, j, no=need_out: (f"=SUMPRODUCT({Es[j // dh].row_range(n, 0, n, sa)},{VT.row_range(j, 0, n, sq)})"
                                              f"/{Zb.ref(n, j // dh, sa)}") if no(n) else 0.0)
        X1 = wb.add_block(p + "x1", sc, N + 2, 1, N, dm)
        wb.fill(X1, lambda n, j, x=x, xs=xs, no=need_out: f"={x.ref(n, j, xs)}+" + _lin(Wo, j, C.row_range(n), dm) if no(n) else 0.0)
        sm = tab(l, "MLP")
        x1row = lambda n: X1.row_range(n, sheet=sc)
        r2 = wb.add_block(p + "r2", sm, 1, 1, N, 1)
        wb.fill(r2, lambda n, _, no=need_out: f"=1/SQRT(SUMPRODUCT({x1row(n)},{x1row(n)})/{dm}+{EPS!r})" if no(n) else 0.0)
        Hd = wb.add_block(p + "hid", sm, 1, 3, N, mlp)
        wb.fill(Hd, lambda n, i, no=need_out: f"=MAX(0,{_lin(W1, i, x1row(n), dm, r2.ref(n, 0))})" if no(n) else 0.0)
        X2 = wb.add_block(p + "x2", sm, N + 2, 3, N, dm)
        wb.fill(X2, lambda n, j, no=need_out: f"={X1.ref(n, j, sc)}+" + _lin(W2, j, Hd.row_range(n), mlp) if no(n) else 0.0)
        x = X2
    return x


def _guidance_cell(wb, g):
    """Guidance weight as an editable input (Prompt!B4; the display starts at row 5). Default = the exported g;
    0 turns guidance off (the unconditional chain still computes)."""
    if not g:
        return None
    wb.set("Prompt", 4, 1, Text("Guidance"))
    wb.set("Prompt", 4, 2, float(g))
    return "Prompt!$B$4"


def compile_p(w, cfg, vocab_words, palette_hex, default_prompt="a red frog", default_seed=1, render=False) -> Workbook:
    L, T, Kp, dm, P = T_L(cfg), cfg["T"], cfg["Kp"], cfg["dm"], cfg["P"]
    N = L + T - 1
    Gd = int(round(T ** 0.5))
    NPX = Gd * P
    g = float(cfg.get("guidance", 0.0))
    inv_tau = 1.0 / float(cfg.get("tau", 1.0))
    assert len(palette_hex) == cfg["K"] and vocab_words[0] == "<pad>"
    wb = Workbook()
    wb.meta.update({"config": dict(cfg), "model": "v3p", "render": render, "N": NPX, "V": len(vocab_words)})
    tk = emit_tokens(wb, w, cfg, vocab_words, default_prompt, default_seed, show_style=False)
    SEED = tk["SEED"]

    # ---- seed state (same as v1/v2: s0 = MOD(INT(ABS(seed)),M31-1)+1, s1 = MOD(48271*s0, M31))
    Sd = "Seed"
    wb.add_block("seed.s0", Sd, 1, 1, 1, 1)
    wb.add_block("seed.s1", Sd, 1, 2, 1, 1)
    wb.set(Sd, 1, 1, f"=MOD(INT(ABS({SEED})),{M31 - 1})+1")
    wb.set(Sd, 1, 2, f"=MOD({LCG_A}*A1,{M31})")
    S1 = "Seed!$B$1"
    wb.meta["seed_range"] = "'Seed'!A1:B1"

    # ---- input rows X0: prompt words, then image tokens fed back from Codes
    Win = _wtab(wb, "W_in", w["Win"].T, w["b_in"])
    Pt = _wtab(wb, "W_postext", w["pos_text"])
    Eb = _wtab(wb, "W_E", w["E"])
    Pi = _wtab(wb, "W_posimg", w["pos_img"])
    Db = _wtab(wb, "W_D", w["D"])
    d = w["wordvec"].shape[1]
    WVr = wb.add_block("WV", "Inputs", 1, 1, L, d)
    wb.fill(WVr, lambda n, k: f"=INDEX({tk['WV']},Tokens!$D${1 + n},{k + 1})")
    X0 = wb.add_block("X0", "Inputs", 1, d + 2, N, dm)
    CODES = wb.add_block("Codes", "Out", 1, 1, T, 1)
    ER = Eb.range("W_E", absolute=True)

    def x0cell(n, j):
        if n < L:
            return f"=" + _lin(Win, j, WVr.row_range(n), d) + f"+{Pt.ref(n, j, 'W_postext')}"
        t = n - L
        return f"=INDEX({ER},{CODES.ref(t, 0, 'Out')}+1,{j + 1})+{Pi.ref(t, j, 'W_posimg')}"
    wb.fill(X0, x0cell)
    keymask = lambda m: f"Tokens!$E${1 + m}" if m < L else None
    xf = emit_stack(wb, w, "", X0, N, dm, cfg["H"], cfg["mlp"], cfg["layers"], keymask)

    # ---- optional unconditional chain (classifier-free guidance); its weight g is a live input cell (Prompt!B4)
    LU = None
    GC = _guidance_cell(wb, g)
    if g:
        U0 = wb.add_block("U.X0", "U_X0", 1, 1, T, cfg["U_dm"])
        UE = _wtab(wb, "U_W_E", w["U_E"])
        UP = _wtab(wb, "U_W_pos", w["U_pos"])
        US = _wtab(wb, "U_W_start", w["U_start"][None, :])
        UER = UE.range("U_W_E", absolute=True)
        wb.fill(U0, lambda t, j: f"={US.ref(0, j, 'U_W_start')}" if t == 0 else
                f"=INDEX({UER},{CODES.ref(t - 1, 0, 'Out')}+1,{j + 1})+{UP.ref(t - 1, j, 'U_W_pos')}")
        ux = emit_stack(wb, w, "U", U0, T, cfg["U_dm"], cfg["U_H"], cfg["U_mlp"], cfg["U_layers"],
                        tab=lambda l, part: f"U{l + 1}_{part}")
        UWo = _wtab(wb, "U_W_out", w["U_Wout"].T, w["U_bout"])
        urf = wb.add_block("U.rf", "U_Out", 1, 1, T, 1)
        uxr = lambda t: ux.row_range(t, sheet=ux.sheet)
        wb.fill(urf, lambda t, _: f"=1/SQRT(SUMPRODUCT({uxr(t)},{uxr(t)})/{cfg['U_dm']}+{EPS!r})")
        # the unconditional logit is inlined in each Score cell (D-55): no separate 64 x Kp logit tab
        LU = lambda t, k: _lin(UWo, k, uxr(t), cfg["U_dm"], urf.ref(t, 0, "U_Out"))

    # ---- logits → Gumbel score → code (one MATCH(MAX) per step)
    Wout = _wtab(wb, "W_out", w["Wout"].T, w["bout"])
    xr = lambda t: xf.row_range(L - 1 + t, sheet=xf.sheet)
    rf = wb.add_block("rf", "Out", 1, 3, T, 1)
    wb.fill(rf, lambda t, _: f"=1/SQRT(SUMPRODUCT({xr(t)},{xr(t)})/{dm}+{EPS!r})")
    Sc = wb.add_block("Score", "Score", 1, 1, T, Kp)

    def score(t, k):
        lc = _lin(Wout, k, xr(t), dm, rf.ref(t, 0, "Out"))
        lg = f"((1+{GC})*({lc})-{GC}*({LU(t, k)}))" if g else f"({lc})"
        u = f"(MOD({LCG_A}*MOD({S1}+{HASH_B}*{t * Kp + k + 1},{M31}),{M31})+0.5)/{M31}"
        return f"={lg}*{inv_tau!r}+(-LN(-LN({u})))"
    wb.fill(Sc, score)
    wb.fill(CODES, lambda t, _: f"=MATCH(MAX({Sc.row_range(t, sheet='Score')}),{Sc.row_range(t, sheet='Score')},0)-1")

    # ---- pixels = patch lookup
    DR = Db.range("W_D", absolute=True)
    px = wb.add_block("Pixels", "Pixels", 1, 1, NPX, NPX)
    wb.fill(px, lambda y, x: f"=INDEX({DR},{CODES.ref((y // P) * Gd + x // P, 0, 'Out')}+1,{(y % P) * P + x % P + 1})")
    emit_output(wb, px, palette_hex, render, tk["PROMPT"])
    emit_about(wb, w, cfg, vocab_words, palette_hex, spec="src/spritesheet/reference_p.py")
    return wb


def T_L(cfg):
    assert cfg.get("L", T.L) == T.L
    return T.L


def compile_pg(w, cfg, vocab_words, palette_hex, default_prompt="a red frog", default_seed=1, render=False) -> Workbook:
    """v3-PG (grouped decoding, spec reference_p.forward_g): rows = prompt, then per group query rows and context rows.
    Query rows are never keys of other rows; Score/Codes are indexed by patch position."""
    from ..reference_p import group_rows
    L, T, Kp, dm, P = T_L(cfg), cfg["T"], cfg["Kp"], cfg["dm"], cfg["P"]
    order, rows = group_rows(cfg)
    R = len(rows)
    N = L + R
    Gd = int(round(T ** 0.5))
    NPX = Gd * P
    g = float(cfg.get("guidance", 0.0))
    inv_tau = 1.0 / float(cfg.get("tau", 1.0))
    qrow = {p: L + i for i, (k, p, _) in enumerate(rows) if k == "q"}      # position → sequence row of its query
    wb = Workbook()
    wb.meta.update({"config": dict(cfg), "model": "v3pg", "render": render, "N": NPX, "V": len(vocab_words)})
    tk = emit_tokens(wb, w, cfg, vocab_words, default_prompt, default_seed, show_style=False)
    Sd = "Seed"
    wb.add_block("seed.s0", Sd, 1, 1, 1, 1)
    wb.add_block("seed.s1", Sd, 1, 2, 1, 1)
    wb.set(Sd, 1, 1, f"=MOD(INT(ABS({tk['SEED']})),{M31 - 1})+1")
    wb.set(Sd, 1, 2, f"=MOD({LCG_A}*A1,{M31})")
    wb.meta["seed_range"] = "'Seed'!A1:B1"
    S1 = "Seed!$B$1"
    Win = _wtab(wb, "W_in", w["Win"].T, w["b_in"])
    Pt = _wtab(wb, "W_postext", w["pos_text"])
    Eb = _wtab(wb, "W_E", w["E"])
    Qp = _wtab(wb, "W_qpos", w["qpos"])
    Cp = _wtab(wb, "W_cpos", w["cpos"])
    Db = _wtab(wb, "W_D", w["D"])
    d = w["wordvec"].shape[1]
    WVr = wb.add_block("WV", "Inputs", 1, 1, L, d)
    wb.fill(WVr, lambda n, k: f"=INDEX({tk['WV']},Tokens!$D${1 + n},{k + 1})")
    X0 = wb.add_block("X0", "Inputs", 1, d + 2, N, dm)
    CODES = wb.add_block("Codes", "Out", 1, 1, T, 1)
    ER = Eb.range("W_E", absolute=True)

    def x0cell(n, j):
        if n < L:
            return "=" + _lin(Win, j, WVr.row_range(n), d) + f"+{Pt.ref(n, j, 'W_postext')}"
        kind, p, _ = rows[n - L]
        if kind == "q":
            return f"={Qp.ref(p, j, 'W_qpos')}"
        return f"=INDEX({ER},{CODES.ref(p, 0, 'Out')}+1,{j + 1})+{Cp.ref(p, j, 'W_cpos')}"
    wb.fill(X0, x0cell)

    def keymask(m):
        if m < L:
            return f"Tokens!$E${1 + m}" if m > 0 else None      # prompt row 0 is always a key (start row)
        return False if rows[m - L][0] == "q" else None
    xf = emit_stack(wb, w, "", X0, N, dm, cfg["H"], cfg["mlp"], cfg["layers"], keymask,
                    is_q=lambda n: n >= L and rows[n - L][0] == "q")

    LU = None
    GC = _guidance_cell(wb, g)
    if g:
        U0 = wb.add_block("U.X0", "U_X0", 1, 1, R + 1, cfg["U_dm"])     # row 0 = learned start row
        UE = _wtab(wb, "U_W_E", w["U_E"])
        UQ = _wtab(wb, "U_W_qpos", w["U_qpos"])
        UC = _wtab(wb, "U_W_cpos", w["U_cpos"])
        UER = UE.range("U_W_E", absolute=True)
        USt = _wtab(wb, "U_W_start", w["U_start"][None, :])

        def ux0(i, j):
            if i == 0:
                return f"={USt.ref(0, j, 'U_W_start')}"
            kind, p, _ = rows[i - 1]
            if kind == "q":
                return f"={UQ.ref(p, j, 'U_W_qpos')}"
            return f"=INDEX({UER},{CODES.ref(p, 0, 'Out')}+1,{j + 1})+{UC.ref(p, j, 'U_W_cpos')}"
        wb.fill(U0, ux0)
        uq = lambda m: m > 0 and rows[m - 1][0] == "q"
        ux = emit_stack(wb, w, "U", U0, R + 1, cfg["U_dm"], cfg["U_H"], cfg["U_mlp"], cfg["U_layers"],
                        keymask=lambda m: False if uq(m) else None, tab=lambda l, part: f"U{l + 1}_{part}", is_q=uq)
        UWo = _wtab(wb, "U_W_out", w["U_Wout"].T, w["U_bout"])
        urf = wb.add_block("U.rf", "U_Out", 1, 1, T, 1)
        uxr = lambda p: ux.row_range(qrow[p] - L + 1, sheet=ux.sheet)
        wb.fill(urf, lambda p, _: f"=1/SQRT(SUMPRODUCT({uxr(p)},{uxr(p)})/{cfg['U_dm']}+{EPS!r})")
        LU = lambda p, k: _lin(UWo, k, uxr(p), cfg["U_dm"], urf.ref(p, 0, "U_Out"))   # inlined in Score (D-55)

    Wout = _wtab(wb, "W_out", w["Wout"].T, w["bout"])
    xr = lambda p: xf.row_range(qrow[p], sheet=xf.sheet)
    rf = wb.add_block("rf", "Out", 1, 3, T, 1)
    wb.fill(rf, lambda p, _: f"=1/SQRT(SUMPRODUCT({xr(p)},{xr(p)})/{dm}+{EPS!r})")
    Sc = wb.add_block("Score", "Score", 1, 1, T, Kp)

    def score(p, k):
        lc = _lin(Wout, k, xr(p), dm, rf.ref(p, 0, "Out"))
        lg = f"((1+{GC})*({lc})-{GC}*({LU(p, k)}))" if g else f"({lc})"
        u = f"(MOD({LCG_A}*MOD({S1}+{HASH_B}*{p * Kp + k + 1},{M31}),{M31})+0.5)/{M31}"
        return f"={lg}*{inv_tau!r}+(-LN(-LN({u})))"
    wb.fill(Sc, score)
    wb.fill(CODES, lambda p, _: f"=MATCH(MAX({Sc.row_range(p, sheet='Score')}),{Sc.row_range(p, sheet='Score')},0)-1")
    DR = Db.range("W_D", absolute=True)
    px = wb.add_block("Pixels", "Pixels", 1, 1, NPX, NPX)
    wb.fill(px, lambda y, x: f"=INDEX({DR},{CODES.ref((y // P) * Gd + x // P, 0, 'Out')}+1,{(y % P) * P + x % P + 1})")
    emit_output(wb, px, palette_hex, render, tk["PROMPT"])
    emit_about(wb, w, cfg, vocab_words, palette_hex, spec="src/spritesheet/reference_p.py (forward_g)")
    return wb
