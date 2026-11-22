"""Pixcell v3-P transformer stack as per-row array formulas (MMULT build, D-59).

Same maths, same block names and shapes as emit_p.emit_stack, so trace_p / the A2 oracles check it unchanged. The
difference is the formula count: each row of a linear layer is ONE spilled formula
    =ARRAYFORMULA(MMULT(x_row, W)*r + b_row)
instead of dm SUMPRODUCT cells, and attention is one spill per (row, head). Rows stay separate formulas, so the
causal chain (row t's input = the code picked at row t-1) has no circular reference.

Browser timing (docs/09 speed experiment 1): per-row MMULT recalculated about 2x faster than per-cell SUMPRODUCT
on a 6-layer dm128 chain.

Weight tabs here are stored in x-by-out orientation with the bias as the last row ("Wm" tabs), unlike emit_p's
transposed tabs, so MMULT can take them directly.
"""
import math

import numpy as np

from .emit import _wtab
from .workbook import Block, Spill, Workbook

EPS = 1e-6


def wtab_io(wb: Workbook, name: str, W, b=None):
    """Weight tab: rows 0..in-1 = W (in x out), row `in` = bias. Returns (W range, bias row range) as absolute refs."""
    W = np.asarray(W, np.float64)
    blk = _wtab(wb, name, np.concatenate([W, np.asarray(b, np.float64)[None, :]]) if b is not None else W)
    n_in, n_out = W.shape
    wr = Block(blk.sheet, blk.r0, blk.c0, n_in, n_out).range(blk.sheet, absolute=True)
    br = Block(blk.sheet, blk.r0 + n_in, blk.c0, 1, n_out).range(blk.sheet, absolute=True) if b is not None else None
    return wr, br


def lin_row(xrow: str, wr: str, br: str | None, scale: str | None = None, add: str | None = None) -> str:
    """ARRAYFORMULA body for [add +] MMULT(x, W)[*scale][+ b] (same operation order as reference_p)."""
    s = f"MMULT({xrow},{wr})"
    if scale:
        s += f"*{scale}"
    if add:
        s = f"{add}+{s}"
    if br:
        s += f"+{br}"
    return s


def emit_stack_mm(wb: Workbook, w, pre: str, X0: Block, N: int, dm: int, H: int, mlp: int, layers: int,
                  keymask=lambda m: None, tab=lambda l, part: f"Blk{l + 1}_{part}") -> Block:
    """Causal transformer blocks over the N rows of X0, one array formula per row (per head for attention).
    keymask(m): cell ref of key m's 0/1 mask, or None (always a key). Returns the final x block."""
    dh = dm // H
    scale = 1.0 / math.sqrt(dh)
    x = X0
    # key mask column (shared by all layers): 1 or the prompt token's mask cell
    km = None
    for l in range(layers):
        p, wp = f"{pre}b{l}.", f"{pre}l{l}."
        WQ = wtab_io(wb, tab(l, "Wq"), w[wp + "Wq"], w[wp + "bq"])
        WK = wtab_io(wb, tab(l, "Wk"), w[wp + "Wk"], w[wp + "bk"])
        WV = wtab_io(wb, tab(l, "Wv"), w[wp + "Wv"], w[wp + "bv"])
        WO = wtab_io(wb, tab(l, "Wo"), w[wp + "Wo"], w[wp + "bo"])
        W1 = wtab_io(wb, tab(l, "W1"), w[wp + "W1"], w[wp + "b1"])
        W2 = wtab_io(wb, tab(l, "W2"), w[wp + "W2"], w[wp + "b2"])
        sq, xs = tab(l, "QKV"), x.sheet
        if l == 0 and any(keymask(m) for m in range(N)):
            km = wb.add_block(f"{pre}kmask", sq, 1, 1, N, 1)
            wb.fill(km, lambda m, _: f"={keymask(m)}" if keymask(m) else 1.0)
        c0 = 3
        rb = wb.add_block(p + "r", sq, 1, 2, N, 1)
        xrow = lambda n, x=x, xs=xs: x.row_range(n, sheet=xs)
        wb.fill(rb, lambda n, _, xrow=xrow: f"=1/SQRT(SUMPRODUCT({xrow(n)},{xrow(n)})/{dm}+{EPS!r})")
        Q = wb.add_block(p + "Q", sq, 1, c0, N, dm)
        K = wb.add_block(p + "K", sq, 1, c0 + dm + 1, N, dm)
        V = wb.add_block(p + "V", sq, 1, c0 + 2 * (dm + 1), N, dm)
        for blk, (wr, br) in ((Q, WQ), (K, WK), (V, WV)):
            for n in range(N):
                wb.set(blk.sheet, blk.r0 + n, blk.c0,
                       Spill(f"=ARRAYFORMULA({lin_row(xrow(n), wr, br, rb.ref(n, 0))})", 1, dm))
        sa = tab(l, "Att")
        Es = []
        Zb = wb.add_block(p + "Z", sa, N + 2, 1, N, H)
        for h in range(H):
            Eb = wb.add_block(p + f"E{h}", sa, 1, 1 + h * (N + 1), N, N)
            Es.append(Eb)
            for n in range(N):
                qh = Q.row_range(n, h * dh, h * dh + dh - 1, sq)
                if n > 0:   # keys m < n: one spill; masked by the key mask (prompt pads), like the spec
                    kh = Block(sq, K.r0, K.c0 + h * dh, n, dh).range(sq)
                    body = f"EXP(MMULT({qh},TRANSPOSE({kh}))*{scale!r})"
                    if km is not None:
                        body += f"*TRANSPOSE({Block(km.sheet, km.r0, km.c0, n, 1).range(km.sheet)})"
                    wb.set(sa, Eb.r0 + n, Eb.c0, Spill(f"=ARRAYFORMULA({body})", 1, n))
                # the row itself (never masked)
                wb.set(sa, Eb.r0 + n, Eb.c0 + n,
                       f"=EXP(SUMPRODUCT({qh},{K.row_range(n, h * dh, h * dh + dh - 1, sq)})*{scale!r})")
            wb.fill(Block(sa, Zb.r0, Zb.c0 + h, N, 1), lambda n, _, Eb=Eb: f"=SUM({Eb.row_range(n, 0, n)})")
        sc = tab(l, "Ctx")
        C = wb.add_block(p + "ctx", sc, 1, 1, N, dm)
        for n in range(N):
            for h in range(H):
                e = Es[h].row_range(n, 0, n, sa)
                v = Block(sq, V.r0, V.c0 + h * dh, n + 1, dh).range(sq)
                wb.set(sc, C.r0 + n, C.c0 + h * dh, Spill(f"=ARRAYFORMULA(MMULT({e},{v})/{Zb.ref(n, h, sa)})", 1, dh))
        X1 = wb.add_block(p + "x1", sc, N + 2, 1, N, dm)
        for n in range(N):
            wb.set(sc, X1.r0 + n, X1.c0, Spill(f"=ARRAYFORMULA({lin_row(C.row_range(n), *WO, add=x.row_range(n, sheet=xs))})", 1, dm))
        sm = tab(l, "MLP")
        r2 = wb.add_block(p + "r2", sm, 1, 1, N, 1)
        x1row = lambda n: X1.row_range(n, sheet=sc)
        wb.fill(r2, lambda n, _: f"=1/SQRT(SUMPRODUCT({x1row(n)},{x1row(n)})/{dm}+{EPS!r})")
        Hp = wb.add_block(p + "hpre", sm, 1, 3, N, mlp)
        Hd = wb.add_block(p + "hid", sm, N + 2, 3, N, mlp)
        X2 = wb.add_block(p + "x2", sm, 2 * N + 3, 3, N, dm)
        for n in range(N):
            wb.set(sm, Hp.r0 + n, Hp.c0, Spill(f"=ARRAYFORMULA({lin_row(x1row(n), *W1, scale=r2.ref(n, 0))})", 1, mlp))
            hp = Hp.row_range(n)
            wb.set(sm, Hd.r0 + n, Hd.c0, Spill(f"=ARRAYFORMULA(IF({hp}>0,{hp},0))", 1, mlp))
            wb.set(sm, X2.r0 + n, X2.c0, Spill(f"=ARRAYFORMULA({lin_row(Hd.row_range(n), *W2, add=X1.row_range(n, sheet=sc))})", 1, dm))
        x = X2
    return x
