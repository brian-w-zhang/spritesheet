"""v3-P reference forward (numpy float64): THE SPEC for the PatchGPT sheet (src/spritesheet/patchgpt.py is the
training twin). The spreadsheet must reproduce `forward()`: continuous values to <= 1e-12 relative, codes and pixels
exactly.

Sequence rows n = 0..N-1, N = L + T - 1:
  n < L   (prompt word n):  x = wordvec[id_n] @ Win + b_in + pos_text[n]
  n >= L  (image token t = n-L, the code picked at step t):  x = E[code_t] + pos_img[t]
Per layer (pre-RMSNorm; the norm gains are folded into the next weights at export):
  r   = 1 / SQRT(SUMPRODUCT(x, x)/dm + eps)
  q,k,v = (x . W) * r + b
  e[h, n, m] = EXP(q_h[n] . k_h[m] * scale) * keymask[m]   for m < n;  EXP(...) for m = n;  0 for m > n
  ctx_h[n] = sum_m e[h,n,m] v_h[m] / sum_m e[h,n,m]
  x1 = x + ctx . Wo + bo
  r2 = 1 / SQRT(SUMPRODUCT(x1, x1)/dm + eps);  hid = MAX(0, (x1 . W1) * r2 + b1);  x2 = x1 + hid . W2 + b2
Final: rf = 1/SQRT(...);  logit[t, k] = (x_{L-1+t} . Wout) * rf + bout      (t = 0..T-1)
Optional guidance (cfg "guidance" w > 0, needs an unconditional prior "U*" weights): see forward().
Sampling: score = logit * inv_tau + (-LN(-LN(u))),  u = seeded uniform (same exact integer hash as v2);
  code_t = MATCH(MAX(score_row), score_row, 0) - 1.
Pixels: pixel[y, x] = D[code_{(y//P)*G + x//P}, (y%P)*P + x%P].
"""
import math

import numpy as np

from .reference import M31, LCG_A, round15

HASH_B = 104729
EPS = 1e-6


def seed_state(seed):
    s0 = int(math.floor(abs(seed))) % (M31 - 1) + 1     # sheet: MOD(INT(ABS(seed)),2147483646)+1
    return (LCG_A * s0) % M31                            # sheet: MOD(48271*s0,2147483647)


def uniforms(seed, T, K):
    """[T, K] exact uniforms: idx = t*K + k + 1; v = MOD(48271*MOD(s1 + 104729*idx, M31), M31); u = (v+0.5)/M31."""
    s1 = seed_state(seed)
    idx = np.arange(T)[:, None] * K + np.arange(K)[None, :] + 1
    a = (s1 + HASH_B * idx) % M31
    v = (LCG_A * a) % M31
    return (v + 0.5) / M31


def gumbel(seed, T, K):
    return -np.log(-np.log(uniforms(seed, T, K)))


def _rms(x, dm):
    return 1.0 / math.sqrt(float(x @ x) / dm + EPS)


def _row_forward(x, n, w, cfg, cache, keyok, pre=""):
    """One sequence row through every layer; appends this row's k, v to the per-layer cache. Returns final x and parts."""
    dm, H = cfg["dm"], cfg["H"]
    dh = dm // H
    scale = 1.0 / math.sqrt(dh)
    parts = []
    for l in range(cfg["layers"]):
        p = f"{pre}l{l}."
        r = _rms(x, dm)
        q = (x @ w[p + "Wq"]) * r + w[p + "bq"]
        k = (x @ w[p + "Wk"]) * r + w[p + "bk"]
        v = (x @ w[p + "Wv"]) * r + w[p + "bv"]
        cache[l]["k"].append(k)
        cache[l]["v"].append(v)
        Kc, Vc = np.array(cache[l]["k"]), np.array(cache[l]["v"])        # [n+1, dm]
        ctx = np.zeros(dm)
        es, zs = [], []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            e = np.exp((Kc[:, sl] @ q[sl]) * scale)
            e[:n] *= keyok[:n]                                          # pad keys off; a row always sees itself
            z = e.sum()
            ctx[sl] = (e @ Vc[:, sl]) / z
            es.append(e)
            zs.append(z)
        x1 = x + ctx @ w[p + "Wo"] + w[p + "bo"]
        r2 = _rms(x1, dm)
        hid = np.maximum(0.0, (x1 @ w[p + "W1"]) * r2 + w[p + "b1"])
        x2 = x1 + hid @ w[p + "W2"] + w[p + "b2"]
        parts.append({"r": r, "q": q, "k": k, "v": v, "e": es, "z": zs, "ctx": ctx, "x1": x1, "r2": r2, "hid": hid, "x2": x2})
        x = x2
    return x, parts


def forward(ids, seed, w, cfg, keep=True):
    """→ {"codes" [T], "pixels" [N,N], "scores" [T,K], "logits" [T,K], "min_margin", rows (intermediates if keep)}.
    Guidance (cfg["guidance"] = g > 0): logit = (1+g)*lc - g*lu, where lu comes from the unconditional prior
    (weights prefixed "U", same architecture with cfg["U_layers"], no prompt rows: its sequence is a learned start row
    + image rows). inv_tau = 1/cfg["tau"]."""
    ids = np.asarray(ids)
    L, T, K, dm = len(ids), cfg["T"], cfg["Kp"], cfg["dm"]
    keyok = np.concatenate([(ids != 1).astype(np.float64), np.ones(T - 1)])
    g = float(cfg.get("guidance", 0.0))
    inv_tau = 1.0 / float(cfg.get("tau", 1.0))
    G = gumbel(seed, T, K)
    cache = [{"k": [], "v": []} for _ in range(cfg["layers"])]
    ucfg = {"dm": cfg["U_dm"], "H": cfg["U_H"], "layers": cfg["U_layers"]} if g else None
    ucache = [{"k": [], "v": []} for _ in range(cfg["U_layers"])] if g else None
    codes = np.zeros(T, np.int64)
    rows, logits, scores, ulogits = [], np.zeros((T, K)), np.zeros((T, K)), np.zeros((T, K))
    margins = []
    for n in range(L + T - 1):
        if n < L:
            wv = w["wordvec"][ids[n] - 1]
            x0 = wv @ w["Win"] + w["b_in"] + w["pos_text"][n]
        else:
            t = n - L
            wv = None
            x0 = w["E"][codes[t]] + w["pos_img"][t]
        x, parts = _row_forward(x0, n, w, cfg, cache, keyok)
        row = {"wv": wv, "x0": x0, "parts": parts, "x": x}
        if n >= L - 1:
            t = n - (L - 1)
            rf = _rms(x, dm)
            lc = (x @ w["Wout"]) * rf + w["bout"]
            lg = lc
            if g:
                # unconditional prior row t: start row (t = 0) or E_U[code_{t-1}] + pos_U[t-1]
                ux0 = w["U_start"] if t == 0 else w["U_E"][codes[t - 1]] + w["U_pos"][t - 1]
                ux, uparts = _row_forward(ux0, t, w, ucfg, ucache, np.ones(T), pre="U")
                urf = _rms(ux, cfg["U_dm"])
                lu = (ux @ w["U_Wout"]) * urf + w["U_bout"]
                ulogits[t] = lu
                lg = (1.0 + g) * lc - g * lu
                row.update({"ux0": ux0, "uparts": uparts, "ux": ux, "urf": urf, "lu": lu})
            s = lg * inv_tau + G[t]
            c = int(np.argmax(s))                     # first max, like MATCH(MAX(r), r, 0)
            codes[t] = c
            top2 = np.sort(s)[-2:]
            margins.append(float(top2[1] - top2[0]))
            logits[t], scores[t] = lg, s
            row.update({"rf": rf, "lc": lc})
        if keep:
            rows.append(row)
    P = cfg["P"]
    Gd = int(round(T ** 0.5))
    D = w["D"].astype(np.int64)
    pix = D[codes].reshape(Gd, Gd, P, P).transpose(0, 2, 1, 3).reshape(Gd * P, Gd * P)
    return {"codes": codes, "pixels": pix, "logits": logits, "scores": scores, "ulogits": ulogits,
            "min_margin": min(margins), "rows": rows, "keyok": keyok, "gumbel": G}


# ---------------------------------------------------------------- export from the training twin
def export(net, D, extra_cfg=None, unet=None):
    """PatchGPT (torch) [+ unconditional PatchGPT-U] → (w, cfg) in this layout, folded norms, rounded to 15 s.d."""
    sd = {k: v.detach().double().cpu().numpy() for k, v in net.state_dict().items()}
    c = net.cfg
    dm = c["dm"]
    w = {"wordvec": sd["wordvec"], "Win": sd["win.weight"].T, "b_in": sd["win.bias"], "pos_text": sd["pos_text"],
         "E": sd["E"], "pos_img": sd["pos_img"], "D": np.asarray(D, np.float64)}
    _export_blocks(w, sd, c["layers"], dm, "")
    w["Wout"], w["bout"] = sd["nf.g"][:, None] * sd["out.weight"].T, sd["out.bias"]
    cfg = {k: c[k] for k in ("dm", "H", "mlp", "layers", "L", "T", "Kp", "P", "K", "N", "d")}
    cfg.update(model="v3p", tau=1.0, guidance=0.0)
    if unet is not None:
        su = {k: v.detach().double().cpu().numpy() for k, v in unet.state_dict().items()}
        cu = unet.cfg
        w.update({"U_start": su["start"], "U_E": su["E"], "U_pos": su["pos_img"]})
        _export_blocks(w, su, cu["layers"], cu["dm"], "U")
        w["U_Wout"], w["U_bout"] = su["nf.g"][:, None] * su["out.weight"].T, su["out.bias"]
        cfg.update(U_dm=cu["dm"], U_H=cu["H"], U_mlp=cu["mlp"], U_layers=cu["layers"])
    cfg.update(extra_cfg or {})
    return {k: round15(v) for k, v in w.items()}, cfg


def _export_blocks(w, sd, layers, dm, pre):
    for l in range(layers):
        b = f"blocks.{l}."
        g1, g2 = sd[b + "n1.g"], sd[b + "n2.g"]
        Wqkv, bqkv = sd[b + "qkv.weight"].T, sd[b + "qkv.bias"]       # [dm, 3dm]
        p = f"{pre}l{l}."
        for i, nm in enumerate("qkv"):
            w[p + f"W{nm}"] = g1[:, None] * Wqkv[:, i * dm:(i + 1) * dm]
            w[p + f"b{nm}"] = bqkv[i * dm:(i + 1) * dm]
        w[p + "Wo"], w[p + "bo"] = sd[b + "o.weight"].T, sd[b + "o.bias"]
        w[p + "W1"], w[p + "b1"] = g2[:, None] * sd[b + "f1.weight"].T, sd[b + "f1.bias"]
        w[p + "W2"], w[p + "b2"] = sd[b + "f2.weight"].T, sd[b + "f2.bias"]
