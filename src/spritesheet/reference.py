"""SpriteSheet reference forward pass: numpy float64. THIS IS THE SPEC (docs/08-interfaces.md).

The spreadsheet must reproduce `forward()`: continuous values to <= 1e-12 relative, and discrete
outputs (style id, class ids) exactly. Any op used here must have a Pixcell emitter.
"""
import json
import math
import os

import numpy as np

M31 = 2147483647
LCG_A = 48271
NEG = -1e4  # additive mask for padded keys


def round15(a):
    """Round to 15 significant digits so values survive xlsx/Sheets serialisation exactly (D-09)."""
    a = np.asarray(a, dtype=np.float64)
    return np.vectorize(lambda v: float("%.15g" % v), otypes=[np.float64])(a) if a.size else a


def load(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    w = dict(np.load(os.path.join(model_dir, "weights.npz")))
    return w, cfg


def lcg_uniform(seed):
    """Two MINSTD steps (the first output is discarded: one step maps small seeds to u ≈ seed·2.2e-5; D-31)."""
    s0 = int(math.floor(abs(seed))) % (M31 - 1) + 1  # sheet: MOD(INT(ABS(seed)),2147483646)+1
    s1 = (LCG_A * s0) % M31                           # sheet: MOD(48271*s0,2147483647)
    return ((LCG_A * s1) % M31) / M31                 # sheet: MOD(48271*s1,2147483647)/2147483647


def seq_sum(v):
    """Left-to-right sum (matches a chain of cell additions / SUM over a short row)."""
    t = 0.0
    for x in v:
        t += float(x)
    return t


def softmax_rows(S):
    m = S.max(axis=-1, keepdims=True)
    e = np.exp(S - m)
    return e / e.sum(axis=-1, keepdims=True)


def conv3x3(x, W, b):
    """x [H,W,Cin], W [k,k,Cin,Cout] (k odd; 3 unless cfg k1..k3 say otherwise), zero padding k//2 → [H,W,Cout]
    (cross-correlation, like torch)."""
    H, Wd, Cin = x.shape
    k = W.shape[0]
    p = k // 2
    xp = np.zeros((H + 2 * p, Wd + 2 * p, Cin))
    xp[p:p + H, p:p + Wd] = x
    out = np.zeros((H, Wd, W.shape[3]))
    for ky in range(k):
        for kx in range(k):
            out += np.einsum("hwc,co->hwo", xp[ky:ky + H, kx:kx + Wd], W[ky, kx])
    return out + b


def upsample2(x):
    return x.repeat(2, axis=0).repeat(2, axis=1)


def encode_text(ids, w, cfg):
    ids = np.asarray(ids)
    X = w["wordvec"][ids - 1] + w["pos"]
    m = (ids != 1).astype(np.float64)
    Q = X @ w["Wq"] + w["bq"]
    K = X @ w["Wk"] + w["bk"]
    V = X @ w["Wv"] + w["bv"]
    S = (Q @ K.T) * cfg["attn_scale"] + (1.0 - m)[None, :] * NEG
    P = softmax_rows(S)
    A = X + (P @ V) @ w["Wo"] + w["bo"]
    H = A + np.maximum(0.0, A @ w["W1"] + w["b1"]) @ w["W2"] + w["b2"]
    denom = m.sum()
    c = (m[:, None] * H).sum(axis=0) / (denom if denom > 0 else 1.0)
    return c, {"X": X, "P": P, "A": A, "H": H, "mask": m}


def pick_style(c, seed, w, cfg):
    logits = c @ w["Ws"] + w["bs"]
    p = softmax_rows(logits[None, :])[0]
    u = lcg_uniform(seed)
    cum, below = 0.0, 0
    for pi in p:
        cum = cum + float(pi)
        if cum < u:
            below += 1
    style = min(below + 1, cfg["S"])  # 1-based
    return style, p, u


def decode(c, style, w, cfg):
    C0 = cfg["C0"]
    z = np.concatenate([c, w["style_emb"][style - 1]])
    G0 = cfg.get("N", 32) // 8  # dense output grid: 4x4 for 32x32, 3x3 for 24x24
    g = np.maximum(0.0, z @ w["Wd"] + w["bd"]).reshape(G0, G0, C0)
    acts = [g]
    for i in (1, 2, 3):
        g = upsample2(g)
        g = np.maximum(0.0, conv3x3(g, w[f"Wc{i}"], w[f"bc{i}"]) + c @ w[f"F{i}"])
        acts.append(g)
    logits = g @ w["Wout"] + w["bout"]
    return logits, acts


def argmax_first(logits):
    return logits.argmax(axis=-1)  # numpy returns the first max, same as MATCH(MAX(r), r, 0)


def forward(ids, seed, w, cfg):
    c, enc = encode_text(ids, w, cfg)
    style, p_style, u = pick_style(c, seed, w, cfg)
    logits, acts = decode(c, style, w, cfg)
    classes = argmax_first(logits)
    top2 = np.sort(logits, axis=-1)[..., -2:]
    return {"c": c, "style": style, "p_style": p_style, "u": u, "logits": logits, "classes": classes,
            "min_margin": float((top2[..., 1] - top2[..., 0]).min()), "acts": acts, "enc": enc}


def random_weights(cfg, seed=0, scale=0.3):
    """Random weights with the right shapes (for compiler/op parity tests before training)."""
    r = np.random.default_rng(seed)
    d, h, S, e, K = cfg["d"], cfg["h"], cfg["S"], cfg["e"], cfg["K"]
    C = [cfg["C0"], cfg["C1"], cfg["C2"], cfg["C3"]]
    g = lambda *s: r.normal(0, scale, s)
    w = {"wordvec": g(cfg["V"], d), "pos": g(cfg["L"], d),
         "Wq": g(d, d), "Wk": g(d, d), "Wv": g(d, d), "Wo": g(d, d),
         "bq": g(d), "bk": g(d), "bv": g(d), "bo": g(d),
         "W1": g(d, h), "b1": g(h), "W2": g(h, d), "b2": g(d),
         "Ws": g(d, S), "bs": g(S), "style_emb": g(S, e),
         "Wd": g(d + e, (cfg.get("N", 32) // 8) ** 2 * C[0]), "bd": g((cfg.get("N", 32) // 8) ** 2 * C[0]),
         "Wout": g(C[3], K + 1), "bout": g(K + 1)}
    w["wordvec"][0] = 0.0
    for i in (1, 2, 3):
        k = cfg.get(f"k{i}", 3)
        w[f"Wc{i}"] = g(k, k, C[i - 1], C[i]) / math.sqrt(k * k * C[i - 1]) * 3
        w[f"bc{i}"] = g(C[i])
        w[f"F{i}"] = g(d, C[i])
    return {k: round15(v) for k, v in w.items()}
