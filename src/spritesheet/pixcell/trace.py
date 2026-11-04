"""Expected value of every Workbook block, from the numpy reference (the spec), laid out like the tabs.

Uses reference.forward for every quantity it exposes and recomputes the remaining intermediates with the
same expressions (Q, K, V, scores, softmax parts, MLP hidden, style-softmax parts, FiLM, upsampled grids).
"""
import numpy as np

from .. import reference as R
from .. import tokenizer as T


def trace(prompt, seed, w, cfg, vocab_index) -> dict:
    ids = T.encode(prompt, vocab_index)
    clean, rests, words = T.split_trace(prompt)
    out = R.forward(ids, seed, w, cfg)
    enc = out["enc"]
    X, m = enc["X"], enc["mask"]
    Q = X @ w["Wq"] + w["bq"]
    K = X @ w["Wk"] + w["bk"]
    V = X @ w["Wv"] + w["bv"]
    S = (Q @ K.T) * cfg["attn_scale"] + (1.0 - m)[None, :] * R.NEG
    smax = S.max(axis=-1, keepdims=True)
    sexp = np.exp(S - smax)
    O = enc["P"] @ V
    hid = np.maximum(0.0, enc["A"] @ w["W1"] + w["b1"])
    c = out["c"]
    sl = c @ w["Ws"] + w["bs"]
    slx = np.exp(sl - sl.max())
    cum = np.cumsum(out["p_style"])  # sequential for a 1-D cumsum
    s0 = int(abs(seed)) % (R.M31 - 1) + 1
    s1 = (R.LCG_A * s0) % R.M31
    z = np.concatenate([c, w["style_emb"][out["style"] - 1]])
    E = {
        "Tokens.clean": [[clean]], "Tokens.rest": [[r] for r in rests], "Tokens.word": [[x] for x in words],
        "Tokens.ids": np.array(ids, float)[:, None], "Tokens.mask": m[:, None],
        "Tokens.denom": np.array([[max(1.0, m.sum())]]),
        "X": X, "Q": Q, "K": K, "VT": V.T, "Scores": S, "Scores.scale": np.array([[cfg["attn_scale"]]]),
        "Softmax.max": smax, "Softmax.exp": sexp, "Softmax.sum": sexp.sum(-1, keepdims=True), "P": enc["P"],
        "Ctx": O, "A": enc["A"], "MLP": hid, "H": enc["H"],
        "c": c[None, :], "style_logits": sl[None, :], "style.max": np.array([[sl.max()]]),
        "style.sum": np.array([[slx.sum()]]), "style.exp": slx[None, :], "p_style": out["p_style"][None, :],
        "cum": cum[None, :], "s0": np.array([[s0]], float), "s1": np.array([[s1]], float), "u": np.array([[out["u"]]]),
        "style": np.array([[out["style"]]], float), "z": z[None, :],
    }
    acts = out["acts"]
    for i, g in enumerate(acts):
        E[f"G{i}"] = g.reshape(g.shape[0], -1)
    for i in (1, 2, 3):
        E[f"FiLM{i}"] = (c @ w[f"F{i}"] + w[f"bc{i}"])[None, :]
        up = R.upsample2(acts[i - 1])
        E[f"Up_{i}"] = np.pad(up, ((1, 1), (1, 1), (0, 0))).reshape(up.shape[0] + 2, -1)
    lg = out["logits"]
    E["Logits"] = lg.reshape(lg.shape[0], -1)
    E["Pixels"] = out["classes"].astype(float)
    return {"ids": ids, "expected": E, "forward": out}


DISCRETE = {"Tokens.ids", "Tokens.mask", "Tokens.denom", "s0", "s1", "style", "Pixels"}
TEXT = {"Tokens.clean", "Tokens.rest", "Tokens.word"}
SKIP = {"Display", "Vocab", "Palette", "Hex"}  # inputs / render; Hex checked via Pixels


def compare(wb, values: dict, expected: dict, skip_weights=True) -> dict:
    """values: {sheet: 2-D grid (0-based) of engine results}. Returns per-block errors and failures.

    Continuous blocks: max |a-b| / max(1, |b|). Discrete: exact. Text: exact.
    """
    res = {"blocks": {}, "fail": [], "errors": 0, "max_rel": 0.0}
    for name, blk in wb.blocks.items():
        if name in SKIP or name not in expected:
            continue
        grid = values.get(blk.sheet)
        exp = expected[name]
        bad, worst = 0, 0.0
        for i in range(blk.nrows):
            for j in range(blk.ncols):
                r, cidx = blk.r0 + i - 1, blk.c0 + j - 1
                try:
                    v = grid[r][cidx]
                except (IndexError, TypeError):
                    v = None
                x = exp[i][j]
                if name in TEXT:
                    ok = (v if v is not None else "") == x
                    if not ok:
                        bad += 1
                    continue
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    if isinstance(v, str) and v.startswith("#"):
                        res["errors"] += 1
                    if v is None and float(x) == 0.0:   # empty border cells read back as empty
                        continue
                    bad += 1
                    continue
                if name in DISCRETE:
                    if float(v) != float(x):
                        bad += 1
                    continue
                err = abs(float(v) - float(x)) / max(1.0, abs(float(x)))
                worst = max(worst, err)
                if not err <= 1e-12:
                    bad += 1
        res["blocks"][name] = {"max_rel": worst, "bad": bad}
        res["max_rel"] = max(res["max_rel"], worst)
        if bad:
            res["fail"].append(name)
    return res
