"""Expected value of every Workbook block, from the numpy reference (the spec), laid out like the tabs.

Uses reference.forward for every quantity it exposes and recomputes the remaining intermediates with the
same expressions (Q, K, V, scores, softmax parts, MLP hidden, style-softmax parts, FiLM, upsampled grids).
"""
import numpy as np

from .. import reference as R
from .. import tokenizer as T


def front_expected(prompt, seed, w, cfg, vocab_index, c, enc, style, p_style, u) -> dict:
    """Expected Tokens / encoder / Latent blocks (shared by v1 and v2). z = cond = [c, style_emb[style]]."""
    clean, rests, words = T.split_trace(prompt)
    ids = T.encode(prompt, vocab_index)
    X, m = enc["X"], enc["mask"]
    Q = X @ w["Wq"] + w["bq"]
    K = X @ w["Wk"] + w["bk"]
    V = X @ w["Wv"] + w["bv"]
    S = (Q @ K.T) * cfg["attn_scale"] + (1.0 - m)[None, :] * R.NEG
    smax = S.max(axis=-1, keepdims=True)
    sexp = np.exp(S - smax)
    O = enc["P"] @ V
    hid = np.maximum(0.0, enc["A"] @ w["W1"] + w["b1"])
    sl = c @ w["Ws"] + w["bs"]
    slx = np.exp(sl - sl.max())
    cum = np.cumsum(p_style)  # sequential for a 1-D cumsum
    s0 = int(abs(seed)) % (R.M31 - 1) + 1
    s1 = (R.LCG_A * s0) % R.M31
    z = np.concatenate([c, w["style_emb"][style - 1]])
    return {
        "Tokens.clean": [[clean]], "Tokens.rest": [[r] for r in rests], "Tokens.word": [[x] for x in words],
        "Tokens.ids": np.array(ids, float)[:, None], "Tokens.mask": m[:, None],
        "Tokens.denom": np.array([[max(1.0, m.sum())]]),
        "X": X, "Q": Q, "K": K, "VT": V.T, "Scores": S, "Scores.scale": np.array([[cfg["attn_scale"]]]),
        "Softmax.max": smax, "Softmax.exp": sexp, "Softmax.sum": sexp.sum(-1, keepdims=True), "P": enc["P"],
        "Ctx": O, "A": enc["A"], "MLP": hid, "H": enc["H"],
        "c": c[None, :], "style_logits": sl[None, :], "style.max": np.array([[sl.max()]]),
        "style.sum": np.array([[slx.sum()]]), "style.exp": slx[None, :], "p_style": p_style[None, :],
        "cum": cum[None, :], "s0": np.array([[s0]], float), "s1": np.array([[s1]], float), "u": np.array([[u]]),
        "style": np.array([[style]], float), "z": z[None, :],
    }


def trace(prompt, seed, w, cfg, vocab_index) -> dict:
    ids = T.encode(prompt, vocab_index)
    out = R.forward(ids, seed, w, cfg)
    c = out["c"]
    E = front_expected(prompt, seed, w, cfg, vocab_index, c, out["enc"], out["style"], out["p_style"], out["u"])
    acts = out["acts"]
    for i, g in enumerate(acts):
        E[f"G{i}"] = g.reshape(g.shape[0], -1)
    for i in (1, 2, 3):
        zz = np.concatenate([c, w["style_emb"][out["style"] - 1]])
        cond = zz if w[f"F{i}"].shape[0] == len(zz) else c
        E[f"FiLM{i}"] = (cond @ w[f"F{i}"] + w[f"bc{i}"])[None, :]
        up = R.upsample2(acts[i - 1])
        pk = np.asarray(w[f"Wc{i}"]).shape[0] // 2  # zero border = kernel radius
        E[f"Up_{i}"] = np.pad(up, ((pk, pk), (pk, pk), (0, 0))).reshape(up.shape[0] + 2 * pk, -1)
    lg = out["logits"]
    E["Logits"] = lg.reshape(lg.shape[0], -1)
    E["Pixels"] = out["classes"].astype(float)
    return {"ids": ids, "expected": E, "forward": out}


def _pad(a):
    """H x W x C -> zero-bordered (H+2) x ((W+2)*C), the interleaved layout of bordered tabs."""
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)))
    return p.reshape(p.shape[0], -1)


def _flat(a):
    return a.reshape(a.shape[0], -1)


def trace_mdm(prompt, seed, w, cfg, vocab_index) -> dict:
    """v2: expected value of every block of every denoising step (spec: mdm.ref_generate / ref_denoise).

    The intermediates are recomputed with ref_denoise's expressions; the logits and canvases are asserted to be
    bit-identical to ref_generate's, so the trace cannot drift from the spec."""
    from .. import mdm as D
    ids = T.encode(prompt, vocab_index)
    c, enc = R.encode_text(ids, w, cfg)
    style, p_style, u = R.pick_style(c, seed, w, cfg)
    E = front_expected(prompt, seed, w, cfg, vocab_index, c, enc, style, p_style, u)
    gen = D.ref_generate(ids, seed, w, cfg)
    assert gen["style"] == style
    cond = np.concatenate([c, w["style_emb"][style - 1]])
    f = {i: cond @ w[f"F{i}"] + w[f"f{i}"] for i in range(1, 7)}
    for i in range(1, 7):
        E[f"FiLM{i}"] = f[i][None, :]
    N, K, TT = cfg["N"], cfg["K"], cfg["T"]
    G = None
    if "Wg" in w:   # v2b global map, same for every step (mdm.ref_denoise)
        G = (cond @ w["Wg"] + w["bg"]).reshape(N // 4, N // 4, cfg["C3"])
        E["GMap"] = _flat(G)
    M = K + 1
    x = np.full((N, N), M, dtype=np.int64)
    margins, gaps, committed = [], [], []
    for t in range(1, TT + 1):
        h0 = w["Emb"][x]
        a1 = np.maximum(0.0, R.conv3x3(h0, w["Wc1"], 0.0) + f[1])
        q1 = D.avg2(a1)
        a2 = np.maximum(0.0, R.conv3x3(q1, w["Wc2"], 0.0) + f[2])
        q2 = D.avg2(a2)
        a3 = np.maximum(0.0, R.conv3x3(q2, w["Wc3"], 0.0) + f[3])
        if G is not None:
            a3 = a3 + G
        a4 = np.maximum(0.0, R.conv3x3(a3, w["Wc4"], 0.0) + f[4])
        p2 = a4 @ w["P2"]
        up2 = R.upsample2(p2)
        u2 = np.maximum(0.0, R.conv3x3(up2, w["Wc5"], 0.0) + a2 + f[5])
        p1 = u2 @ w["P1"]
        up1 = R.upsample2(p1)
        u1 = np.maximum(0.0, R.conv3x3(up1, w["Wc6"], 0.0) + a1 + f[6])
        lg = u1 @ w["Wout"] + w["bout"]
        st = gen["steps"][t - 1]
        assert np.array_equal(lg, st["logits"]), "trace_mdm drifted from mdm.ref_denoise"
        mx = lg.max(-1)
        conf = 1.0 / np.exp(lg - mx[..., None]).sum(-1)
        masked = x == M
        tp = float(cfg.get("temp", 0.0) or 0.0)
        if tp > 0:   # seeded Gumbel-max: the discrete choice is the argmax of the score, so its margin is reported
            score = lg + tp * (-np.log(-np.log(D.seeded_uniform(D.seed_state(seed), t, N, K + 1))))
            am = score.argmax(-1)
            conf = np.take_along_axis(np.exp(lg - mx[..., None]), am[..., None], -1)[..., 0] * conf
            E[f"S{t}.score"] = _flat(score)
            top2 = np.sort(score, -1)[..., -2:]
        else:
            am = lg.argmax(-1)
            top2 = np.sort(lg, -1)[..., -2:]
        margins.append(float((top2[..., 1] - top2[..., 0])[masked].min()) if masked.any() else None)
        tau = cfg["tau"][t - 1] if t < TT else None
        gaps.append(float(np.abs(conf - tau)[masked].min()) if (tau is not None and masked.any()) else None)
        x = st["canvas"]
        committed.append(int((masked & (x != M)).sum()))
        pre = f"S{t}."
        E.update({pre + "emb": _pad(h0), pre + "a1": _flat(a1), pre + "pool1": _pad(q1), pre + "a2": _flat(a2),
                  pre + "pool2": _pad(q2), pre + "a3": _pad(a3), pre + "a4": _flat(a4), pre + "p2": _flat(p2),
                  pre + "up2": _pad(up2), pre + "u2": _flat(u2), pre + "p1": _flat(p1), pre + "up1": _pad(up1),
                  pre + "u1": _flat(u1), pre + "logits": _flat(lg), pre + "max": mx, pre + "conf": conf,
                  pre + "argmax": am.astype(float), pre + "canvas": x.astype(float)})
    assert np.array_equal(x, gen["classes"])
    E["Pixels"] = x.astype(float)
    ok_m = [m for m in margins if m is not None]
    fwd = {"style": style, "classes": x, "c": c, "u": u, "p_style": p_style, "steps": gen["steps"],
           "margins": margins, "tau_gaps": gaps, "committed": committed,
           "min_margin": float(min(ok_m)) if ok_m else float("inf")}
    return {"ids": ids, "expected": E, "forward": fwd}


DISCRETE = {"Tokens.ids", "Tokens.mask", "Tokens.denom", "s0", "s1", "style", "Pixels"}


def is_discrete(name: str) -> bool:
    return name in DISCRETE or name.endswith((".canvas", ".argmax"))
TEXT = {"Tokens.clean", "Tokens.rest", "Tokens.word"}
SKIP = {"Display", "Vocab", "Palette", "Hex"}  # inputs / render; Hex checked via Pixels (and Film{t})


def compare(wb, values: dict, expected: dict, skip_weights=True) -> dict:
    """values: {sheet: 2-D grid (0-based) of engine results}. Returns per-block errors and failures.

    Continuous blocks: max |a-b| / max(1, |b|). Discrete: exact. Text: exact.
    """
    res = {"blocks": {}, "fail": [], "errors": 0, "max_rel": 0.0}
    for name, blk in wb.blocks.items():
        if name in SKIP or name not in expected:
            continue
        if blk.sheet not in values:
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
                if is_discrete(name):
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
