"""Small/realistic random models for compiler tests and spikes (random weights, synthetic vocab, palette)."""
import math
import random

from .. import reference as R
from .. import tokenizer as T

SMALL = dict(N=32, L=T.L, V=60, d=16, h=32, S=4, e=4, C0=8, C1=8, C2=8, C3=8, K=8)
V32 = dict(N=32, L=T.L, V=5000, d=32, h=64, S=8, e=8, C0=64, C1=48, C2=32, C3=24, K=32)
BASE_WORDS = ["a", "red", "apple", "blue", "cat", "dog", "green", "tree", "big", "small", "the", "of", "with",
              "pixel", "art", "3d", "emoji", "house", "sun", "moon", "star", "fish", "car", "heart", "yellow",
              "2", "10", "sword", "hat", "flower"]


def make_cfg(base=SMALL, **kw):
    cfg = dict(base, **kw)
    cfg["attn_scale"] = float("%.15g" % (1 / math.sqrt(cfg["d"])))
    return cfg


def vocab_words(V, seed=0):
    rnd = random.Random(seed)
    words = ["<pad>", "<unk>"] + BASE_WORDS[: max(0, V - 2)]
    seen = set(words)
    while len(words) < V:
        w = "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(rnd.randint(2, 9)))
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words[:V]


def palette(K, seed=0):
    rnd = random.Random(seed)
    return ["#%02x%02x%02x" % (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(K)]


def random_model(cfg, seed=0):
    w = R.random_weights(cfg, seed=seed)
    return w, vocab_words(cfg["V"], seed), palette(cfg["K"], seed)


def adversarial_prompts(n, vocab, seed=0, long_words=True):
    """Tokenizer stress prompts: caps, all ASCII punctuation (incl. MATCH wildcards * ? ~), space runs, unknown
    words, >12 words, digits, empty/blank, cumulative length > 100, near-limit and (optionally) >100-char words."""
    rnd = random.Random(seed)
    known = [w for w in vocab[2:]]
    punct = T.PUNCT
    fixed = ["", " ", "     ", "a red apple", "A RED APPLE", "  a   red    apple  ", "*", "?", "~", "a*", "*apple",
             "app?e", "~apple", "a~b c?d e*f", "apple!!!", "!!!", "'\"", "\\", "\\a\\b\\", "a-b_c+d=e", "100 2 3d",
             "3D emoji", "x" * 98, " ".join(["dog"] * 20), " ".join(["pixel"] * 13), "<pad>", "<unk>", "pad unk",
             "=1+1", "'quoted'", "\"double\"", "true", "FALSE", "1", "1/2", "12:30", "1e5", "-3", "#ref!", "#n/a",
             "a.b.c.d", " ".join(f"word{i}" for i in range(15)), " ".join("abcdefghij" for _ in range(14)),
             "a " * 30]
    if long_words:
        fixed += ["y" * 100, "z" * 150 + " apple", "q" * 99 + " " + "r" * 101 + " red", "s" * 250]
    out = list(fixed)
    while len(out) < n:
        k = rnd.randint(0, 18)
        toks = []
        for _ in range(k):
            r = rnd.random()
            if r < 0.45 and known:
                wd = rnd.choice(known)
            elif r < 0.7:
                wd = "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(rnd.randint(1, 12)))
            elif r < 0.8:
                wd = "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(20, 60)))
            else:
                wd = "".join(rnd.choice(punct) for _ in range(rnd.randint(1, 3)))
            if rnd.random() < 0.25:
                wd = wd.upper() if rnd.random() < 0.5 else wd.capitalize()
            if rnd.random() < 0.25:
                wd = wd + rnd.choice(punct)
            if rnd.random() < 0.15:
                wd = rnd.choice(punct) + wd
            toks.append(wd)
        sep = [" " * rnd.choice([1, 1, 1, 2, 3, 5]) for _ in toks]
        s = "".join(t + sp for t, sp in zip(toks, sep))
        if rnd.random() < 0.3:
            s = " " * rnd.randint(1, 3) + s
        if rnd.random() < 0.5:
            s = s.rstrip()
        out.append(s)
    return out[:n]


# ---------------------------------------------------------------- v2 (masked discrete diffusion) random models
MDM_SMALL = dict(N=32, L=T.L, V=60, d=16, h=32, S=4, e=4, K=8, E=8, C1=8, C2=8, C3=8, T=3, tau=[0.6, 0.4, 0.0])


def random_mdm(cfg, seed=0, wordvec=None, logit_scale=1.0):
    """Random-init mdm.MDMNet exported to reference layout (valid shapes, weights rounded to 15 digits).

    cfg needs N, L, d, h, S, e, K, E, C1..C3, T, tau and V (when wordvec is None). logit_scale multiplies W_out /
    b_out so that random logits are peaked enough for some pixels to pass tau before the last step.
    Returns (w, cfg) with cfg["model"] = "mdm"."""
    import numpy as np
    import torch

    from .. import mdm as D
    torch.manual_seed(seed)
    if wordvec is None:
        wordvec = np.random.default_rng(seed).normal(0, 0.5, (cfg["V"], cfg["d"]))
        wordvec[0] = 0.0
    net = D.MDMNet(cfg, np.asarray(wordvec, np.float64))
    w, c2 = net.export()
    if logit_scale != 1.0:
        w["Wout"] = R.round15(w["Wout"] * logit_scale)
        w["bout"] = R.round15(w["bout"] * logit_scale)
    c2["model"] = "mdm"
    c2["tau"] = [float(x) for x in cfg["tau"]]
    return w, c2
