"""v3 data: PixelGPT 24x24 216K only, native 24x24 (no padding, no emoji, no styles).

Build (once; outputs in data/build/v3/, gitignored because they contain PixelGPT images):
  uv run python -m spritesheet.data.v3 build [--K 32]
    → rgba_24.npy [n,24,24,4] (cache), imgs_24.npz (class grids + split + caption), palette.json, vocab.txt,
      wordvec.npy, report.json

Training stream: Sampler3 (captions from the dataset + colour/background augmentation).

Licence: the purchased 216K pack ships LICENSE_TEMPLATE.md (training and distributing trained weights permitted;
redistributing images not). Images never leave data/ (gitignored).
"""
import argparse, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .normalize import Palette, srgb_to_lab, lab_to_srgb, ALPHA_ON

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(ROOT, "data", "raw", "PixelGPT-24x24-216K")
OUT = os.path.join(ROOT, "data", "build", "v3")
SPLITS = {"train": 0, "validation": 1, "test": 2}


def _load_png(path):
    from PIL import Image
    im = Image.open(os.path.join(RAW, path)).convert("RGBA")
    a = np.asarray(im, dtype=np.uint8)
    assert a.shape == (24, 24, 4), (path, a.shape)
    return a


def load_meta():
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, "metadata", "all.parquet"),
                         columns=["id", "caption", "split", "category_path", "image_transparent", "color_count"])
    return df.reset_index(drop=True)


def rgba_cache(df, workers=8):
    p = os.path.join(OUT, "rgba_24.npy")
    if os.path.exists(p):
        a = np.load(p, mmap_mode="r")
        if a.shape[0] == len(df):
            return a
    with ProcessPoolExecutor(workers) as ex:
        arr = np.stack(list(ex.map(_load_png, df["image_transparent"].tolist(), chunksize=512)))
    np.save(p, arr)
    return arr


def fit_palette(rgba, train_mask, K=32, n_sample=400_000, iters=40, seed=0):
    """k-means in CIELAB over opaque training pixels (pixel-weighted). Returns Palette and mean ΔE76."""
    r = np.random.default_rng(seed)
    idx = np.nonzero(train_mask)[0]
    px = rgba[idx].reshape(-1, 4)
    px = px[px[:, 3] >= ALPHA_ON][:, :3]
    s = px[r.choice(len(px), min(n_sample, len(px)), replace=False)]
    X = srgb_to_lab(s)
    C = X[r.choice(len(X), K, replace=False)]
    # k-means++ init
    C = [X[r.integers(len(X))]]
    for _ in range(K - 1):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[r.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (a == k).any():
                C[k] = X[a == k].mean(0)
    rgb = np.clip(np.round(lab_to_srgb(C)), 0, 255).astype(np.uint8)
    hexes = []
    for c in rgb:
        h = "#%02x%02x%02x" % tuple(int(v) for v in c)
        while h in hexes:  # duplicate after rounding: nudge
            c = np.clip(c.astype(int) + 1, 0, 255)
            h = "#%02x%02x%02x" % tuple(int(v) for v in c)
        hexes.append(h)
    # order: dark → light (stable, readable palette tab)
    L = srgb_to_lab(np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=np.uint8))[:, 0]
    hexes = [hexes[i] for i in np.argsort(L, kind="stable")]
    pal = Palette(hexes)
    de = float(np.sqrt(((X - pal.lab[pal.nearest(s)]) ** 2).sum(-1)).mean())
    return pal, de


def snap_all(rgba, pal, chunk=8192):
    out = np.zeros(rgba.shape[:3], np.uint8)
    for i in range(0, len(rgba), chunk):
        a = np.asarray(rgba[i:i + chunk])
        on = a[..., 3] >= ALPHA_ON
        ids = np.zeros(a.shape[:3], np.uint8)
        ids[on] = pal.nearest(a[..., :3][on]).astype(np.uint8) + 1
        out[i:i + chunk] = ids
    return out


def build(K=32, top_new_words=4000, d=64):
    from .. import tokenizer as T
    from .. import vocab as Vb
    os.makedirs(OUT, exist_ok=True)
    df = load_meta()
    print("records", len(df), flush=True)
    rgba = rgba_cache(df)
    split = df["split"].map(SPLITS).to_numpy().astype(np.int8)
    pal, de = fit_palette(rgba, split == 0, K=K)
    pal.save(os.path.join(OUT, "palette.json"), K)
    imgs = snap_all(rgba, pal)
    fg = (imgs > 0).mean((1, 2))
    caps = np.array([str(c).lower().strip() for c in df["caption"]])
    np.savez_compressed(os.path.join(OUT, "imgs_24.npz"), imgs=imgs, split=split, caption=caps,
                        category=df["category_path"].to_numpy().astype(str))
    # vocabulary: the existing v1 words (CLIP common + emoji/attribute words) plus the most frequent new
    # caption words from the 216K TRAIN split; one CLIP table/PCA over the union.
    old = [w.strip() for w in open(os.path.join(ROOT, "data", "build", "vocab.txt"))][2:]
    from collections import Counter
    cnt = Counter(w for c in caps[split == 0] for w in T.normalize(c))
    new = [w for w, _ in cnt.most_common() if w not in set(old) and re.fullmatch(r"[a-z]{2,20}", w)][:top_new_words]
    extra = sorted(set(old) | set(new))
    vb_build = Vb.BUILD
    try:
        Vb.BUILD = OUT
        vocab, wv, rep = Vb.build(extra, d=d, n_common=0)
    finally:
        Vb.BUILD = vb_build
    vset = set(vocab)
    tot = sum(cnt.values())
    cov = sum(v for w, v in cnt.items() if w in vset) / tot
    report = {"records": len(df), "splits": {k: int((split == v).sum()) for k, v in SPLITS.items()}, "K": K,
              "palette_mean_deltaE76": de, "fg_frac_mean": float(fg.mean()), "fg_frac_p10": float(np.quantile(fg, .1)),
              "fg_frac_p90": float(np.quantile(fg, .9)), "full_frame_frac": float((fg > 0.95).mean()),
              "V": len(vocab), "new_words_added": len(new), "train_token_coverage": cov, "pca_explained": rep["pca_explained"]}
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


# ---------------------------------------------------------------- training stream
PREP = {"of", "with", "in", "on", "from", "for", "under", "over", "at", "by", "into", "inside", "behind", "near",
        "beside", "through", "against", "without", "made"}


def shorten(cap, r, p_keep=0.35, p_drop_tail=0.7):
    """Head-noun-preserving caption dropout: the head noun (last word before the first preposition) is always kept;
    the prepositional tail is dropped with p_drop_tail; every other word survives with p_keep (order kept)."""
    w = cap.split()
    cut = next((i for i, x in enumerate(w) if i > 0 and x.lower() in PREP), len(w))
    head, tail = w[:cut], w[cut:]
    keep = [x for x in head[:-1] if r.random() < p_keep] + [head[-1]]
    if tail and r.random() >= p_drop_tail:
        keep += tail
    return " ".join(keep)


class Sampler3:
    """Deterministic stream of (caption, 24x24 class grid). Train-time augmentation (D-51):
    colour recolour + adjective ("blue <caption>") with p_colour, named background with p_bg, optional
    leading article. Eval splits ("validation", "test") return verbatim captions."""

    def __init__(self, split="train", seed=0, p_colour=0.25, p_bg=0.08, p_article=0.5, p_short=0.0):
        from .synth import recolor_luts, bg_classes, saturated_mask, COLORS, BG_COLORS, BG_PHRASES, _WORDS_OF
        from .. import tokenizer as T
        z = np.load(os.path.join(OUT, "imgs_24.npz"))
        self.imgs, self.caps, sp = z["imgs"], z["caption"], z["split"]
        self.pool = np.nonzero(sp == SPLITS[split])[0]
        self.split = split
        self.rng = np.random.default_rng([seed, SPLITS[split]])
        self.pal = Palette.load(os.path.join(OUT, "palette.json"))
        self.luts, self.bgc = recolor_luts(self.pal), bg_classes(self.pal)
        sat = saturated_mask(self.pal)
        fg = self.imgs > 0
        self.colorable = (sat[self.imgs] & fg).sum((1, 2)) >= 0.5 * np.maximum(1, fg.sum((1, 2)))
        self.COLORS, self.BG, self.BGP, self.WORDS_OF = list(COLORS), list(BG_COLORS), BG_PHRASES, _WORDS_OF
        self.p = (p_colour, p_bg, p_article) if split == "train" else (0.0, 0.0, 0.0)
        self.p_short = p_short if split == "train" else 0.0
        self.L = T.L
        self.nw = lambda c: len(T.normalize(c))

    def sample(self):
        r = self.rng
        b = int(self.pool[int(r.integers(len(self.pool)))])
        g, cap = self.imgs[b], str(self.caps[b])
        if self.p_short and r.random() < self.p_short:
            cap = shorten(cap, r)
        pc, pb, pa = self.p
        if self.colorable[b] and r.random() < pc and self.nw(cap) + 1 <= self.L:
            c = self.COLORS[int(r.integers(len(self.COLORS)))]
            g = self.luts[c][g]
            cap = self.WORDS_OF[c][int(r.integers(len(self.WORDS_OF[c])))] + " " + cap if c in self.WORDS_OF else c + " " + cap
        if r.random() < pb and self.nw(cap) + 4 <= self.L:
            bg = self.BG[int(r.integers(len(self.BG)))]
            g = np.where(g == 0, self.bgc[bg], g)
            ph = self.BGP[int(r.integers(len(self.BGP)))].replace("{c}", self.WORDS_OF[bg][int(r.integers(len(self.WORDS_OF[bg])))])
            cap = cap + " " + ph
        if r.random() < pa and self.nw(cap) + 1 <= self.L:
            cap = ("an " if cap[:1] in "aeiou" else "a ") + cap
        return cap, np.ascontiguousarray(g, dtype=np.uint8), b

    def batch(self, B, with_idx=False):
        out = [self.sample() for _ in range(B)]
        caps, g = [o[0] for o in out], np.stack([o[1] for o in out])
        return (caps, g, np.array([o[2] for o in out])) if with_idx else (caps, g)


def eval_set3(split="validation", n=512, seed=7):
    s = Sampler3(split, seed=seed)
    return s.batch(n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--new-words", type=int, default=4000)
    a = ap.parse_args()
    build(a.K, a.new_words)
