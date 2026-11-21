"""Slot factorisation (v4 idea): every PixelGPT sprite has <= 4 colours, so sprite = (palette of 4 colour ids, sorted
dark -> light; 0 = unused) + slot map (24x24 over 0..4; 0 = transparent, s = palette[s-1]). Exact and invertible.

Shape patches over slots carry no colour, so one dictionary entry ("dark dot on body colour") serves every colour.
"""
import json, os

import numpy as np

from .data.v3 import OUT as V3

NS = 4   # palette slots


def luminance_order():
    """palette class id (1..48) → luminance rank, from the v3 palette (CIE L*)."""
    from .data.normalize import srgb_to_lab
    pal = json.load(open(os.path.join(V3, "palette.json")))["colors"]
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)
    L = srgb_to_lab(rgb)[:, 0]
    return np.concatenate([[-1.0], L])    # index by class id; 0 = transparent


def split(grids, Lum=None):
    """grids [n,24,24] class ids → pal [n,4] (class ids, dark→light, 0-padded), slot maps [n,24,24] in 0..4.
    Sprites with > 4 colours keep the 4 most frequent; the rest map to the nearest-luminance kept slot."""
    Lum = luminance_order() if Lum is None else Lum
    n = len(grids)
    pal = np.zeros((n, NS), np.uint8)
    smap = np.zeros_like(grids)
    for i, g in enumerate(grids):
        u, c = np.unique(g[g > 0], return_counts=True)
        if len(u) > NS:
            u = u[np.argsort(-c)[:NS]]
        u = u[np.argsort(Lum[u], kind="stable")]
        pal[i, :len(u)] = u
        lut = np.zeros(256, np.uint8)
        lut[u] = np.arange(1, len(u) + 1)
        m = lut[g]
        stray = (g > 0) & (m == 0)
        if stray.any():
            for v in np.unique(g[stray]):
                lut[v] = 1 + int(np.argmin(np.abs(Lum[u] - Lum[v])))
            m = lut[g]
        smap[i] = m
    return pal, smap


def join(pal, smap):
    """inverse of split: pal [n,4], smap [n,24,24] → class grids."""
    full = np.concatenate([np.zeros((len(pal), 1), np.uint8), pal], 1)
    return np.take_along_axis(full, smap.reshape(len(smap), -1).astype(np.int64), 1).reshape(smap.shape).astype(np.uint8)


def fit_dict(K=1024, P=3, n_sprites=120_000, iters=12, seed=0, log=print):
    """k-modes over slot-map patches (5 symbols), fit on the augmented training stream (as patches.fit)."""
    import time
    from .data.v3 import Sampler3
    from .patches import to_patches, assign
    s = Sampler3("train", seed=seed)
    _, g = s.batch(n_sprites)
    _, sm = split(g)
    pt = to_patches(sm, P).reshape(-1, P * P)
    U, cnt = np.unique(pt, axis=0, return_counts=True)
    K = min(K, len(U))
    D = U[np.argsort(-cnt)[:K]].copy()
    w = cnt.astype(np.float64)
    hist, t0 = [], time.time()
    for it in range(iters):
        a = assign(U, D, NS + 1)
        idx = (a[:, None] * (P * P) + np.arange(P * P)[None]) * (NS + 1) + U
        votes = np.bincount(idx.ravel(), weights=np.repeat(w, P * P), minlength=K * P * P * (NS + 1)).reshape(K, P * P, NS + 1)
        used = votes.sum((1, 2)) > 0
        D[used] = votes[used].argmax(-1)
        mism = (U != D[a]).sum(1)
        if (~used).any():
            D[~used] = U[np.argsort(-(mism * w))[: int((~used).sum())]]
        hist.append({"iter": it, "pixel_err": round(float((mism * w).sum() / (w.sum() * P * P)), 5), "s": round(time.time() - t0, 1)})
        log(hist[-1])
    return D.astype(np.uint8), hist


class SlotCodec:
    """tokens = [4 palette tokens: Ks + class id (0..48)] + [T shape tokens: 0..Ks-1]."""

    def __init__(self, D, P, n_classes=49):
        self.D, self.P, self.Ks, self.nc = D, P, len(D), n_classes
        self.T = NS + (24 // P) ** 2
        self.V = self.Ks + n_classes

    def encode(self, grids):
        import torch
        from .patches import to_patches
        pal, sm = split(np.asarray(grids))
        pt = torch.as_tensor(to_patches(sm, self.P), dtype=torch.long)
        Dh = torch.nn.functional.one_hot(torch.as_tensor(self.D, dtype=torch.long), NS + 1).flatten(1).float()
        x = torch.nn.functional.one_hot(pt, NS + 1).flatten(2).float()
        sc = (x @ Dh.T).argmax(-1).numpy()
        return np.concatenate([pal.astype(np.int64) + self.Ks, sc], 1)

    def decode(self, tok):
        from .patches import from_patches
        tok = np.asarray(tok)
        pal = np.clip(tok[:, :NS] - self.Ks, 0, self.nc - 1).astype(np.uint8)
        sm = from_patches(self.D[np.clip(tok[:, NS:], 0, self.Ks - 1)], self.P)
        return join(pal, sm)

    def allowed(self):
        """[T, V] bool: palette rows may only pick palette tokens, shape rows only shape tokens."""
        a = np.zeros((self.T, self.V), bool)
        a[:NS, self.Ks:] = True
        a[NS:, :self.Ks] = True
        return a


if __name__ == "__main__":
    import argparse
    from .patches import from_patches, to_patches, assign
    from .data.v3 import eval_set3
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--P", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D, hist = fit_dict(a.K, a.P)
    codec = SlotCodec(D, a.P)
    _, g = eval_set3("validation", 2048)
    rec = codec.decode(codec.encode(g))
    fg = g != 0
    ev = {"recon_fg_acc": float(((rec == g) & fg).sum() / fg.sum()), "recon_px_acc": float((rec == g).mean())}
    print(ev)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "patches.npy"), D)
    json.dump({"K": len(D), "P": a.P, "K1": NS + 1, "slots": NS, "kind": "slot", "fit_log": hist, **ev},
              open(os.path.join(a.out, "patchdict.json"), "w"), indent=1)
