"""Patch dictionary for v3-P: every 24x24 sprite becomes a (24/P)x(24/P) grid of indices into a dictionary of
K hard PxP pixel patches. Decoding is pure lookup (crisp by construction; in the sheet it is INDEX formulas).

The dictionary is fit by weighted k-modes on (augmented) training patches: distance = number of mismatched pixels,
update = per-pixel majority colour. No neural network.

uv run python -m spritesheet.patches fit --K 1024 --P 4 --out artifacts/model/patchdict_k1024_p4
"""
import argparse, json, os, time

import numpy as np
import torch

from .data.v3 import OUT as V3, Sampler3, eval_set3


def to_patches(grids, P):
    """[n, N, N] → [n, G*G, P*P] (row-major over the grid, row-major inside each patch)."""
    n, N, _ = grids.shape
    G = N // P
    return grids.reshape(n, G, P, G, P).transpose(0, 1, 3, 2, 4).reshape(n, G * G, P * P)


def from_patches(pt, P):
    n, T, _ = pt.shape
    G = int(round(T ** 0.5))
    return pt.reshape(n, G, G, P, P).transpose(0, 1, 3, 2, 4).reshape(n, G * P, G * P)


def _onehot(x, K1):
    return torch.nn.functional.one_hot(torch.as_tensor(x, dtype=torch.long), K1).flatten(1).float()


def assign(patches, D, K1, chunk=65536, weight_fg=1.0):
    """patches [M, P*P] uint8, D [K, P*P] → nearest entry (min mismatches) [M]."""
    Dh = _onehot(D, K1)                                     # [K, P*P*K1]
    out = np.empty(len(patches), np.int64)
    for i in range(0, len(patches), chunk):
        x = _onehot(patches[i:i + chunk], K1)
        out[i:i + chunk] = (x @ Dh.T).argmax(1).numpy()  # max matches = min mismatches
    return out


def fit(K, P, n_sprites=120_000, iters=12, seed=0, K1=None, log=print):
    torch.manual_seed(seed)
    s = Sampler3("train", seed=seed)
    K1 = K1 or (int(s.pal.n) + 1 if hasattr(s.pal, "n") else int(s.imgs.max()) + 1)
    t0 = time.time()
    _, g = s.batch(n_sprites)
    pt = to_patches(g, P).reshape(-1, P * P)
    U, cnt = np.unique(pt, axis=0, return_counts=True)
    log(f"{len(pt):,} patches, {len(U):,} unique ({time.time() - t0:.0f}s)")
    D = U[np.argsort(-cnt)[:K]].copy()                      # init: the K most frequent patches
    w = cnt.astype(np.float64)
    hist = []
    for it in range(iters):
        a = assign(U, D, K1)
        # weighted per-pixel majority vote
        idx = (a[:, None] * (P * P) + np.arange(P * P)[None]) * K1 + U
        votes = np.bincount(idx.ravel(), weights=np.repeat(w, P * P), minlength=K * P * P * K1).reshape(K, P * P, K1)
        used = votes.sum((1, 2)) > 0
        D[used] = votes[used].argmax(-1)
        # re-seed dead entries with the worst-fit frequent patches
        mism = (U != D[a]).sum(1)
        if (~used).any():
            worst = np.argsort(-(mism * w))[: int((~used).sum())]
            D[~used] = U[worst]
        err = float((mism * w).sum() / (w.sum() * P * P))
        hist.append({"iter": it, "pixel_err": round(err, 5), "dead": int((~used).sum()), "s": round(time.time() - t0, 1)})
        log(json.dumps(hist[-1]))
    return D.astype(np.uint8), hist, K1


def recon_eval(D, P, K1, n=2048):
    _, g = eval_set3("validation", n)
    pt = to_patches(g, P)
    a = assign(pt.reshape(-1, P * P), D, K1).reshape(len(g), -1)
    rec = from_patches(D[a], P)
    fg = g != 0
    return {"recon_fg_acc": float(((rec == g) & fg).sum() / fg.sum()), "recon_px_acc": float((rec == g).mean()),
            "codes_used_val": int(len(np.unique(a)))}, g, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--n-sprites", type=int, default=120_000)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    pal = json.load(open(os.path.join(V3, "palette.json")))["colors"]
    K1 = len(pal) + 1
    D, hist, _ = fit(a.K, a.P, a.n_sprites, a.iters, K1=K1)
    ev, g, rec = recon_eval(D, a.P, K1)
    print(json.dumps(ev))
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "patches.npy"), D)
    json.dump({"K": a.K, "P": a.P, "K1": K1, "n_sprites": a.n_sprites, "iters": a.iters, "fit_log": hist, **ev},
              open(os.path.join(a.out, "patchdict.json"), "w"), indent=1)
    # local-only visual check (shows real PixelGPT sprites: never commit; *_local.png is gitignored)
    from PIL import Image
    rgb = np.concatenate([np.array([[216, 212, 221]], np.uint8), np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])
    rows = [np.concatenate([np.pad(rgb[x], ((1, 1), (1, 1), (0, 0)), constant_values=255) for x in r[:16]], 1) for r in (g, rec)]
    Image.fromarray(np.concatenate(rows, 0)).resize((16 * 26 * 4, 2 * 26 * 4), Image.NEAREST).save(os.path.join(a.out, "recon_local.png"))


if __name__ == "__main__":
    main()
