"""Can a fixed codebook of P×P patches reproduce our sprites? (decides patch-token vs pixel diffusion)
Builds codebooks of the most frequent patches (+ nearest-by-Hamming assignment) from tier-1 emoji bases,
reports reconstruction pixel accuracy on held-out bases. → artifacts/metrics/codebook_check.json"""
import collections, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
b = np.load(os.path.join(ROOT, "data", "build", "tier1", "bases_32.npz"))
imgs = b["imgs"]
style = b["style"]
styles = json.load(open(os.path.join(ROOT, "data", "build", "styles.json")))
emoji = np.array([styles[s] != "pixelgpt" for s in style])
X = imgs[emoji]
rng = np.random.default_rng(0)
idx = rng.permutation(len(X)); tr, te = X[idx[:-500]], X[idx[-500:]]
res = {}
for P in (2, 4):
    def patches(A):
        n = A.shape[1] // P
        return A.reshape(len(A), n, P, n, P).transpose(0, 1, 3, 2, 4).reshape(len(A), n * n, P * P)
    pt = patches(tr).reshape(-1, P * P)
    cnt = collections.Counter(map(bytes, pt))
    for K in (256, 512, 1024, 2048, 4096):
        if K > len(cnt):
            continue
        book = np.array([np.frombuffer(k, dtype=np.uint8) for k, _ in cnt.most_common(K)])
        pe = patches(te).reshape(-1, P * P)
        best = np.zeros(len(pe), dtype=int)
        bd = np.full(len(pe), 99)
        for i in range(0, K, 256):
            d = (pe[:, None, :] != book[None, i:i + 256, :]).sum(-1)
            j = d.argmin(1); m = d[np.arange(len(pe)), j]
            upd = m < bd; bd[upd] = m[upd]; best[upd] = j[upd] + i
        rec = book[best]
        acc = float((rec == pe).mean())
        fg = pe != 0
        res[f"P{P}_K{K}"] = {"patch": P, "codebook": K, "recon_pixel_acc": acc,
                             "recon_fg_pixel_acc": float(((rec == pe) & fg).sum() / fg.sum()),
                             "exact_patch_rate": float((bd == 0).mean()), "unique_train_patches": len(cnt)}
        print(f"P={P} K={K}: pixel {acc:.4f} fg {res[f'P{P}_K{K}']['recon_fg_pixel_acc']:.4f} exact-patch {(bd==0).mean():.3f}", flush=True)
json.dump(res, open(os.path.join(ROOT, "artifacts", "metrics", "codebook_check.json"), "w"), indent=1)
