"""Why d=64 for the CLIP word table: rank of the true synonym among a word's neighbours after PCA to d.
Lower rank = synonym stays closer. → artifacts/metrics/wordtable_dim.json"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet import vocab as V  # noqa: E402
PAIRS = [("puppy", "dog"), ("kitten", "cat"), ("automobile", "car"), ("spaceship", "rocket"), ("crimson", "red"),
         ("azure", "blue"), ("tiny", "small"), ("huge", "big")]
w = V.clip_common_words(4000)
allw = w + [x for p in PAIRS for x in p if x not in w]
E = V.clip_text_embed(allw)
idx = {x: i for i, x in enumerate(allw)}
res = {"vocab": len(allw), "pairs": [f"{a}->{b}" for a, b in PAIRS], "dims": {}}
for d in (32, 48, 64, 512):
    Z = E if d == 512 else V.pca(E, d)[0]
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    ranks = []
    for a, b in PAIRS:
        s = Zn @ Zn[idx[a]]
        ranks.append(int((s > s[idx[b]]).sum()))
    res["dims"][str(d)] = {"ranks": ranks, "median_rank": float(np.median(ranks))}
json.dump(res, open(os.path.join(ROOT, "artifacts", "metrics", "wordtable_dim.json"), "w"), indent=1)
print(json.dumps(res["dims"]))
