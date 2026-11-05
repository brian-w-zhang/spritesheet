"""Flatness check: mean distinct colours and foreground share per sprite, real PixelGPT test sprites vs model samples
(400 test captions, seeds 1-400). Low colour counts = flat single-colour silhouettes.

uv run python scripts/colour_stats.py --models 'v3p_a@t=0.8,v3p_a@t=1' --tag r10
→ artifacts/metrics/colour_stats_<tag>.json
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_v3 import Model  # noqa: E402


def stats(g):
    return {"colours": round(float(np.mean([len(np.unique(x[x > 0])) for x in g])), 3),
            "fg_share": round(float(np.mean([(x > 0).mean() for x in g])), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, "data", "build", "v3", "imgs_24.npz"))
    te = z["imgs"][z["split"] == 2][: a.n]
    caps = [str(c) for c in z["caption"][z["split"] == 2][: a.n]]
    out = {"real_test": stats(te), "n": a.n}
    print("real", out["real_test"])
    for name in a.models.split(","):
        g, _ = Model(name).generate(caps, list(range(1, a.n + 1)))
        out[name] = stats(g)
        print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "metrics", f"colour_stats_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
