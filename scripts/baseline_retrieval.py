"""C1: retrieval baseline. For each eval prompt, return the target of the most CLIP-similar TRAINING caption.
A lookup can't produce held-out combinations; this measures how much the decoder actually composes.

uv run python scripts/baseline_retrieval.py --tier 1 --bank 20000 --n 300
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet import evaluate as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--bank", type=int, default=20000)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sets", default=",".join(E.SETS))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "metrics", "baseline_retrieval.json"))
    a = ap.parse_args()
    from spritesheet.data.synth import Sampler, eval_set
    s = Sampler(split="train", seed=123, tier=a.tier)
    caps, tg = [], []
    while len(caps) < a.bank:
        c, t, _, _ = s.batch(1000)
        caps += list(c)
        tg += list(np.asarray(t))
    clip = E.Clip()
    bank = clip.txt(caps)
    concepts = json.load(open(os.path.join(E.BUILD, "concepts.json")))
    names = [c["id"] for c in concepts]
    ctxt = clip.txt([f"an emoji of {c['name']}" for c in concepts])
    rgb = E.palette_rgb()
    res = {"bank": len(caps), "sets": {}}
    for name in a.sets.split(","):
        try:
            data = eval_set(name, a.n, tier=a.tier)
        except Exception as e:
            res["sets"][name] = {"error": str(e)}
            continue
        if not data:
            continue
        q = clip.txt([d[0] for d in data])
        idx = (q @ bank.T).argmax(1).numpy()
        outs = [tg[i] for i in idx]
        res["sets"][name] = E.evaluate_grids(name, [d[0] for d in data], outs, [d[1] for d in data],
                                             [d[3] for d in data], clip, names, ctxt, rgb)
        print(name, json.dumps({k: v for k, v in res["sets"][name].items() if not isinstance(v, dict)}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
