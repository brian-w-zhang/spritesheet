"""Sampler hypotheses on a trained v2 model (no retraining): commit rule × temperature × steps.
Writes a grid per variant + metrics (given-style fg acc, blank-rate) → artifacts/metrics/v2_sampler_sweep_<tag>.json

uv run python scripts/v2_sampler_sweep.py --v2 artifacts/model/v2_mdm_a --tag a
"""
import argparse, json, math, os, sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from spritesheet.mdm import seeded_uniform, seed_state  # noqa: E402
from compare_models import PROMPTS, load_v2  # noqa: E402
from make_figures import font, INK, PAPER  # noqa: E402


@torch.no_grad()
def sample(net, ids, sty, seeds, T_, rule, temp, frac_sched="cos"):
    """Deterministic given seeds. rule: 'conf' (top confidence), 'fgfirst' (background only at the last step).
    Commits a FIXED fraction per step (cosine schedule) by confidence rank; the in-sheet version would use
    calibrated thresholds; this sweep isolates the commit rule itself."""
    B, N, K = ids.shape[0], net.cfg["N"], net.cfg["K"]
    M = K + 1
    _, cond = net.cond(ids, sty)
    x = torch.full((B, N, N), M, dtype=torch.long)
    for t in range(1, T_ + 1):
        lg = net.denoise(x, cond)
        p = torch.softmax(lg, -1)
        if temp > 0:
            g = torch.stack([torch.from_numpy(-np.log(-np.log(seeded_uniform(seed_state(s), t, N, K + 1)))) for s in seeds]).to(lg.dtype)
            am = (lg + temp * g).argmax(-1)
        else:
            am = lg.argmax(-1)
        conf = torch.gather(p, -1, am[..., None])[..., 0]
        masked = x == M
        keep = math.cos(math.pi / 2 * t / T_) if t < T_ else 0.0
        score = torch.where(masked, conf, torch.full_like(conf, -1.0))
        if rule == "fgfirst" and t < T_:
            score = torch.where(am == 0, torch.full_like(conf, -0.5), score)  # background waits for the last step
        for b in range(B):
            n_commit = int(masked[b].sum().item() - round(keep * N * N))
            if n_commit <= 0:
                continue
            idx = torch.topk(score[b].flatten(), n_commit).indices
            sel = torch.zeros(N * N, dtype=torch.bool)
            sel[idx] = True
            sel = sel.view(N, N) & masked[b] & (score[b] > -0.75)
            x[b] = torch.where(sel, am[b], x[b])
    x = torch.where(x == M, torch.zeros_like(x), x)
    return x.numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    from spritesheet.data.synth import eval_set
    net, cfg = load_v2(a.v2)
    vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt"))
    rgb = E.palette_rgb()
    data = eval_set("heldout_combo", a.n, tier=1) + eval_set("train_sample", a.n, tier=1)
    caps, tg, sid = [d[0] for d in data], np.stack([d[1] for d in data]), [d[2] for d in data]
    variants = [("conf", 0.0, 6), ("fgfirst", 0.0, 6), ("conf", 1.0, 6), ("fgfirst", 1.0, 6), ("fgfirst", 0.5, 6),
                ("fgfirst", 1.0, 4), ("fgfirst", 1.0, 8), ("fgfirst", 1.0, 12)]
    res = {}
    for rule, temp, T_ in variants:
        key = f"{rule}_t{temp}_T{T_}"
        out = []
        for i in range(0, len(caps), 64):
            ids = torch.tensor([T.encode(c, vocab) for c in caps[i:i + 64]])
            out.append(sample(net, ids, torch.tensor(sid[i:i + 64]), list(range(1 + i, 1 + i + len(ids))), T_, rule, temp))
        o = np.concatenate(out)
        fg = tg != 0
        res[key] = {"fg_pixel_acc_given_style": float(((o == tg) & fg).sum() / fg.sum()),
                    "pixel_acc": float((o == tg).mean()),
                    "blank_rate": float(((o != 0).sum((1, 2)) < 20).mean()),
                    "fg_area_ratio": float((o != 0).sum() / max(1, fg.sum()))}
        print(key, json.dumps({k: round(v, 3) for k, v in res[key].items()}), flush=True)
        # grid of the 24 demo prompts
        ids = torch.tensor([T.encode(p, vocab) for p in PROMPTS])
        st = torch.zeros(len(PROMPTS), dtype=torch.long)  # noto for comparability
        g = sample(net, ids, st, [1] * len(PROMPTS), T_, rule, temp)
        cs = 100
        im = Image.new("RGB", (12 * (cs + 8) + 16, 2 * (cs + 28) + 60), PAPER)
        d = ImageDraw.Draw(im)
        d.text((16, 14), key, fill=INK, font=font(26))
        for j in range(len(PROMPTS)):
            r, c = divmod(j, 12)
            im.paste(Image.fromarray(E.to_rgb(g[j], rgb)).resize((cs, cs), Image.NEAREST), (16 + c * (cs + 8), 54 + r * (cs + 28)))
            d.text((16 + c * (cs + 8), 54 + r * (cs + 28) + cs + 4), PROMPTS[j][:16], fill=INK, font=font(13))
        im.save(os.path.join(ROOT, "artifacts", "figures", f"v2sweep_{a.tag}_{key}.png"))
    json.dump(res, open(os.path.join(ROOT, "artifacts", "metrics", f"v2_sampler_sweep_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
