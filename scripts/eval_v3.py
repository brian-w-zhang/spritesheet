"""Evaluate v3 (24x24, variants) models — and v1/v2 models for comparison — on the same prompts.

Metrics (artifacts/metrics/v3_eval_<tag>.json):
  * CLIP R@1/R@5/score on 100 NOVEL prompts and 300 PixelGPT TEST captions (verbatim, held out);
    ground-truth PixelGPT test sprites give the ceiling.
  * colour adjectives: "<colour> <test caption>" → dominant hue bucket matches (12 colours × 20 captions).
  * sharpness / anti-averaging: mean per-pixel max-probability (averaging shows as low confidence),
    speckle (share of pixels a 3x3 majority filter would change), fg connected components.
  * seed diversity: mean pairwise pixel difference across 8 seeds of the same prompt.
Figures: artifacts/figures/v3_lineup_<tag>_<group>.png (models × prompts), v3_seeds_<tag>_<model>.png (8 seeds).

uv run python scripts/eval_v3.py --models v3_var16,v3_base,v32 --tag r1
"""
import argparse, json, os, sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from spritesheet.reference import lcg_uniform  # noqa: E402
from eval_generalization import NOVEL, r_precision  # noqa: E402
from make_figures import font, INK, MUTED, PAPER  # noqa: E402

BG = np.array([0xD8, 0xD4, 0xDD], np.uint8)
GROUPS = {
    "creatures": ["a frog", "a fox", "a cat", "a dragon", "a ghost", "a slime", "an owl", "a skeleton warrior", "a wizard", "a knight"],
    "objects": ["a sword", "a treasure chest", "a red potion", "a golden key", "a mushroom", "a tree", "a castle", "a crown",
                "a wooden shield", "a lantern"],
    "adjectives": ["a blue frog", "a red frog", "a green sword", "a purple mushroom", "a golden crown", "a pink cat",
                   "a black cat", "a white ghost", "an orange fox", "a blue potion"],
}
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "golden"]
# expected hue bucket(s) (30-degree buckets, 99 = achromatic) per colour word
HUE_OK = {"red": {0, 11}, "orange": {0, 1}, "yellow": {1, 2}, "green": {3, 4, 5}, "blue": {6, 7, 8}, "purple": {8, 9, 10},
          "pink": {10, 11, 0}, "brown": {0, 1}, "black": {99}, "white": {99}, "gray": {99}, "golden": {1, 2}}


class Model:
    def __init__(self, name):
        d = os.path.join(ROOT, "artifacts", "model", name)
        self.name, self.dir = name, d
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.v3 = self.cfg.get("model") == "v3"
        self.mdm = self.cfg.get("model") == "mdm"
        vp = os.path.join(d, "vocab.txt")
        self.vocab = T.load_vocab(vp if os.path.exists(vp) else os.path.join(E.BUILD, "vocab.txt"))
        pp = os.path.join(d, "palette.json")
        pal = json.load(open(pp if os.path.exists(pp) else os.path.join(E.BUILD, "palette.json")))["colors"]
        self.rgb = np.concatenate([BG[None], np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])
        if self.mdm:
            from compare_models import load_v2
            self.net, self.c2 = load_v2(d)
        else:
            from spritesheet.model import SpriteNet
            wv = np.load(os.path.join(d, "weights.npz"))["wordvec"] if self.v3 else np.load(os.path.join(E.BUILD, "wordvec.npy"))
            self.net = SpriteNet(self.cfg, wv)
            self.net.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
            self.net = self.net.double().eval()

    @torch.no_grad()
    def generate(self, caps, seeds):
        """→ class grids [n,N,N], per-pixel max prob [n] (None for v2), picked variant/style ids."""
        if self.mdm:
            from compare_models import gen_v2
            g, _, _ = gen_v2(self.net, self.c2, caps, self.vocab, seeds)
            return np.asarray(g), None
        ids = torch.tensor([T.encode(c, self.vocab) for c in caps])
        out_g, out_c = [], []
        for i in range(0, len(caps), 128):
            c = self.net.encode(ids[i:i + 128])
            p = torch.softmax(self.net.style_head(c), -1).numpy()
            st = []
            for j, s in enumerate(seeds[i:i + 128]):
                u, cum, below = lcg_uniform(s), 0.0, 0
                for pi in p[j]:
                    cum += float(pi)
                    below += cum < u
                st.append(min(below, p.shape[1] - 1))
            pr = torch.softmax(self.net.decode(c, torch.tensor(st)), 1)
            out_g.append(pr.argmax(1).numpy().astype(np.uint8))
            out_c.append(pr.max(1).values.mean((1, 2)).numpy())
        return np.concatenate(out_g), np.concatenate(out_c)

    def rgbs(self, grids):
        return [self.rgb[g] for g in grids]


def mode3_changes(g):
    N = g.shape[0]
    ch = 0
    for i in range(N):
        for j in range(N):
            win = g[max(0, i - 1):i + 2, max(0, j - 1):j + 2].ravel()
            cnt = np.bincount(win)
            ch += cnt[g[i, j]] < cnt.max()
    return ch / (N * N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--n-test", type=int, default=300)
    a = ap.parse_args()
    from spritesheet.data.v3 import eval_set3, Sampler3
    clip = E.Clip()
    tcaps, ttgt = eval_set3("test", a.n_test, seed=11)
    sets = {"novel": NOVEL, "pixelgpt_test": tcaps}
    txt = {k: clip.txt(v) for k, v in sets.items()}
    out_p = os.path.join(ROOT, "artifacts", "metrics", f"v3_eval_{a.tag}.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    s3 = Sampler3("test")
    gt_rgb = [np.where(t[..., None] > 0, s3.pal.rgb[np.maximum(t, 1) - 1], BG) for t in ttgt]
    res["_ground_truth_pixelgpt_test"] = r_precision(clip.img(gt_rgb), txt["pixelgpt_test"])
    models = [Model(n) for n in a.models.split(",")]
    adj_caps, adj_col = [], []
    for c in COLOURS:
        for cap in tcaps[:20]:
            adj_caps.append(f"{c} {cap}")
            adj_col.append(c)
    for m in models:
        r = {"params": m.cfg.get("params"), "N": m.cfg.get("N", 32), "steps": m.cfg.get("steps")}
        for k, caps in sets.items():
            g, conf = m.generate(caps, [1 + i for i in range(len(caps))])
            r[k] = r_precision(clip.img(m.rgbs(g)), txt[k])
            if conf is not None:
                r[k]["mean_pixel_confidence"] = float(conf.mean())
            r[k]["speckle"] = float(np.mean([mode3_changes(x) for x in g]))
            r[k]["fg_components"] = float(np.mean([E.n_components(E.fg_mask(x)[0]) for x in g]))
        g, _ = m.generate(adj_caps, [1] * len(adj_caps))
        hb = [E.hue_bucket(x, m.rgb) for x in g]
        r["colour_adjective_acc"] = float(np.mean([h in HUE_OK[c] for h, c in zip(hb, adj_col)]))
        div = []
        for p in GROUPS["creatures"] + GROUPS["objects"]:
            gg, _ = m.generate([p] * 8, list(range(1, 9)))
            div.append(np.mean([(gg[i] != gg[j]).mean() for i in range(8) for j in range(i + 1, 8)]))
        r["seed_diversity"] = float(np.mean(div))
        res[m.name] = r
        print(m.name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v if not isinstance(v, dict) else
                                      {kk: round(vv, 3) for kk, vv in v.items() if kk != "chance_R@1"}) for k, v in r.items()}), flush=True)
    json.dump(res, open(out_p, "w"), indent=1)
    # line-ups: rows = models, seed 1
    for gname, prompts in GROUPS.items():
        cs, pad, left = 96, 6, 170
        im = Image.new("RGB", (left + len(prompts) * (cs + pad) + 10, 70 + len(models) * (cs + pad) + 10), PAPER)
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"{gname} · seed 1", fill=INK, font=font(24))
        for j, p in enumerate(prompts):
            d.text((left + j * (cs + pad), 42), p[:16], fill=MUTED, font=font(12))
        for i, m in enumerate(models):
            g, _ = m.generate(prompts, [1] * len(prompts))
            y = 62 + i * (cs + pad)
            d.text((10, y + cs // 2 - 8), m.name, fill=INK, font=font(16))
            for j, x in enumerate(g):
                im.paste(Image.fromarray(m.rgb[x]).resize((cs, cs), Image.NEAREST), (left + j * (cs + pad), y))
        im.save(os.path.join(ROOT, "artifacts", "figures", f"v3_lineup_{a.tag}_{gname}.png"))
    # seeds: per model, 8 seeds × 10 prompts
    for m in models:
        prompts = GROUPS["creatures"]
        cs, pad, left = 80, 4, 150
        im = Image.new("RGB", (left + 8 * (cs + pad) + 10, 60 + len(prompts) * (cs + pad)), PAPER)
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"{m.name} · seeds 1-8 (columns)", fill=INK, font=font(22))
        for i, p in enumerate(prompts):
            g, _ = m.generate([p] * 8, list(range(1, 9)))
            y = 50 + i * (cs + pad)
            d.text((10, y + cs // 2 - 8), p, fill=INK, font=font(14))
            for j, x in enumerate(g):
                im.paste(Image.fromarray(m.rgb[x]).resize((cs, cs), Image.NEAREST), (left + j * (cs + pad), y))
        im.save(os.path.join(ROOT, "artifacts", "figures", f"v3_seeds_{a.tag}_{m.name}.png"))
    print("wrote", out_p)


if __name__ == "__main__":
    main()
