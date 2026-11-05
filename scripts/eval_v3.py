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
        # "dir@g=2:t=0.7": VQ sampling overrides (g = classifier-free guidance weight, t = prior temperature)
        base, _, opts = name.partition("@")
        d = os.path.join(ROOT, "artifacts", "model", base)
        self.name, self.dir = name, d
        self.fname = name.replace("@", "_").replace("=", "").replace(":", "_")
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        self.unet_dir = None
        for kv in filter(None, opts.split(":")):
            k, v = kv.split("=")
            if k == "u":   # v3-P: unconditional prior dir (for guidance)
                self.unet_dir = os.path.join(ROOT, "artifacts", "model", v)
                continue
            self.cfg[{"g": "guidance", "t": "sample_tau", "c": "noise_clip", "m": "trunc"}[k]] = float(v)
        self.v3p = self.cfg.get("model") in ("v3p", "v3pg")
        self.grouped = self.cfg.get("model") == "v3pg"
        self.v3 = self.cfg.get("model") in ("v3", "v3vq", "v3p", "v3pg")
        self.vq = self.cfg.get("model") == "v3vq"
        self.mdm = self.cfg.get("model") == "mdm"
        vp = os.path.join(d, "vocab.txt")
        self.vocab = T.load_vocab(vp if os.path.exists(vp) else os.path.join(E.BUILD, "vocab.txt"))
        pp = os.path.join(d, "palette.json")
        pal = json.load(open(pp if os.path.exists(pp) else os.path.join(E.BUILD, "palette.json")))["colors"]
        self.rgb = np.concatenate([BG[None], np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])
        if self.mdm:
            from compare_models import load_v2
            self.net, self.c2 = load_v2(d)
        elif self.v3p:
            from spritesheet.patchgpt import PatchGPT, UPatchGPT, GroupGPT
            wv3 = np.load(os.path.join(ROOT, "data", "build", "v3", "wordvec.npy"))
            self.net = GroupGPT(self.cfg, wv3) if self.grouped else PatchGPT(self.cfg, wv3)
            self.net.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
            self.net = self.net.eval()
            self.D = np.load(os.path.join(d, "patches.npy"))
            self.unet = None
            if self.unet_dir:
                uc = json.load(open(os.path.join(self.unet_dir, "config.json")))
                self.unet = GroupGPT(uc) if "G" in uc else UPatchGPT(uc)
                self.unet.load_state_dict(torch.load(os.path.join(self.unet_dir, "model.pt"), map_location="cpu"))
                self.unet = self.unet.eval()
        elif self.vq:
            from spritesheet.vq import VQNet
            self.net = VQNet(self.cfg, np.load(os.path.join(ROOT, "data", "build", "v3", "wordvec.npy")))
            self.net.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
            self.net = self.net.double().eval()
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
        if self.v3p:
            from spritesheet.patchgpt import sample_guided, sample_grouped
            from spritesheet.patches import from_patches
            from spritesheet import reference_p as RP
            Tn, Kp = self.cfg["T"], self.cfg["Kp"]
            g, tau = self.cfg.get("guidance", 0.0), self.cfg.get("sample_tau", 1.0)
            out = []
            for i in range(0, len(caps), 128):
                nz = torch.tensor(np.stack([RP.gumbel(s, Tn, Kp) for s in seeds[i:i + 128]]), dtype=torch.float32)
                if "noise_clip" in self.cfg:   # truncated Gumbel: MIN(c, g) caps how far a low-logit code can jump
                    nz = nz.clamp(max=self.cfg["noise_clip"])
                fn = sample_grouped if self.grouped else sample_guided
                codes = fn(self.net, self.unet, ids[i:i + 128], nz, g if self.unet is not None else 0.0, tau,
                           self.cfg.get("trunc"))
                out.append(from_patches(self.D[codes.numpy()], self.cfg["P"]).astype(np.uint8))
            return np.concatenate(out), None
        if self.vq:
            from spritesheet.vq import seed_uniforms
            Tn = self.cfg["G"] ** 2
            out_g, out_c = [], []
            for i in range(0, len(caps), 128):
                u = torch.tensor([seed_uniforms(s, Tn) for s in seeds[i:i + 128]], dtype=torch.float64)
                lg, _ = self.net.generate(ids[i:i + 128], u, self.cfg.get("sample_tau", 1.0), self.cfg.get("guidance", 0.0))
                pr = torch.softmax(lg, 1)
                out_g.append(pr.argmax(1).numpy().astype(np.uint8))
                out_c.append(pr.max(1).values.mean((1, 2)).numpy())
            return np.concatenate(out_g), np.concatenate(out_c)
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


def boot_ci(img_emb, txt_emb, n=1000, seed=0):
    """95% bootstrap CI (over prompts) for R@5 and CLIP score."""
    sims = (img_emb @ txt_emb.T).numpy()
    rank = (sims > np.diag(sims)[:, None]).sum(1)
    r5, cs = (rank < 5).astype(float), np.diag(sims)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cs), (n, len(cs)))
    q = lambda v: [float(np.quantile(v[idx].mean(1), 0.025)), float(np.quantile(v[idx].mean(1), 0.975))]
    return {"R@5_ci95": q(r5), "clip_score_ci95": q(cs)}


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
    ap.add_argument("--n-test", type=int, default=1000)
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
            ie = clip.img(m.rgbs(g))
            r[k] = r_precision(ie, txt[k])
            r[k].update(boot_ci(ie, txt[k]))
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
                                      {kk: (round(vv, 3) if isinstance(vv, float) else [round(x, 3) for x in vv]) for kk, vv in v.items() if kk != "chance_R@1"}) for k, v in r.items()}), flush=True)
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
        im.save(os.path.join(ROOT, "artifacts", "figures", f"v3_seeds_{a.tag}_{m.fname}.png"))
    # panel: 2 held-out test captions per PixelGPT family + 10 novel + 12 adjective prompts, 4 seeds each
    import collections
    z = np.load(os.path.join(ROOT, "data", "build", "v3", "imgs_24.npz"))
    te = np.nonzero(z["split"] == 2)[0]
    fam = collections.defaultdict(list)
    for i in te:
        fam[str(z["category"][i]).split("/")[0]].append(i)
    rr = np.random.default_rng(5)
    panel = []
    for f in sorted(fam):
        for i in rr.choice(fam[f], 2, replace=False):
            panel.append(("a " + str(z["caption"][i]), int(i)))
    panel += [(p, None) for p in NOVEL[:10]] + [(p, None) for p in GROUPS["adjectives"]]
    seeds = [1, 2, 3, 4]
    for m in models:
        cs, pad, per_row = 64, 3, 4
        cols = 4  # prompts per row, each with 4 seeds
        nrow = (len(panel) + cols - 1) // cols
        W = cols * (4 * (cs + pad) + 24)
        im = Image.new("RGB", (W + 10, 40 + nrow * (cs + 24)), PAPER)
        d = ImageDraw.Draw(im)
        d.text((10, 8), f"{m.name} · {len(panel)} prompts × seeds 1-4 (2 held-out captions per PixelGPT family, novel, adjectives)",
               fill=INK, font=font(16))
        for k, (p, gi) in enumerate(panel):
            g, _ = m.generate([p] * 4, seeds)
            r_, c_ = divmod(k, cols)
            x0, y0 = 10 + c_ * (4 * (cs + pad) + 24), 34 + r_ * (cs + 24)
            for j in range(4):
                im.paste(Image.fromarray(m.rgb[g[j]]).resize((cs, cs), Image.NEAREST), (x0 + j * (cs + pad), y0))
            d.text((x0, y0 + cs + 2), p[:44], fill=MUTED, font=font(11))
        im.save(os.path.join(ROOT, "artifacts", "figures", f"v3_panel_{a.tag}_{m.fname}.png"))
    # ground truth vs models on 24 random held-out test captions (seed 1)
    pick = rr.choice(te, 24, replace=False)
    gcaps = [str(z["caption"][i]) for i in pick]
    cs, pad, left = 72, 4, 150
    im = Image.new("RGB", (left + 24 * (cs + pad) + 10, 60 + (len(models) + 1) * (cs + pad) + 10), PAPER)
    d = ImageDraw.Draw(im)
    d.text((10, 8), "held-out test captions: real sprite (top) vs each model (seed 1)", fill=INK, font=font(18))
    for j, cap in enumerate(gcaps):
        d.text((left + j * (cs + pad), 34), cap[:11], fill=MUTED, font=font(9))
        t = z["imgs"][pick[j]]
        im.paste(Image.fromarray(np.where(t[..., None] > 0, s3.pal.rgb[np.maximum(t, 1) - 1], BG)).resize((cs, cs), Image.NEAREST),
                 (left + j * (cs + pad), 50))
    d.text((10, 50 + cs // 2), "ground truth", fill=INK, font=font(14))
    for i, m in enumerate(models):
        g, _ = m.generate(gcaps, [1] * len(gcaps))
        y = 50 + (i + 1) * (cs + pad)
        d.text((10, y + cs // 2 - 8), m.name, fill=INK, font=font(14))
        for j in range(len(gcaps)):
            im.paste(Image.fromarray(m.rgb[g[j]]).resize((cs, cs), Image.NEAREST), (left + j * (cs + pad), y))
    im.save(os.path.join(ROOT, "artifacts", "figures", f"v3_groundtruth_{a.tag}.png"))
    print("wrote", out_p)


if __name__ == "__main__":
    main()
