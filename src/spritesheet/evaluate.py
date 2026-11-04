"""Quality metrics (doc 06 §B) and baselines (C1). All automatic.

uv run python -m spritesheet.evaluate --model artifacts/model/v32 --out artifacts/metrics/quality.json [--n 300]
"""
import argparse
import json
import os

import numpy as np
import torch

from . import tokenizer as T
from .model import SpriteNet

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "data", "build")
CLIP_DIR = os.path.join(ROOT, "data", "models", "clip-vit-base-patch32")
SETS = ["train_sample", "heldout_combo", "heldout_phrase", "unseen_words", "underspecified", "pixelgpt_test"]


# ---------------------------------------------------------------- model loading / inference
def load_net(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    wv = np.load(os.path.join(BUILD, "wordvec.npy"))
    net = SpriteNet(cfg, wv, bow=cfg.get("bow", False), learn_wordvec=cfg.get("learn_wordvec", False))
    net.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    return net.double().eval(), cfg


@torch.no_grad()
def generate(net, captions, vocab, seeds, style_override=None):
    """Seeded style sampling exactly as in reference.pick_style, then decode. Returns class grids, style ids (0-based)."""
    from .reference import lcg_uniform
    ids = torch.tensor([T.encode(c, vocab) for c in captions])
    c = net.encode(ids)
    p = torch.softmax(net.style_head(c), -1).numpy()
    st = []
    for i, s in enumerate(seeds):
        if style_override is not None:
            st.append(int(style_override[i]))
            continue
        u, cum, below = lcg_uniform(s), 0.0, 0
        for pi in p[i]:
            cum += float(pi)
            below += cum < u
        st.append(min(below, p.shape[1] - 1))
    logits = net.decode(c, torch.tensor(st))
    return logits.argmax(1).numpy().astype(np.uint8), np.array(st)


# ---------------------------------------------------------------- detectors (applied to output AND target)
def palette_rgb():
    pal = json.load(open(os.path.join(BUILD, "palette.json")))["colors"]
    rgb = np.array([[0xF6, 0xF4, 0xEF]] + [[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], dtype=np.uint8)
    return rgb


def to_rgb(grid, rgb=None):
    rgb = palette_rgb() if rgb is None else rgb
    return rgb[grid]


def fg_mask(g):
    corner = np.bincount(np.array([g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]])).argmax()
    return g != corner, int(corner)


def hue_bucket(g, rgb):
    m, _ = fg_mask(g)
    if m.sum() == 0:
        return -1
    px = rgb[g[m]].astype(float) / 255
    mx, mn = px.max(1), px.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    px = px[sat > 0.25]
    if len(px) < 3:
        return 99  # achromatic
    r, gg, b = px.T
    h = np.degrees(np.arctan2(np.sqrt(3) * (gg - b), 2 * r - gg - b)) % 360
    return int(np.bincount((h // 30).astype(int), minlength=12).argmax())


def n_components(m):
    seen = np.zeros_like(m, bool)
    n = 0
    for y, x in zip(*np.nonzero(m)):
        if seen[y, x]:
            continue
        n += 1
        stack = [(y, x)]
        seen[y, x] = True
        while stack:
            a, b = stack.pop()
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c, d = a + da, b + db
                if 0 <= c < m.shape[0] and 0 <= d < m.shape[1] and m[c, d] and not seen[c, d]:
                    seen[c, d] = True
                    stack.append((c, d))
    return n


def attrs(g, rgb):
    m, bg = fg_mask(g)
    N = g.shape[0]
    area = int(m.sum())
    if area:
        ys, xs = np.nonzero(m)
        cell = (int(min(2, ys.mean() * 3 // N)), int(min(2, xs.mean() * 3 // N)))
    else:
        cell = (-1, -1)
    big = m & np.pad(m, 1)[2:, 1:-1]  # crude speck filter for counting
    return {"hue": hue_bucket(g, rgb), "size": 0 if area < 0.18 * N * N else 1, "pos": cell,
            "count": min(3, n_components(big)) if area else 0, "bg": bg}


# ---------------------------------------------------------------- CLIP scoring
class Clip:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained(CLIP_DIR).eval()
        self.p = CLIPProcessor.from_pretrained(CLIP_DIR)

    @torch.no_grad()
    def img(self, rgbs):
        from PIL import Image
        ims = [Image.fromarray(r).resize((224, 224), Image.NEAREST) for r in rgbs]
        out = []
        for i in range(0, len(ims), 64):
            e = self.m.get_image_features(**self.p(images=ims[i:i + 64], return_tensors="pt"))
            e = getattr(e, "pooler_output", e)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def txt(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            e = self.m.get_text_features(**self.p(text=texts[i:i + 256], return_tensors="pt", padding=True))
            e = getattr(e, "pooler_output", e)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)


# ---------------------------------------------------------------- main evaluation
def evaluate_grids(name, caps, outs, tgts, metas, clip, concept_names, concept_txt, rgb):
    r = {"n": len(caps)}
    o, t = np.stack(outs), np.stack(tgts)
    if t.size:
        r["pixel_acc"] = float((o == t).mean())
        fg = t != 0
        r["fg_pixel_acc"] = float(((o == t) & fg).sum() / max(1, fg.sum()))
        r["pixel_perfect"] = float((o == t).all(axis=(1, 2)).mean())
        keys = ["hue", "size", "pos", "count", "bg"]
        agree = {k: [] for k in keys}
        for a, b, m in zip(o, t, metas):
            A, B = attrs(a, rgb), attrs(b, rgb)
            for k in keys:
                if m.get({"hue": "colour", "size": "size", "pos": "position", "count": "count", "bg": "bg"}[k]) is not None or k == "hue":
                    agree[k].append(A[k] == B[k])
        r["attribute_agreement"] = {k: (float(np.mean(v)) if v else None) for k, v in agree.items()}
        r["attribute_n"] = {k: len(v) for k, v in agree.items()}
    oi = clip.img([to_rgb(g, rgb) for g in o])
    ti = clip.img([to_rgb(g, rgb) for g in t]) if t.size else None
    pt = clip.txt(caps)
    r["clip_score"] = float((oi * pt).sum(1).mean())
    if ti is not None:
        r["clip_score_target"] = float((ti * pt).sum(1).mean())
        r["clip_score_rel"] = r["clip_score"] / r["clip_score_target"]
    cids = [m.get("concept") for m in metas]
    if all(c in concept_names for c in cids) and cids:
        sims = oi @ concept_txt.T
        gold = torch.tensor([concept_names.index(c) for c in cids])
        top = sims.topk(5, dim=1).indices
        r["clip_concept_top1"] = float((top[:, 0] == gold).float().mean())
        r["clip_concept_top5"] = float((top == gold[:, None]).any(1).float().mean())
        if ti is not None:
            tt = (ti @ concept_txt.T).topk(5, dim=1).indices
            r["clip_concept_top1_target"] = float((tt[:, 0] == gold).float().mean())
            r["clip_concept_top5_target"] = float((tt == gold[:, None]).any(1).float().mean())
    return r


def main():
    from .data.synth import eval_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--retrieval", action="store_true", help="also run the C1 retrieval baseline")
    ap.add_argument("--tier", type=int, default=None)
    a = ap.parse_args()
    net, cfg = load_net(a.model)
    vocab = T.load_vocab(os.path.join(BUILD, "vocab.txt"))
    concepts = json.load(open(os.path.join(BUILD, "concepts.json")))
    names = [c["id"] for c in concepts]
    clip = Clip()
    concept_txt = clip.txt([f"an emoji of {c['name']}" for c in concepts])
    rgb = palette_rgb()
    res = {"model": a.model, "config": cfg, "sets": {}}
    for s in a.sets.split(","):
        try:
            data = eval_set(s, a.n) if a.tier is None else eval_set(s, a.n, tier=a.tier)
        except Exception as e:
            res["sets"][s] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if not data:
            continue
        caps = [d[0] for d in data]
        tgts = [d[1] for d in data]
        metas = [d[3] for d in data]
        seeds = [1 + i for i in range(len(caps))]
        outs, sts = generate(net, caps, vocab, seeds)
        res["sets"][s] = evaluate_grids(s, caps, list(outs), tgts, metas, clip, names, concept_txt, rgb)
        # decoder quality given the target's style (the seed-sampled style may legitimately differ from the target's)
        outs_g, _ = generate(net, caps, vocab, seeds, style_override=[d[2] for d in data])
        o, t = np.stack(outs_g), np.stack(tgts)
        fg = t != 0
        res["sets"][s]["given_style"] = {"pixel_acc": float((o == t).mean()),
                                         "fg_pixel_acc": float(((o == t) & fg).sum() / max(1, fg.sum())),
                                         "pixel_perfect": float((o == t).all(axis=(1, 2)).mean())}
        named = [i for i, m in enumerate(metas) if len(m.get("styles_ok") or []) == 1]
        ok = [i for i, m in enumerate(metas) if m.get("styles_ok")]
        if ok:
            res["sets"][s]["sampled_style_valid"] = float(np.mean([metas[i]["styles_ok"] and int(sts[i]) in
                                                                   [int(x) for x in metas[i]["styles_ok"]] for i in ok]))
        if named:
            res["sets"][s]["named_style_acc"] = float(np.mean([int(sts[i]) == int(data[i][2]) for i in named]))
        if s == "underspecified":  # seed diversity: same prompt, 8 seeds
            div = []
            for cap in caps[:40]:
                g, st = generate(net, [cap] * 8, vocab, list(range(1, 9)))
                pairs = [(g[i] != g[j]).mean() for i in range(8) for j in range(i + 1, 8)]
                div.append((float(np.mean(pairs)), len(set(st.tolist()))))
            res["sets"][s]["seed_pixel_diversity"] = float(np.mean([d[0] for d in div]))
            res["sets"][s]["distinct_styles_per_8_seeds"] = float(np.mean([d[1] for d in div]))
        print(s, json.dumps({k: v for k, v in res["sets"][s].items() if not isinstance(v, dict)}), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
