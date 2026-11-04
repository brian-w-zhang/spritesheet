"""S9 per-style quality + clean-up filter prototype, for one v1 model.

For each of the 7 styles (forced, instead of seed-sampled): CLIP R-precision on the NOVEL prompts and PixelGPT
test captions, a speckle measure (fraction of pixels a 3x3 majority filter would change) and a line-up row.
Also scores the same model with the 3x3 majority ("mode") clean-up applied to its output.

uv run python scripts/per_style.py --model artifacts/model/v32 --tag v32
→ artifacts/metrics/per_style_<tag>.json, artifacts/figures/per_style_<tag>_{trained,novel,PixelGPT-style}.png
"""
import argparse, json, os, sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from eval_generalization import NOVEL, r_precision  # noqa: E402
from visual_lineup import GROUPS  # noqa: E402
from make_figures import font, INK, MUTED, PAPER  # noqa: E402


def mode3(g):
    """3x3 majority filter: each pixel takes the most common class in its 3x3 neighbourhood (itself included,
    ties keep the pixel's own class). Borders use only in-canvas neighbours."""
    N = g.shape[0]
    out = g.copy()
    for i in range(N):
        for j in range(N):
            win = g[max(0, i - 1):i + 2, max(0, j - 1):j + 2].ravel()
            cnt = np.bincount(win)
            best = cnt.max()
            if cnt[g[i, j]] < best:
                out[i, j] = int(np.argmax(cnt))
    return out


def speckle(gs):
    return float(np.mean([(mode3(g) != g).mean() for g in gs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts/model/v32"))
    ap.add_argument("--tag", default="v32")
    ap.add_argument("--n-pixelgpt", type=int, default=300)
    a = ap.parse_args()
    from spritesheet.data.synth import eval_set
    vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt"))
    styles = json.load(open(os.path.join(E.BUILD, "styles.json")))
    styles = styles if isinstance(styles, list) else list(styles)
    rgb = E.palette_rgb()
    clip = E.Clip()
    net, _ = E.load_net(a.model)
    pg_caps = [d[0] for d in eval_set("pixelgpt_test", a.n_pixelgpt, tier=1)]
    sets = {"novel": NOVEL, "pixelgpt_test": pg_caps}
    txt = {k: clip.txt(v) for k, v in sets.items()}

    def score(gs, k):
        return r_precision(clip.img([E.to_rgb(x, rgb) for x in gs]), txt[k])

    res = {"model": a.model, "styles": {}}
    rows = {}  # (label) -> {group: grids}
    for label, sty in [("sampled (seed)", None)] + [(s, i) for i, s in enumerate(styles)]:
        r = {}
        for k, caps in sets.items():
            g, _ = E.generate(net, caps, vocab, [1 + i for i in range(len(caps))], style_override=None if sty is None else [sty] * len(caps))
            r[k] = score(g, k)
            r[k]["speckle"] = speckle(g)
            gm = [mode3(x) for x in g]
            r[k + "_mode3"] = score(gm, k)
        res["styles"][label] = r
        rows[label] = {gn: E.generate(net, ps, vocab, [1] * len(ps), style_override=None if sty is None else [sty] * len(ps))[0] for gn, ps in GROUPS.items()}
        print(label, json.dumps({k: {m: round(v, 3) for m, v in d.items() if m != "chance_R@1"} for k, d in r.items()}), flush=True)
    json.dump(res, open(os.path.join(ROOT, "artifacts", "metrics", f"per_style_{a.tag}.json"), "w"), indent=1)

    labels = list(rows)
    for gname, prompts in GROUPS.items():
        cs, pad, left = 96, 6, 190
        W = left + len(prompts) * (cs + pad) + 10
        H = 70 + (len(labels) + 1) * (cs + pad) + 20
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"{gname} · {os.path.basename(a.model)} · one row per style (seed 1)", fill=INK, font=font(24))
        for j, p in enumerate(prompts):
            d.text((left + j * (cs + pad), 42), p[:15], fill=MUTED, font=font(12))
        allrows = [(lb, rows[lb][gname]) for lb in labels] + [("sampled + 3x3 clean-up", [mode3(x) for x in rows[labels[0]][gname]])]
        for i, (lb, g) in enumerate(allrows):
            y = 62 + i * (cs + pad)
            d.text((10, y + cs // 2 - 8), lb, fill=INK, font=font(16))
            for j in range(len(prompts)):
                im.paste(Image.fromarray(E.to_rgb(g[j], rgb)).resize((cs, cs), Image.NEAREST), (left + j * (cs + pad), y))
        out = os.path.join(ROOT, "artifacts", "figures", f"per_style_{a.tag}_{gname.split()[0]}.png")
        im.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
