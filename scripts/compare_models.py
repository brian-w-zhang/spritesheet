"""Visual + metric comparison: v1 one-shot vs v2 masked diffusion on the same prompts (and steps filmstrip).

uv run python scripts/compare_models.py --v1 artifacts/model/v32 --v2 artifacts/model/v2_mdm_a --tag a
→ artifacts/figures/compare_v1_v2_<tag>.png, artifacts/figures/v2_steps_<tag>.png, artifacts/metrics/compare_<tag>.json
"""
import argparse, json, os, sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from spritesheet.mdm import MDMNet  # noqa: E402
from spritesheet.reference import lcg_uniform  # noqa: E402
from make_figures import font, INK, MUTED, PAPER  # noqa: E402

PROMPTS = ["a fox", "a red fox", "a cat", "a dog", "an owl", "a fish", "a turtle", "a butterfly",
           "a rocket", "a car", "a bicycle", "a house", "a crown", "a guitar", "a pizza", "a strawberry",
           "a tree", "a flower", "a mushroom", "a ghost", "a robot", "a snowman", "a blue cat", "a tiny dog in the top left"]
MASK_RGB = (185, 180, 194)


def load_v2(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    wv = np.load(os.path.join(E.BUILD, "wordvec.npy"))
    net = MDMNet(cfg, wv)
    net.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu"))
    net.cfg.update(cfg)
    return net.double().eval(), cfg


@torch.no_grad()
def gen_v2(net, cfg, caps, vocab, seeds, style_override=None):
    ids = torch.tensor([T.encode(c, vocab) for c in caps])
    c = net.encode(ids)
    p = torch.softmax(net.style_head(c), -1).numpy()
    st = []
    for i, s in enumerate(seeds):
        if style_override is not None:
            st.append(int(style_override[i])); continue
        u, cum, below = lcg_uniform(s), 0.0, 0
        for pi in p[i]:
            cum += float(pi); below += cum < u
        st.append(min(below, p.shape[1] - 1))
    x, hist = net.generate(ids, torch.tensor(st), cfg["T"], cfg["tau"])
    return x.numpy().astype(np.uint8), [h.numpy() for h in hist], np.array(st)


def rgb_with_mask(g, rgb, K):
    out = np.zeros(g.shape + (3,), np.uint8)
    m = g == K + 1
    out[~m] = rgb[np.clip(g[~m], 0, K)]
    out[m] = MASK_RGB
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=os.path.join(ROOT, "artifacts", "model", "v32"))
    ap.add_argument("--v2", required=True)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt"))
    rgb = E.palette_rgb()
    net1, cfg1 = E.load_net(a.v1)
    net2, cfg2 = load_v2(a.v2)
    K = cfg2["K"]
    seeds = [a.seed] * len(PROMPTS)
    g1, st1 = E.generate(net1, PROMPTS, vocab, seeds)
    g2, hist, st2 = gen_v2(net2, cfg2, PROMPTS, vocab, seeds, style_override=st1)  # same style for a fair look
    cell, pad, lab = 150, 16, 34
    cols = 8
    rows = (len(PROMPTS) + cols - 1) // cols
    W = pad + cols * (cell * 2 + 8 + pad)
    H = 110 + rows * (cell + lab + pad)
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    d.text((pad, 24), "v1 one-shot (left) vs v2 masked diffusion (right), same prompt, seed and style", fill=INK, font=font(34))
    d.text((pad, 70), f"v2: {cfg2['T']} denoising steps, {cfg2.get('params')} params; v1: {cfg1.get('params')} params", fill=MUTED, font=font(20))
    f = font(18)
    for i, p in enumerate(PROMPTS):
        r, c = divmod(i, cols)
        x, y = pad + c * (cell * 2 + 8 + pad), 110 + r * (cell + lab + pad)
        im.paste(Image.fromarray(E.to_rgb(g1[i], rgb)).resize((cell, cell), Image.NEAREST), (x, y))
        im.paste(Image.fromarray(E.to_rgb(g2[i], rgb)).resize((cell, cell), Image.NEAREST), (x + cell + 8, y))
        d.text((x, y + cell + 6), p, fill=INK, font=f)
    os.makedirs(os.path.join(ROOT, "artifacts", "figures"), exist_ok=True)
    out1 = os.path.join(ROOT, "artifacts", "figures", f"compare_v1_v2_{a.tag}.png")
    im.save(out1)
    # filmstrip for 6 prompts
    sel = [0, 2, 8, 11, 16, 20]
    Tn = len(hist)
    cs = 120
    im2 = Image.new("RGB", (pad + (Tn) * (cs + 10) + 150, 90 + len(sel) * (cs + 14)), PAPER)
    d2 = ImageDraw.Draw(im2)
    d2.text((pad, 24), "v2: the image forming over the denoising steps (grey = still unknown)", fill=INK, font=font(30))
    for j, i in enumerate(sel):
        y = 80 + j * (cs + 14)
        d2.text((pad, y + cs // 2 - 10), PROMPTS[i], fill=INK, font=font(18))
        for t in range(Tn):
            im2.paste(Image.fromarray(rgb_with_mask(hist[t][i], rgb, K)).resize((cs, cs), Image.NEAREST), (150 + t * (cs + 10), y))
    out2 = os.path.join(ROOT, "artifacts", "figures", f"v2_steps_{a.tag}.png")
    im2.save(out2)
    print("wrote", out1, out2)


if __name__ == "__main__":
    main()
