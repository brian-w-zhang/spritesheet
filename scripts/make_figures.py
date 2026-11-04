"""F: slide figures from the trained model + metrics. Every sprite shown is produced by our model
(or is a ground-truth target, labelled as such).

uv run python scripts/make_figures.py --model artifacts/model/v32_main
"""
import argparse, json, os, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402

FIG = os.path.join(ROOT, "artifacts", "figures")
INK, MUTED, PAPER, ACCENT = (32, 30, 38), (120, 116, 128), (255, 255, 255), (232, 112, 91)


def font(sz):
    for p in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def grid(items, cols, cell=200, title=None, sub=None, path=None, label_lines=3):
    """items: list of (rgb uint8 [N,N,3] or None, caption, small_label)"""
    rows = (len(items) + cols - 1) // cols
    pad, lab = 28, 26 * label_lines + 10
    top = 110 if title else 30
    W = max(1600, pad + cols * (cell + pad))
    cell = (W - pad) // cols - pad
    H = top + rows * (cell + lab + pad) + pad
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    if title:
        d.text((pad, 26), title, fill=INK, font=font(40))
        if sub:
            d.text((pad, 74), sub, fill=MUTED, font=font(22))
    f, fs = font(21), font(17)
    for i, (img, cap, small) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = pad + c * (cell + pad), top + r * (cell + lab + pad)
        if img is not None:
            im.paste(Image.fromarray(img).resize((cell, cell), Image.NEAREST), (x, y))
        d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(228, 226, 232))
        for j, line in enumerate(wrap(d, cap, f, cell)[:label_lines - (1 if small else 0)]):
            d.text((x, y + cell + 8 + j * 25), line, fill=INK, font=f)
        if small:
            d.text((x, y + cell + 8 + (label_lines - 1) * 25), small, fill=MUTED, font=fs)
    os.makedirs(FIG, exist_ok=True)
    im.save(path)
    print("wrote", path, im.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    from spritesheet.data.synth import eval_set
    net, cfg = E.load_net(a.model)
    vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt"))
    rgb = E.palette_rgb()
    styles = json.load(open(os.path.join(E.BUILD, "styles.json")))
    gen = lambda caps, seeds, so=None: E.generate(net, caps, vocab, seeds, style_override=so)

    # 1. hero grid: free prompts (written here; not from the training templates)
    hero = ["a red apple", "a blue rocket", "a golden crown", "a green dragon", "a tiny cat in the top left",
            "two pizzas", "a purple mushroom", "an upside down ghost", "a pink heart on a blue background",
            "a dog, twemoji style", "a 3d robot", "pixel art sword", "a yellow star", "a mirrored car",
            "a white snowman", "a black cat"]
    g, st = gen(hero, list(range(1, 17)))
    grid([(E.to_rgb(x, rgb), p, f"seed {i + 1} · style {styles[s]}") for i, (x, p, s) in enumerate(zip(g, hero, st))], 4,
         title="Prompts → sprites, computed by SpriteSheet v32", sub="every sprite is model output (argmax over a 33-class palette)",
         path=os.path.join(FIG, "hero_grid.png"))

    # 2. variations of one concept
    var = ["a cat", "a blue cat", "a red cat", "a mirrored cat", "an upside down cat", "a tiny cat",
           "a tiny cat in the top left", "a tiny cat in the bottom right", "two cats", "a cat on a green background",
           "a cat, openmoji style", "a cat, 3d"]
    g, st = gen(var, [3] * len(var))
    grid([(E.to_rgb(x, rgb), p, styles[s]) for x, p, s in zip(g, var, st)], 6, title="One concept, controlled attributes",
         sub="colour · direction · size · position · count · background · style", path=os.path.join(FIG, "variations.png"))

    # 3. model vs exact target on held-out combos (given the target's style), with diff count
    data = eval_set("heldout_combo", 8, tier=int(cfg.get("tier", 0)))
    caps, tg, sid = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    g, _ = gen(caps, list(range(1, 9)), sid)
    items = []
    for c, t, x in zip(caps, tg, g):
        items.append((E.to_rgb(t, rgb), c, "target (exact, never trained on)"))
        diff = E.to_rgb(x, rgb).copy()
        diff[x != t] = (diff[x != t] * 0.35 + np.array(ACCENT) * 0.65).astype(np.uint8)
        items.append((E.to_rgb(x, rgb), "model", f"{int((x != t).sum())} px differ"))
        items.append((diff, "diff", "wrong pixels tinted"))
        items.append((None, "", ""))
    grid([it for it in items if it[1] != ""], 6, title="Held-out combinations: model vs exact target",
         sub="(concept, attribute) pairs excluded from training; style fixed to the target's",
         path=os.path.join(FIG, "vs_target.png"))

    # 4. unseen words
    data = eval_set("unseen_words", 12, tier=int(cfg.get("tier", 0)))
    caps = [d[0] for d in data]
    g, st = gen(caps, list(range(1, 13)), [d[2] for d in data])
    grid([(E.to_rgb(x, rgb), c, "word never in a training caption") for x, c in zip(g, caps)], 6,
         title="Unseen words (only the frozen CLIP word table links them to trained concepts)",
         path=os.path.join(FIG, "unseen_words.png"))

    # 5. seeds
    p = "an apple"
    g, st = gen([p] * 8, list(range(1, 9)))
    grid([(E.to_rgb(x, rgb), f"seed {i + 1}", styles[s]) for i, (x, s) in enumerate(zip(g, st))], 8,
         title=f'"{p}" × 8 seeds', sub="the in-sheet LCG samples a drawing style when the prompt doesn't name one",
         path=os.path.join(FIG, "seeds.png"))

    # 6. vs SpriteX32 (examples saved by bench_spritex32.py)
    ex = os.path.join(ROOT, "artifacts", "tmp_sx_examples.npy")
    if os.path.exists(ex):
        items = []
        for cap, sx, ours in np.load(ex, allow_pickle=True):
            items += [(np.asarray(sx, dtype=np.uint8), cap, "SpriteX32 (11.4M + CLIP, 50 steps, GPU)"),
                      (np.asarray(ours, dtype=np.uint8), cap, "SpriteSheet (1 pass, in a spreadsheet)")]
        grid(items, 6, title="Same prompts: SpriteX32 vs SpriteSheet", path=os.path.join(FIG, "vs_spritex32.png"))


if __name__ == "__main__":
    main()
