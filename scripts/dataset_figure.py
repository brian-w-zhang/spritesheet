"""Grid of synthetic training samples (captions + exact targets), nearest-neighbour upscaled.
Run: uv run python scripts/dataset_figure.py [--tier 1]  ->  artifacts/figures/dataset_samples.png
"""
import argparse
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

from spritesheet.data import ROOT
from spritesheet.data.synth import Sampler


def pick(sampler, want, tries=4000):
    """Deterministically pick samples covering each wanted kind."""
    got = {k: [] for k in want}
    for _ in range(tries):
        cap, g, s, m = sampler.sample()
        tags = []
        if m["kind"] == "pixelgpt":
            tags.append("pixelgpt")
        elif m["kind"] == "emoji_plain":
            tags.append("plain")
        for k in ("colour", "transform", "position", "count", "bg"):
            if m[k] is not None:
                tags.append(k)
        if m["size"] and m["position"] is None and m["count"] is None:
            tags.append("size")
        for t in tags:
            if t in got and len(got[t]) < want[t]:
                got[t].append((cap, g, m))
                break
        if all(len(got[k]) >= want[k] for k in want):
            break
    return [x for k in want for x in got[k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    s = Sampler("train", seed=a.seed, tier=a.tier)
    want = {"plain": 4, "colour": 4, "transform": 3, "size": 2, "position": 3, "count": 2, "bg": 2,
            "pixelgpt": 4 if len(s.pg["train"]) else 0}
    items = pick(s, want)
    cols, cell, pad, txt_h = 8, 224, 16, 84
    rows = (len(items) + cols - 1) // cols
    W, H = cols * (cell + pad) + pad, rows * (cell + pad + txt_h) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for i, (cap, g, m) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = pad + c * (cell + pad), pad + r * (cell + pad + txt_h)
        img = Image.fromarray(s.pal.to_rgb(g, bg=(236, 236, 236))).resize((cell, cell), Image.NEAREST)
        sheet.paste(img, (x, y))
        lines = textwrap.wrap(f'"{cap}"', 26)[:3] + [f"[{m['style']}]"]
        for j, ln in enumerate(lines):
            d.text((x, y + cell + 4 + j * 19), ln, fill=(30, 30, 30) if j < len(lines) - 1 else (120, 120, 120), font=font)
    out = os.path.join(ROOT, "artifacts", "figures", "dataset_samples.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print(out, sheet.size, len(items), "samples")


if __name__ == "__main__":
    main()
