"""Visual line-up: same prompts × several models (rows), same seed. For eyeballing, not metrics.
uv run python scripts/visual_lineup.py --models v32,abl_base,abl_emojionly,v2_mdm_a --tag now
"""
import argparse, json, os, sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from eval_generalization import gen_any  # noqa: E402
from make_figures import font, INK, MUTED, PAPER  # noqa: E402

GROUPS = {
    "trained emoji concepts": ["a fox", "a cat", "a dog", "an owl", "a rocket", "a car", "a house", "a pizza",
                               "a strawberry", "a tree", "a mushroom", "a ghost"],
    "novel prompts (in neither dataset)": ["a knight", "a castle", "a dragon", "a treasure chest", "a sword", "a potion bottle",
                                           "a cat wearing a hat", "a blue dragon", "a slime monster", "a lighthouse",
                                           "a pirate ship", "a robot dog"],
    "PixelGPT-style captions": ["ghost with a snail shell", "sad penguin", "pickup truck", "toad in mud", "noir drinking cup",
                                "cute mummy toad", "wizard with a staff", "red potion", "golden key", "wooden shield",
                                "green slime", "skeleton warrior"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", default="now")
    a = ap.parse_args()
    vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt"))
    rgb = E.palette_rgb()
    models = a.models.split(",")
    for gname, prompts in GROUPS.items():
        cs, pad, left = 96, 6, 190
        W = left + len(prompts) * (cs + pad) + 10
        H = 70 + len(models) * (cs + pad) + 60
        im = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(im)
        d.text((10, 10), gname + " · seed 1 for every prompt and model", fill=INK, font=font(24))
        for j, p in enumerate(prompts):
            d.text((left + j * (cs + pad), 42), p[:15], fill=MUTED, font=font(12))
        for i, m in enumerate(models):
            g = gen_any(os.path.join(ROOT, "artifacts", "model", m), prompts, vocab)
            y = 62 + i * (cs + pad)
            d.text((10, y + cs // 2 - 8), m, fill=INK, font=font(16))
            for j in range(len(prompts)):
                im.paste(Image.fromarray(E.to_rgb(g[j], rgb)).resize((cs, cs), Image.NEAREST), (left + j * (cs + pad), y))
        out = os.path.join(ROOT, "artifacts", "figures", f"lineup_{a.tag}_{gname.split()[0]}.png")
        im.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
