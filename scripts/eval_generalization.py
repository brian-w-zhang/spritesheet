"""Generalisation metrics that don't assume one correct drawing:
- CLIP R-precision (standard T2I metric): for each generated image, rank all prompts of the set by CLIP similarity;
  R@1 = fraction whose own prompt ranks first (R@5 likewise). Chance = 1/len(set).
- Mean CLIP score (image·prompt cosine).
Sets: NOVEL (prompts from neither dataset, written here) and PixelGPT test captions (free text). For PixelGPT we also
score the ground-truth sprites, as an upper reference.

uv run python scripts/eval_generalization.py --models artifacts/model/v32,artifacts/model/abl_emojionly --out artifacts/metrics/generalization.json
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402

NOVEL = ["a knight", "a wizard hat", "a castle", "a dragon", "a teapot", "a lighthouse", "a treasure chest", "a sword",
         "a shield", "a potion bottle", "a magic wand", "a skull", "a ninja", "a pirate ship", "a cactus", "a palm tree",
         "a volcano", "a rainbow", "a moon", "a sun", "a cloud", "a lightning bolt", "a snowflake", "a campfire",
         "a tent", "a backpack", "a key", "a lock", "a bell", "a candle", "a lantern", "a book", "a scroll", "a map",
         "a compass", "a telescope", "a camera", "a robot dog", "a cat wearing a hat", "a blue dragon", "a red car",
         "a green frog", "a yellow duck", "a pink pig", "a brown bear", "a white rabbit", "a black cat", "an orange fish",
         "a purple flower", "a tiny mouse", "a giant mushroom", "a slime monster", "a ghost with a lantern",
         "a cupcake", "an ice cream cone", "a donut", "a hamburger", "a slice of watermelon", "a carrot", "a banana",
         "a cherry", "a coffee cup", "a bottle of milk", "a fried egg", "a sandwich", "a chocolate bar", "a lollipop",
         "a trophy", "a medal", "a crown with jewels", "a diamond", "a coin", "a heart", "a star", "an arrow",
         "a bow", "an axe", "a hammer", "a pickaxe", "a shovel", "a bucket", "a flower pot", "a house with a red roof",
         "a tower", "a bridge", "a windmill", "a train", "a bus", "a boat", "a submarine", "a hot air balloon",
         "a rocket ship", "an alien", "a ufo", "a planet", "a satellite", "a dinosaur", "a whale", "an octopus", "a crab"]


def r_precision(img_emb, txt_emb):
    sims = img_emb @ txt_emb.T
    rank = (sims > sims.diag()[:, None]).sum(1)
    return {"R@1": float((rank == 0).float().mean()), "R@5": float((rank < 5).float().mean()),
            "clip_score": float(sims.diag().mean()), "chance_R@1": 1.0 / len(txt_emb)}


def gen_any(model_dir, caps, vocab):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    seeds = [1 + i for i in range(len(caps))]
    if cfg.get("model") == "mdm":
        from compare_models import load_v2, gen_v2
        net, c2 = load_v2(model_dir)
        g, _, _ = gen_v2(net, c2, caps, vocab, seeds)
        return g
    net, _ = E.load_net(model_dir)
    g, _ = E.generate(net, caps, vocab, seeds)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n-pixelgpt", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "metrics", "generalization.json"))
    a = ap.parse_args()
    from spritesheet.data.synth import eval_set
    vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt"))
    rgb = E.palette_rgb()
    clip = E.Clip()
    pg = eval_set("pixelgpt_test", a.n_pixelgpt, tier=1)
    pg_caps = [d[0] for d in pg]
    sets = {"novel": NOVEL, "pixelgpt_test": pg_caps}
    txt = {k: clip.txt(v) for k, v in sets.items()}
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res["_targets_pixelgpt"] = r_precision(clip.img([E.to_rgb(d[1], rgb) for d in pg]), txt["pixelgpt_test"])
    for md in a.models.split(","):
        name = os.path.basename(md.rstrip("/"))
        res[name] = {}
        for k, caps in sets.items():
            g = gen_any(md, caps, vocab)
            res[name][k] = r_precision(clip.img([E.to_rgb(x, rgb) for x in g]), txt[k])
        print(name, json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in res[name].items()}), flush=True)
    print("targets pixelgpt", {m: round(v, 3) for m, v in res["_targets_pixelgpt"].items()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
