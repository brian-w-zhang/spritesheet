"""Generation bank: thousands of sprites from the deployed sheet models, generated headless, for picking slide images.

  uv run python scripts/gen_bank.py                      # 740 prompts x 8 seeds x {v3, v2, v1-XL} as deployed
  uv run python scripts/gen_bank.py --per-cat 5 --seeds 2  # quick run

Prompts come from the training captions: per dataset category, the most frequent words and two-word phrases
(generic words like "shape" or "corner" dropped), so the bank covers what the data shows most.

Models (the exact configurations on the Google Sheets):
  v2: masked diffusion v2c_T4_b06, 4 steps (sheet "diffusion")
  v3: PatchGPT v3p_a + guidance chain v3p_u48, guidance 0.75, temperature 0.8 (sheet "v3")
  v1: v1-XL, one pass (sheet "v32")
Generation uses the batched PyTorch twins; a sample is checked against the float64 specs (reference_p.forward /
reference.forward), which the sheets match pixel for pixel (parity_sheets_v3p.json, docs/09).

Scores per sprite (higher is better), to sort by "looks like the thing":
  clip_text:  CLIP similarity to "pixel art of <prompt>"
  clip_real:  CLIP similarity to the mean embedding of up to 16 real training sprites whose caption contains the prompt
  score:      mean of the two, each z-scored over the whole bank
Real sprites are only embedded, never written out, unless --with-real (local-only viewer, gitignored, never commit).

Outputs:
  artifacts/bank/bank.npz, artifacts/bank/index.html (viewer: sort, filter, star favourites, export JSON)
  artifacts/metrics/bank_summary.json (timing, parity, counts)
"""
import argparse, base64, collections, gzip, json, os, re, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from eval_v3 import Model  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "bank")
MODELS = {"v3": "v3p_a@g=0.75:t=0.8:u=v3p_u48", "v2": "v2c_T4_b06", "v1": "v1_xl"}
SPEC = {"v3": "v3p_a_g1_sheet", "v2": "v2c_T4_b06", "v1": "v1_xl"}
LABEL = {"v3": "v3 (sheet: guidance 0.75)", "v2": "v2 diffusion (4 steps)", "v1": "v1-XL"}
STOP = set("""a an the of and with in on for to from at by or its it is as into over under out up off no not
small large big tiny little old new open closed full empty half single double round square side top end corner shape
line point patch strip piece part edge front back left right view icon sprite pixel art style simple glow glowing
silhouette cluster pile bundle stack set pair group design pattern detail details base inside outside long short
tall wide thin thick dark light bright pale deep""".split())
# adjectives/materials: fine inside a two-word phrase ("rusty key"), not as a prompt on their own
ADJ = set("""red orange yellow green blue purple pink brown black white gray grey golden gold silver bronze brass copper
iron steel wooden wood stone glass plastic metal paper leather cloth clay ceramic porcelain rubber aluminum cotton wool
silk ghostly hollow rusty rusted broken torn bent worn frayed tarnished dented chipped twisted coiled tied tangled woven
sealed folded smashed melted frozen cracked dried wrapped patched spiked floating shattered magnifying fishing solar
space power garden miniature geometric color split""".split())


DEMO = ["red potion", "golden crown", "blue slime", "red frog", "black cat", "cat", "dog", "frog", "fox", "owl", "red apple", "potion", "health potion", "sword", "shield",
        "treasure chest", "dragon", "ghost", "slime", "wizard", "knight", "mushroom", "tree", "house", "castle",
        "rocket", "heart", "star", "skull", "goose", "duck", "fish", "pizza", "cake"]


# adjective variants: base nouns x attribute axes, to see what the models learn about colour, material, state, size, pose
NOUNS = {
    "creature": """cat dog frog fox owl bird fish snake turtle rabbit bear wolf mouse horse pig cow chicken duck goose bat
        spider crab snail butterfly bee beetle lizard dragon slime ghost golem demon skeleton zombie wizard knight robot
        goblin unicorn octopus shark whale penguin deer lion tiger monkey elephant squirrel jellyfish worm fairy imp witch
        king princess pirate ninja""",
    "object": """sword shield axe bow dagger hammer potion bottle jar cup bowl key ring crown coin gem crystal orb amulet
        candle lantern lamp book scroll map chest box barrel bag basket chair table bed door helmet boot hat glove mask
        bell clock mirror wand staff flag anchor wheel rocket car boat ship house castle tower tent bridge gate fence
        ladder rope chain skull bone heart star umbrella teapot guitar drum trophy feather egg""",
    "plant": "tree flower mushroom cactus leaf rose vine tulip sunflower bush log acorn seed",
    "food": "apple pear cake pizza bread cheese cookie carrot banana cherry grape pie burger donut candy strawberry watermelon",
    "nature": "rock cloud flame sun moon mountain volcano planet snowflake island",
}
AXES = {
    "colour": ["red", "blue", "green", "golden", "purple", "black", "white", "pink"],
    "material": ["wooden", "stone", "iron", "glass", "crystal", "ice", "gold"],
    "state": {"object": ["broken", "burning", "frozen", "rusty", "glowing", "cracked", "ancient", "magic"],
              "food": ["rotten", "frozen", "sliced", "burnt", "glowing", "giant slice of"],
              "plant": ["dead", "glowing", "frozen", "burning", "flowering"],
              "nature": ["burning", "frozen", "glowing", "dark", "floating"],
              "creature": ["angry", "happy", "baby", "armored", "skeleton", "zombie", "ghost", "robot"]},
    "size": ["tiny", "giant"],
    "pose": ["sleeping", "flying", "running", "sitting", "jumping"],
}
# per kind: how many adjectives to draw from each axis
QUOTA = {"creature": {"colour": 3, "material": 1, "state": 3, "size": 1, "pose": 2},
         "object": {"colour": 3, "material": 3, "state": 3, "size": 1},
         "plant": {"colour": 3, "material": 1, "state": 3, "size": 1},
         "food": {"colour": 2, "material": 1, "state": 3, "size": 1},
         "nature": {"colour": 3, "material": 1, "state": 3, "size": 1}}


# frequent caption modifiers (mined: words that precede >= 60 different words and start >= 40 captions;
# kept the adjectives and materials, dropped nouns)
COMMON = """wooden stone iron metal broken frozen steel glass crystal wrapped copper cracked hollow leather plastic clay
baby giant folded brass red bamboo paper silver floating tin ice torn bronze woven ghostly canvas ceramic cute dried
blue green white rusty black obsidian bent golden happy marble silk miniature wax friendly cotton rainbow fabric cloth
coral hanging spiral split sleepy soft rusted concrete cursed rubber mini smashed twisted carved frayed melted wool
smiling sealed cheerful stacked chipped straw cardboard candy shattered neon shadow lava snow moss jade glowing ancient
burnt tiny old magic""".split()


def caption_modifiers(caps, nouns, min_count=2, top=20):
    """Per noun: words that come right before it in training captions (e.g. 'poison apple', 'locked chest')."""
    pre = collections.defaultdict(collections.Counter)
    for c in caps:
        t = re.findall(r"[a-z][a-z'-]+", str(c).lower())
        for x, y in zip(t, t[1:]):
            if y in nouns and x not in STOP and x != y:
                pre[y][x] += 1
    return {n: [(w, c) for w, c in pre[n].most_common(top) if c >= min_count] for n in nouns}


def variant_prompts(seen, caps=None, n_common=12):
    out = []
    nouns = [(kind, w) for kind, ws in NOUNS.items() for w in ws.split()]
    if caps is not None:   # 1) modifiers seen with this noun in the training captions
        mods = caption_modifiers(caps, {w for _, w in nouns})
        for kind, noun in nouns:
            for w, c in mods[noun]:
                p = f"{w} {noun}"
                if p not in seen:
                    seen.add(p)
                    out.append({"prompt": p, "category": f"v_{kind}", "base": noun, "axis": "seen", "n_train": int(c)})
    for n, (kind, noun) in enumerate(nouns):   # 2) frequent caption adjectives, rotated over the nouns
        for t in range(n_common):
            p = f"{COMMON[(n * n_common + t) % len(COMMON)]} {noun}"
            if p not in seen:
                seen.add(p)
                out.append({"prompt": p, "category": f"v_{kind}", "base": noun, "axis": "common"})
    for kind, words in NOUNS.items():
        for n, noun in enumerate(words.split()):
            for axis, q in QUOTA[kind].items():
                pool = AXES[axis][kind] if axis == "state" else AXES[axis]
                for t in range(q):   # rotate through the pool so every adjective appears across nouns
                    adj = pool[(n * q + t) % len(pool)]
                    p = f"{adj} {noun}"
                    if p not in seen:
                        seen.add(p)
                        out.append({"prompt": p, "category": f"v_{kind}", "base": noun, "axis": axis})
    return out


def pick_prompts(caps, cats, per_cat, min_count=8):
    by = collections.defaultdict(list)
    for c, k in zip(caps, cats):
        by[k.split("/")[0]].append(c)
    low = [c.lower() for c in caps]
    out = [{"prompt": p, "category": "00_demo", "base": p.split()[-1], "axis": "demo", "n_train": sum(bool(re.search(rf"\b{p}\b", c)) for c in low)}
           for p in DEMO]
    seen = set(DEMO)
    for k in sorted(by):
        words, bigr = collections.Counter(), collections.Counter()
        for c in by[k]:
            w = [x for x in re.findall(r"[a-z][a-z'-]+", c.lower())]
            words.update(set(x for x in w if x not in STOP and x not in ADJ and len(x) > 2))
            bigr.update(set(f"{a} {b}" for a, b in zip(w, w[1:]) if a not in STOP - {"red", "blue"} and b not in STOP
                            and len(a) > 2 and len(b) > 2))
        nw, nb = per_cat * 2 // 3, per_cat - per_cat * 2 // 3
        for pool, n in ((words, nw), (bigr, nb)):
            got = 0
            for p, cnt in pool.most_common():
                if got >= n or cnt < min_count:
                    break
                if p in seen:
                    continue
                seen.add(p)
                out.append({"prompt": p, "category": k, "base": p.split()[-1], "axis": "base", "n_train": int(cnt)})
                got += 1
    return out


def real_ids(prompts, caps, cap_n=16, seed=0):
    """Up to cap_n training sprites whose caption contains the prompt as a phrase (falls back to the base noun)."""
    rank = np.random.default_rng(seed).permutation(len(caps))
    low = [c.lower() for c in caps.astype(str)]
    widx = collections.defaultdict(set)
    for i, c in enumerate(low):
        for w in set(re.findall(r"[a-z][a-z'-]*", c)):
            widx[w].add(i)

    def hits(phrase):
        ws = phrase.split()
        cand = set.intersection(*[widx.get(w, set()) for w in ws]) if ws else set()
        pat = re.compile(rf"\b{re.escape(phrase)}\b")
        return sorted(i for i in cand if pat.search(low[i]))
    res = []
    for p in prompts:
        h = hits(p["prompt"])
        p.setdefault("n_train", len(h))
        if len(h) < 4 and p.get("base") and p["base"] != p["prompt"]:
            h = hits(p["base"])
        h = sorted(h, key=lambda i: rank[i])[:cap_n]   # fixed order: variants falling back to a noun share its set
        res.append(np.array(h, int))
    return res


def check_parity(key, m, prompts, seeds, n):
    """Batched torch twin vs the float64 spec on n (prompt, seed) pairs."""
    from spritesheet.pixcell import gates as G
    w, cfg, vocab, _, _ = G.load_model(os.path.join(ROOT, "artifacts", "model", SPEC[key]))
    vix = T.index_vocab(vocab)
    rng = np.random.default_rng(1)
    pairs = [(prompts[i]["prompt"], int(seeds[j])) for i, j in
             zip(rng.integers(0, len(prompts), n), rng.integers(0, len(seeds), n))]
    grids, _ = m.generate([p for p, _ in pairs], [s for _, s in pairs])
    ok = 0
    for (p, s), g in zip(pairs, grids):
        if key == "v3":
            from spritesheet import reference_p as RP
            ref = RP.forward(T.encode(p, vix), s, w, dict(cfg, guidance=0.75), keep=False)["pixels"]
        elif key == "v2":
            from spritesheet import mdm
            ref = mdm.ref_generate(T.encode(p, vix), s, w, cfg, keep_steps=False)["classes"]
        else:
            from spritesheet import reference as R
            ref = R.forward(T.encode(p, vix), s, w, cfg)["classes"]
        ok += bool((np.asarray(ref) == g).all())
    return {"spec_model": SPEC[key], "pairs": n, "pixel_exact": ok}


def clip_on(clip, dev):
    clip.m = clip.m.to(dev)
    proc = clip.p

    @torch.no_grad()
    def img(rgbs):
        from PIL import Image
        out = []
        for i in range(0, len(rgbs), 256):
            ims = [Image.fromarray(r).resize((224, 224), Image.NEAREST) for r in rgbs[i:i + 256]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            e = clip.m.get_image_features(pixel_values=px)
            e = getattr(e, "pooler_output", e)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def txt(texts):
        out = []
        for i in range(0, len(texts), 256):
            b = proc(text=texts[i:i + 256], return_tensors="pt", padding=True)
            e = clip.m.get_text_features(**{k: v.to(dev) for k, v in b.items()})
            e = getattr(e, "pooler_output", e)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(out)
    return img, txt


def b64(a):
    """gzip + base64 (the viewer inflates it with DecompressionStream; ~10x smaller than raw)."""
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a, np.uint8).tobytes(), 9)).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=34)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--parity", type=int, default=24)
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--no-variants", action="store_true", help="base prompts only (no adjective variants)")
    ap.add_argument("--with-real", action="store_true", help="also write index_local.html with 4 real sprites per prompt")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "data", "build", "v3", "imgs_24.npz"), allow_pickle=True)
    tr = z["split"] == 0
    caps, cats, imgs = z["caption"][tr], z["category"][tr], z["imgs"][tr]
    prompts = pick_prompts(caps, cats, a.per_cat)
    if not a.no_variants:
        prompts += variant_prompts({p["prompt"] for p in prompts}, caps)[: a.max_variants]
    seeds = list(range(1, a.seeds + 1))
    print(f"{len(prompts)} prompts x {len(seeds)} seeds", flush=True)

    summary = {"prompts": len(prompts), "base_prompts": sum(p["axis"] in ("base", "demo") for p in prompts),
               "variant_prompts": sum(p["axis"] not in ("base", "demo") for p in prompts), "seeds": seeds, "models": {}, "time": time.strftime("%Y-%m-%d %H:%M")}
    caps_all = [p["prompt"] for p in prompts for _ in seeds]
    seeds_all = [s for _ in prompts for s in seeds]
    grids, rgbpal = {}, {}
    for key, spec in MODELS.items():
        m = Model(spec)
        m.generate(caps_all[:16], seeds_all[:16])  # warm-up
        t0 = time.time()
        g = np.concatenate([m.generate(caps_all[i:i + 1024], seeds_all[i:i + 1024])[0]
                            for i in range(0, len(caps_all), 1024)])   # chunks: v2 would otherwise hold all logits
        dt = time.time() - t0
        grids[key], rgbpal[key] = g, m.rgb
        par = check_parity(key, m, prompts, seeds, a.parity)
        dist = [len({x.tobytes() for x in g[i * len(seeds):(i + 1) * len(seeds)]}) for i in range(len(prompts))]
        summary["models"][key] = {"spec": spec, "sprites": len(g), "seconds": round(dt, 1),
                                  "distinct_per_prompt_mean": round(float(np.mean(dist)), 2),
                                  "prompts_all_seeds_identical": int(sum(d == 1 for d in dist)),
                                  "ms_per_sprite": round(1000 * dt / len(g), 2), "parity_vs_spec": par}
        print(key, summary["models"][key], flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    img, txt = clip_on(E.Clip(), dev)
    t_emb = txt([f"pixel art of {p['prompt']}" for p in prompts])
    rid = real_ids(prompts, caps)
    flat = np.concatenate(rid)
    BG = rgbpal["v3"][0]
    rpal = rgbpal["v3"]  # dataset sprites use the v3 48-colour palette (index 0 = background)
    uniq, inv = np.unique(flat, return_inverse=True)
    r_emb = img([rpal[imgs[i]] for i in uniq])[torch.from_numpy(inv.reshape(-1))] if len(flat) else torch.zeros(0, 512)
    cent, k = [], 0
    for r in rid:
        e = r_emb[k:k + len(r)].mean(0) if len(r) else torch.zeros(r_emb.shape[1])
        cent.append(torch.nn.functional.normalize(e, dim=0) if len(r) else e)
        k += len(r)
    cent = torch.stack(cent)
    pidx = np.repeat(np.arange(len(prompts)), len(seeds))
    scores = {}
    for key in MODELS:
        e = img([rgbpal[key][x] for x in grids[key]])
        ct = (e * t_emb[pidx]).sum(1).numpy()
        cr = (e * cent[pidx]).sum(1).numpy()
        scores[key] = (ct, cr)
    allct = np.concatenate([s[0] for s in scores.values()]); allcr = np.concatenate([s[1] for s in scores.values()])
    zt = lambda x: (x - allct.mean()) / allct.std()
    zr = lambda x: (x - allcr.mean()) / allcr.std()

    np.savez_compressed(os.path.join(OUT, "bank.npz"), prompts=np.array([p["prompt"] for p in prompts]),
                        seeds=np.array(seeds), **{f"grids_{k}": grids[k] for k in MODELS},
                        **{f"clip_text_{k}": scores[k][0] for k in MODELS}, **{f"clip_real_{k}": scores[k][1] for k in MODELS})
    data = {"prompts": prompts, "seeds": seeds, "models": {}}
    for key in MODELS:
        ct, cr = scores[key]
        sc = (zt(ct) + zr(cr)) / 2
        n = grids[key].shape[1]
        data["models"][key] = {"label": LABEL[key], "n": int(n),
                               "palette": ["#%02x%02x%02x" % tuple(c) for c in rgbpal[key]],
                               "sprites": b64(grids[key].reshape(len(grids[key]), -1)),
                               "clip_text": np.round(ct, 4).tolist(), "clip_real": np.round(cr, 4).tolist(),
                               "score": np.round(sc, 3).tolist()}
        summary["models"][key]["mean_clip_text"] = round(float(ct.mean()), 4)
        summary["models"][key]["mean_clip_real"] = round(float(cr.mean()), 4)
    html = open(os.path.join(ROOT, "scripts", "bank_viewer.html")).read()
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html.replace("/*DATA*/null", json.dumps(data)))
    if a.with_real:
        real = [b64(imgs[r[:4]].reshape(len(r[:4]), -1)) if len(r) else "" for r in rid]
        data["real"] = {"n": 24, "palette": data["models"]["v3"]["palette"], "sprites": real}
        with open(os.path.join(OUT, "index_local.html"), "w") as f:
            f.write(html.replace("/*DATA*/null", json.dumps(data)))
    summary["real_refs_per_prompt_mean"] = round(float(np.mean([len(r) for r in rid])), 1)
    json.dump(summary, open(os.path.join(ROOT, "artifacts", "metrics", "bank_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
