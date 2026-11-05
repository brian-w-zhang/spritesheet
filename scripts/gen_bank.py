"""Generation bank: thousands of sprites from the deployed sheet models, generated headless, for picking slide images.

  uv run python scripts/gen_bank.py                      # ~700 prompts x 8 seeds x {v3 deployed, v1-XL}
  uv run python scripts/gen_bank.py --per-cat 5 --seeds 2  # quick run

Prompts come from the training captions: per dataset category, the most frequent words and two-word phrases
(generic words like "shape" or "corner" dropped), so the bank covers what the data shows most.

Models (the exact configurations on the Google Sheets):
  v3: PatchGPT v3p_a + guidance chain v3p_u48, guidance 0.75, temperature 0.8 (sheet "v3")
  v1: v1-XL, one pass (sheet "v32")
Generation uses the batched PyTorch twins; a sample is checked against the float64 specs (reference_p.forward /
reference.forward), which the sheets match pixel for pixel (parity_sheets_v3p.json, docs/09).

Scores per sprite (higher is better), to sort by "looks like the thing":
  clip_text:  CLIP similarity to "pixel art of <prompt>"
  clip_real:  CLIP similarity to the mean embedding of up to 32 real training sprites whose caption contains the prompt
  score:      mean of the two, each z-scored over the whole bank
Real sprites are only embedded, never written out, unless --with-real (local-only viewer, gitignored, never commit).

Outputs:
  artifacts/bank/bank.npz, artifacts/bank/index.html (viewer: sort, filter, star favourites, export JSON)
  artifacts/metrics/bank_summary.json (timing, parity, counts)
"""
import argparse, base64, collections, json, os, re, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from eval_v3 import Model  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "bank")
MODELS = {"v3": "v3p_a@g=0.75:t=0.8:u=v3p_u48", "v1": "v1_xl"}
SPEC = {"v3": "v3p_a_g1_sheet", "v1": "v1_xl"}
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


DEMO = ["black cat", "cat", "dog", "frog", "fox", "owl", "red apple", "potion", "health potion", "sword", "shield",
        "treasure chest", "dragon", "ghost", "slime", "wizard", "knight", "mushroom", "tree", "house", "castle",
        "rocket", "heart", "star", "skull", "goose", "duck", "fish", "pizza", "cake"]


def pick_prompts(caps, cats, per_cat, min_count=8):
    by = collections.defaultdict(list)
    for c, k in zip(caps, cats):
        by[k.split("/")[0]].append(c)
    low = [c.lower() for c in caps]
    out = [{"prompt": p, "category": "00_demo", "n_train": sum(bool(re.search(rf"\b{p}\b", c)) for c in low)}
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
                out.append({"prompt": p, "category": k, "n_train": int(cnt)})
                got += 1
    return out


def real_ids(prompts, caps, cap_n=32, seed=0):
    rng = np.random.default_rng(seed)
    low = np.char.lower(caps.astype(str))
    res = []
    for p in prompts:
        pat = re.compile(rf"\b{re.escape(p['prompt'])}\b")
        hit = np.array([i for i, c in enumerate(low) if pat.search(c)])
        res.append(rng.choice(hit, min(cap_n, len(hit)), replace=False) if len(hit) else np.zeros(0, int))
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
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=34)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--parity", type=int, default=24)
    ap.add_argument("--with-real", action="store_true", help="also write index_local.html with 4 real sprites per prompt")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "data", "build", "v3", "imgs_24.npz"), allow_pickle=True)
    tr = z["split"] == 0
    caps, cats, imgs = z["caption"][tr], z["category"][tr], z["imgs"][tr]
    prompts = pick_prompts(caps, cats, a.per_cat)
    seeds = list(range(1, a.seeds + 1))
    print(f"{len(prompts)} prompts x {len(seeds)} seeds", flush=True)

    summary = {"prompts": len(prompts), "seeds": seeds, "models": {}, "time": time.strftime("%Y-%m-%d %H:%M")}
    caps_all = [p["prompt"] for p in prompts for _ in seeds]
    seeds_all = [s for _ in prompts for s in seeds]
    grids, rgbpal = {}, {}
    for key, spec in MODELS.items():
        m = Model(spec)
        m.generate(caps_all[:16], seeds_all[:16])  # warm-up
        t0 = time.time()
        g, _ = m.generate(caps_all, seeds_all)
        dt = time.time() - t0
        grids[key], rgbpal[key] = g, m.rgb
        par = check_parity(key, m, prompts, seeds, a.parity)
        summary["models"][key] = {"spec": spec, "sprites": len(g), "seconds": round(dt, 1),
                                  "ms_per_sprite": round(1000 * dt / len(g), 2), "parity_vs_spec": par}
        print(key, summary["models"][key], flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    img, txt = clip_on(E.Clip(), dev)
    t_emb = txt([f"pixel art of {p['prompt']}" for p in prompts])
    rid = real_ids(prompts, caps)
    flat = np.concatenate(rid)
    BG = rgbpal["v3"][0]
    rpal = rgbpal["v3"]  # dataset sprites use the v3 48-colour palette (index 0 = background)
    r_emb = img([rpal[imgs[i]] for i in flat])
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
        data["models"][key] = {"label": "v3 (sheet: guidance 0.75)" if key == "v3" else "v1-XL", "n": int(n),
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
