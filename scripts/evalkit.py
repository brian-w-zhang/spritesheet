"""E1 evaluation harness: every model on the same fixed prompts (eval/prompts_e1.json) and the same 8 seeds.
Answers the MP2 critiques: no best-of-8 (every number is a mean over all seeds), a judge other than CLIP (a local
VLM does blind multiple-choice naming, scripts/vlm_judge.py), a near-duplicate check against the training set, and
side-by-side contact sheets so differences are confirmed by eye.

  uv run python scripts/evalkit.py gen   --name v3 --spec "v3p_a@g=0.75:t=0.8:u=v3p_u48"
  uv run python scripts/evalkit.py gen   --name real          # real training/test sprites (LOCAL ONLY: the ceiling)
  uv run python scripts/evalkit.py score --name v3            # → artifacts/metrics/e1_v3.json (+ VLM if judged)
  uv run python scripts/evalkit.py sheet --names v3,v3g0 --group creatures   # → artifacts/eval/e1/sheets/*.png

Outputs live in artifacts/eval/e1/<name>/ (grids.npz; "real" is gitignored: it holds PixelGPT sprites).
"""
import argparse, json, os, re, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
EV = os.path.join(ROOT, "artifacts", "eval", "e1")
BG = np.array([0xD8, 0xD4, 0xDD], np.uint8)
HUE_OK = {"red": {0, 11}, "orange": {0, 1}, "yellow": {1, 2}, "green": {3, 4, 5}, "blue": {6, 7, 8}, "purple": {8, 9, 10},
          "pink": {10, 11, 0}, "black": {99}, "white": {99}, "golden": {1, 2}}


def load_set():
    d = json.load(open(os.path.join(ROOT, "eval", "prompts_e1.json")))
    return d["meta"], d["prompts"]


def palette_rgb():
    from spritesheet.data.v3 import OUT as V3
    pal = json.load(open(os.path.join(V3, "palette.json")))["colors"]
    return np.concatenate([BG[None], np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])


def gen(a):
    meta, P = load_set()
    seeds = meta["seeds"]
    out = os.path.join(EV, a.name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if a.name == "real":
        # up to 8 real sprites per prompt: test prompts → the test sprite (repeated); others → training sprites whose
        # caption contains the phrase (else the noun). LOCAL ONLY.
        from spritesheet.data.v3 import OUT as V3
        z = np.load(os.path.join(V3, "imgs_24.npz"))
        imgs, caps, split = z["imgs"], z["caption"], z["split"]
        tr = np.nonzero(split == 0)[0]
        norm = [" " + re.sub(r"[^a-z ]", " ", str(c).lower()) + " " for c in caps[tr]]
        rng = np.random.default_rng(0)
        grids, src = np.zeros((len(P), len(seeds), 24, 24), np.uint8), []
        for i, p in enumerate(P):
            if "test_index" in p:
                grids[i] = imgs[p["test_index"]]
                src.append("test")
                continue
            phrase = " " + p["prompt"].split(" ", 1)[1] + " "
            hit = [j for j, c in enumerate(norm) if phrase in c]
            kind = "phrase"
            if not hit:
                hit, kind = [j for j, c in enumerate(norm) if f" {p['noun']} " in c], "noun"
            sel = rng.choice(hit, len(seeds), replace=len(hit) < len(seeds)) if hit else []
            for k, j in enumerate(sel):
                grids[i, k] = imgs[tr[j]]
            src.append(f"{kind}:{len(hit)}")
        np.savez_compressed(os.path.join(out, "grids.npz"), grids=grids, source=np.array(src))
    else:
        from eval_v3 import Model
        m = Model(a.spec)
        caps = [p["prompt"] for p in P for _ in seeds]
        ss = [s for _ in P for s in seeds]
        g, _ = m.generate(caps, ss)
        grids = np.asarray(g).reshape(len(P), len(seeds), 24, 24).astype(np.uint8)
        np.savez_compressed(os.path.join(out, "grids.npz"), grids=grids)
    json.dump({"name": a.name, "spec": a.spec, "set": meta["name"], "seconds": round(time.time() - t0, 1),
               "made": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(out, "gen.json"), "w"), indent=1)
    print(a.name, grids.shape, f"{time.time() - t0:.1f}s")


def hue_bucket(rgb):
    """dominant 30-degree hue bucket of the foreground pixels (99 = mostly achromatic)."""
    x = rgb.reshape(-1, 3).astype(float) / 255
    mx, mn = x.max(1), x.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    chrom = (sat > 0.25) & (mx > 0.2)
    if chrom.mean() < 0.3:
        return 99
    r, g, b = x[chrom].T
    d = np.maximum(mx[chrom] - mn[chrom], 1e-6)
    h = np.where(mx[chrom] == r, ((g - b) / d) % 6, np.where(mx[chrom] == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return int(np.bincount((h // 30).astype(int) % 12, minlength=12).argmax())


def score(a):
    from spritesheet import evaluate as E
    meta, P = load_set()
    S = len(meta["seeds"])
    grids = np.load(os.path.join(EV, a.name, "grids.npz"))["grids"]
    rgb = palette_rgb()
    n = len(P)
    fg = grids > 0
    res = {"name": a.name, "set": meta["name"], "n_prompts": n, "seeds": S}
    groups = list(dict.fromkeys(p["group"] for p in P))
    gi = {g: [i for i, p in enumerate(P) if p["group"] == g] for g in groups}
    empty = fg.sum((2, 3)) == 0                                     # [n, S]
    # CLIP retrieval: each sprite (all seeds) against the n prompt texts
    clip = E.Clip()
    txt = clip.txt([f"pixel art of {p['prompt']}" for p in P])
    flat = grids.reshape(-1, 24, 24)
    ie = clip.img([rgb[g] for g in flat]).reshape(n, S, -1)
    sims = torch.einsum("nsd,md->nsm", ie, txt)
    own = sims[torch.arange(n), :, torch.arange(n)]                   # [n, S]
    rank = (sims > own[..., None]).sum(-1)                           # [n, S]
    r5 = (rank < 5).float().numpy()
    res["clip_R@5"] = {g: round(float(r5[gi[g]].mean()), 4) for g in groups} | {"all": round(float(r5.mean()), 4)}
    res["clip_score"] = round(float(own.mean()), 4)
    # colour words: dominant hue matches
    col = []
    for i in gi.get("colour", []):
        c = P[i]["prompt"].split()[1]
        for s in range(S):
            if not empty[i, s]:
                col.append(hue_bucket(rgb[grids[i, s]][fg[i, s]]) in HUE_OK[c])
            else:
                col.append(False)
    res["colour_acc"] = round(float(np.mean(col)), 4) if col else None
    # size: foreground area of "giant X" / "tiny X" (same noun, same seed)
    area = fg.sum((2, 3)).astype(float)
    pairs = {}
    for i in gi.get("size", []):
        pairs.setdefault(P[i]["noun"], {})[P[i]["prompt"].split()[1]] = i
    ratios = [area[d["giant"]].mean() / max(area[d["tiny"]].mean(), 1) for d in pairs.values() if len(d) == 2]
    res["size_giant_over_tiny_area"] = round(float(np.mean(ratios)), 3) if ratios else None
    res["size_giant_bigger_share"] = round(float(np.mean([area[d["giant"]].mean() > area[d["tiny"]].mean() for d in pairs.values()])), 3) if pairs else None
    # basic image statistics (vs "real", compare by eye)
    ncol = np.array([[len(np.unique(g[g > 0])) for g in row] for row in grids])
    res["empty_share"] = round(float(empty.mean()), 4)
    res["fg_frac"] = round(float(fg.mean()), 4)
    res["colours_per_sprite"] = round(float(ncol.mean()), 2)
    # seed diversity: mean pairwise share of differing pixels across the 8 seeds
    dv = [(grids[i][:, None] != grids[i][None]).mean((2, 3))[np.triu_indices(S, 1)].mean() for i in range(n)]
    res["seed_diversity"] = round(float(np.mean(dv)), 4)
    distinct = [len({g.tobytes() for g in grids[i]}) for i in range(n)]
    res["distinct_per_8"] = round(float(np.mean(distinct)), 2)
    # near-duplicates of training sprites: min share of differing pixels over the training set
    if a.name != "real":
        res.update(near_dup(grids.reshape(-1, 24, 24)))
    # VLM judgments (scripts/vlm_judge.py) if present
    jp = os.path.join(EV, a.name, "vlm.json")
    if os.path.exists(jp):
        J = json.load(open(jp))
        res["vlm"] = summarize_vlm(J, P, S, gi)
    out = os.path.join(ROOT, "artifacts", "metrics", f"e1_{a.name}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def near_dup(flat, thresh=0.05):
    from spritesheet.data.v3 import OUT as V3
    z = np.load(os.path.join(V3, "imgs_24.npz"))
    tr = torch.as_tensor(z["imgs"][z["split"] == 0].reshape(-1, 576))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr = tr.to(dev)
    x = torch.as_tensor(flat.reshape(-1, 576)).to(dev)
    best = torch.full((len(x),), 576, dtype=torch.int32, device=dev)
    for i in range(0, len(x), 64):
        xb = x[i:i + 64]
        m = torch.full((len(xb),), 576, dtype=torch.int32, device=dev)
        for j in range(0, len(tr), 20000):
            d = (xb[:, None] != tr[None, j:j + 20000]).sum(-1, dtype=torch.int32)
            m = torch.minimum(m, d.min(1).values)
        best[i:i + 64] = m
    b = best.float().cpu().numpy() / 576
    return {"neardup_share_lt5pct": round(float((b < thresh).mean()), 4), "neardup_min_diff_median": round(float(np.median(b)), 4)}


def summarize_vlm(J, P, S, gi):
    """J = {"mc": [[letter-correct bool per seed] per prompt], "eyes": {...}, "facing": {...}, "named": ...}"""
    out = {}
    mc = np.array(J["mc"], dtype=float)                               # [n, S]
    out["naming_acc"] = {g: round(float(mc[gi[g]].mean()), 4) for g in gi} | {"all": round(float(mc.mean()), 4)}
    if "eyes" in J:
        e = np.array(J["eyes"], dtype=float)                          # [n_creatures, S]
        out["eyes_visible"] = round(float(e.mean()), 4)
    if "facing" in J:
        out["facing_acc"] = round(float(np.mean(J["facing"])), 4)
    if "quality" in J:
        out["quality_1to5"] = round(float(np.mean(J["quality"])), 3)
    return out


def sheet(a):
    from PIL import Image, ImageDraw, ImageFont
    meta, P = load_set()
    names = a.names.split(",")
    G = {n: np.load(os.path.join(EV, n, "grids.npz"))["grids"] for n in names}
    rgb = palette_rgb()
    idx = [i for i, p in enumerate(P) if a.group in ("all", p["group"])][: a.max_rows]
    S, z, pad = len(meta["seeds"]), a.zoom, 2
    cw = 24 * z + pad
    lw = 230
    W, H = lw + S * cw + 10, len(idx) * (len(names) * cw + 10) + 30
    im = Image.new("RGB", (W, H), (250, 250, 247))
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        f = ImageFont.load_default()
    dr.text((8, 8), f"E1 · group {a.group} · rows: {' / '.join(names)} · seeds {meta['seeds'][0]}–{meta['seeds'][-1]} (no picking)", fill=(40, 40, 40), font=f)
    y = 30
    for i in idx:
        for k, n in enumerate(names):
            label = (P[i]["prompt"][:30] + f"  [{P[i]['n_train']}]") if k == 0 else ""
            dr.text((8, y + k * cw + 8), label, fill=(20, 20, 20), font=f)
            dr.text((lw - 70, y + k * cw + 8 + (14 if k == 0 else 0)), n[:10], fill=(120, 120, 120), font=f)
            for s in range(S):
                tile = Image.fromarray(rgb[G[n][i, s]]).resize((24 * z, 24 * z), Image.NEAREST)
                im.paste(tile, (lw + s * cw, y + k * cw))
        y += len(names) * cw + 10
    os.makedirs(os.path.join(EV, "sheets"), exist_ok=True)
    tag = a.out or ("_vs_".join(names) + f"_{a.group}")
    local = "real" in names
    p = os.path.join(EV, "sheets", f"{tag}{'_local' if local else ''}.png")
    im.save(p)
    print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score", "sheet"])
    ap.add_argument("--name")
    ap.add_argument("--spec")
    ap.add_argument("--names")
    ap.add_argument("--group", default="creatures")
    ap.add_argument("--max-rows", type=int, default=40)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    {"gen": gen, "score": score, "sheet": sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
