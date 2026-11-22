"""Add new models to the existing generation bank (artifacts/bank) without regenerating the old ones.

  uv run python scripts/bank_add_models.py --add v4s=v4s_a@g=0.75:t=0.8:u=v4s_u48 --label v4s="v4 slots (sheet size)" \
      --spec v4s=v4s_a_sheet

Keeps the bank's prompts, seeds, models and visual-review verdicts. For each new model: generates every prompt x seed
with the batched torch twin (CPU, so the serial GPU queue is not disturbed), checks --parity pairs against the
float64 spec export (--spec, optional), and scores with CLIP on the bank's existing scale (z-scores use the mean/std of
the models already in the bank, so old scores do not move). New models are listed first in each row.
Rewrites artifacts/bank/index.html (public: model sprites only) and, if present, index_local.html (keeps its real
sprites; local only, gitignored). Appends to artifacts/metrics/bank_summary.json under "added".
"""
import argparse, json, os, re, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402
from eval_v3 import Model  # noqa: E402
from gen_bank import OUT, b64, clip_on, real_ids  # noqa: E402

RX = re.compile(r"const DATA = (\{.*?\});\n", re.S)


def kv(items):
    return dict(x.split("=", 1) for x in items or [])


def parity(spec_dir, m, prompts, seeds, n, guidance):
    from spritesheet import reference_p as RP
    from spritesheet.pixcell import gates as G
    w, cfg, vocab, _, _ = G.load_model(os.path.join(ROOT, "artifacts", "model", spec_dir))
    vix = T.index_vocab(vocab)
    rng = np.random.default_rng(1)
    pairs = [(prompts[i]["prompt"], int(seeds[j])) for i, j in zip(rng.integers(0, len(prompts), n), rng.integers(0, len(seeds), n))]
    grids, _ = m.generate([p for p, _ in pairs], [s for _, s in pairs])
    c = dict(cfg, guidance=guidance) if guidance is not None else cfg
    ok = sum(bool((RP.forward(T.encode(p, vix), s, w, c, keep=False)["pixels"] == g).all()) for (p, s), g in zip(pairs, grids))
    return {"spec_model": spec_dir, "pairs": n, "pixel_exact": int(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="+", required=True, help="key=model spec (eval_v3 syntax)")
    ap.add_argument("--label", nargs="*", help="key=label")
    ap.add_argument("--spec", nargs="*", help="key=spec export dir under artifacts/model (parity check)")
    ap.add_argument("--parity", type=int, default=24)
    ap.add_argument("--replace", action="store_true", help="allow replacing a key already in the bank")
    a = ap.parse_args()
    add, labels, specs = kv(a.add), kv(a.label), kv(a.spec)
    path = os.path.join(OUT, "index.html")
    data = json.loads(RX.search(open(path).read()).group(1))
    prompts, seeds = data["prompts"], data["seeds"]
    for k in add:
        if k in data["models"] and not a.replace:
            sys.exit(f"{k} is already in the bank (use --replace)")
    caps_all = [p["prompt"] for p in prompts for _ in seeds]
    seeds_all = [s for _ in prompts for s in seeds]
    summ_path = os.path.join(ROOT, "artifacts", "metrics", "bank_summary.json")
    summary = json.load(open(summ_path))
    grids, pal = {}, {}
    for k, spec in add.items():
        m = Model(spec)
        cache = os.path.join(OUT, f"bank_{k}_grids.npz")
        t0 = time.time()
        if os.path.exists(cache) and str(np.load(cache)["spec"]) == spec:
            g, dt = np.load(cache)["grids"], float(np.load(cache)["seconds"])
        else:
            g = np.concatenate([m.generate(caps_all[i:i + 512], seeds_all[i:i + 512])[0] for i in range(0, len(caps_all), 512)])
            dt = time.time() - t0
            np.savez_compressed(cache, grids=g, spec=spec, seconds=dt)
        grids[k], pal[k] = g, m.rgb
        dist = [len({x.tobytes() for x in g[i * len(seeds):(i + 1) * len(seeds)]}) for i in range(len(prompts))]
        info = {"spec": spec, "sprites": len(g), "seconds": round(dt, 1), "ms_per_sprite": round(1000 * dt / len(g), 2),
                "distinct_per_prompt_mean": round(float(np.mean(dist)), 2), "device": "cpu",
                "time": time.strftime("%Y-%m-%d %H:%M")}
        if k in specs:
            info["parity_vs_spec"] = parity(specs[k], m, prompts, seeds, a.parity, m.cfg.get("guidance"))
        summary.setdefault("added", {})[k] = info
        print(k, info, flush=True)

    img, txt = clip_on(E.Clip(), "cpu")
    t_emb = txt([f"pixel art of {p['prompt']}" for p in prompts])
    z = np.load(os.path.join(ROOT, "data", "build", "v3", "imgs_24.npz"), allow_pickle=True)
    tr = z["split"] == 0
    caps, imgs = z["caption"][tr], z["imgs"][tr]
    rid = real_ids([dict(p) for p in prompts], caps)
    rpal = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in data["models"]["v3"]["palette"]], np.uint8)
    flat = np.concatenate(rid)
    uniq, inv = np.unique(flat, return_inverse=True)
    r_emb = img([rpal[imgs[i]] for i in uniq])[torch.from_numpy(inv.reshape(-1))]
    cent, j = [], 0
    for r in rid:
        e = r_emb[j:j + len(r)].mean(0) if len(r) else torch.zeros(r_emb.shape[1])
        cent.append(torch.nn.functional.normalize(e, dim=0) if len(r) else e)
        j += len(r)
    cent = torch.stack(cent)
    pidx = np.repeat(np.arange(len(prompts)), len(seeds))
    old = [m for k, m in data["models"].items() if k not in add]
    allct = np.concatenate([np.array(m["clip_text"]) for m in old])
    allcr = np.concatenate([np.array(m["clip_real"]) for m in old])
    new_models = {}
    for k in add:
        e = img([pal[k][x] for x in grids[k]])
        ct, cr = (e * t_emb[pidx]).sum(1).numpy(), (e * cent[pidx]).sum(1).numpy()
        sc = ((ct - allct.mean()) / allct.std() + (cr - allcr.mean()) / allcr.std()) / 2
        new_models[k] = {"label": labels.get(k, k), "n": int(grids[k].shape[1]),
                         "palette": ["#%02x%02x%02x" % tuple(c) for c in pal[k]],
                         "sprites": b64(grids[k].reshape(len(grids[k]), -1)),
                         "clip_text": np.round(ct, 4).tolist(), "clip_real": np.round(cr, 4).tolist(), "score": np.round(sc, 3).tolist()}
        summary["added"][k].update(mean_clip_text=round(float(ct.mean()), 4), mean_clip_real=round(float(cr.mean()), 4))
        np.savez_compressed(os.path.join(OUT, f"bank_{k}.npz"), grids=grids[k], clip_text=ct, clip_real=cr)
    for f in ("index.html", "index_local.html"):
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            continue
        html = open(p).read()
        d = json.loads(RX.search(html).group(1))
        d["models"] = {**new_models, **{k: v for k, v in d["models"].items() if k not in new_models}}
        open(p, "w").write(RX.sub(lambda _: "const DATA = " + json.dumps(d) + ";\n", html, count=1))
        print("wrote", p, list(d["models"]))
    json.dump(summary, open(summ_path, "w"), indent=1)


if __name__ == "__main__":
    main()
