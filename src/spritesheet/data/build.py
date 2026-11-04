"""Build the normalised dataset (docs/03, docs/08).

    uv run python -m spritesheet.data.build --tier 0|1 [--no-openmoji] [--refit-palette] [--workers 8]

Writes data/build/tier{T}/{palette.json, styles.json, concepts.json, bases_32.npz, splits.json,
build_report.json}. The palette and styles are shared by all tiers (palette fitted once on a fixed
sample of the full tier-1 pool) and are also written to data/build/{palette.json, styles.json};
data/build/concepts.json is the full (tier-1) concept list, of which tier 0 is a subset with identical
entries. data/build/caption_words.txt lists every word the captions can contain (for vocab.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np

from . import BUILD, build_dir
from . import sources as S
from . import synth
from .normalize import Palette, kmeans_palette, pad_pixel_art, render_emoji, small_size

N = 32
K = 32
TIER0_GROUPS = ["Food & Drink", "Animals & Nature", "Travel & Places", "Objects", "Activities"]
TIER0_PER_GROUP = 60
MIN_FG = 8  # drop renders with fewer foreground pixels


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def _render(job):
    cid, style, path, n = job
    img = S.load_rgba(path)
    big = render_emoji(img, n)
    sm = render_emoji(img, n, small_size(n))
    if big is None or sm is None or (big[..., 3] > 0).sum() < MIN_FG:
        return None
    return big, sm


def render_all(jobs, workers):
    if workers <= 1:
        return [_render(j) for j in jobs]
    with Pool(workers) as p:
        return p.map(_render, jobs, chunksize=32)


def select_concepts(concepts, tier, limit=None):
    if tier == 0:
        out = []
        for g in TIER0_GROUPS:
            cand = [c for c in concepts if c["group"] == g and len(c["paths"]) >= 3]
            cand.sort(key=lambda c: ("-" in c["id"], -len(c["paths"]), _h(c["id"])))
            out += cand[:TIER0_PER_GROUP]
        out.sort(key=lambda c: c["id"])
    else:
        out = [c for c in concepts if len(c["paths"]) >= 2]
    if limit:
        out = sorted(out, key=lambda c: _h(c["id"]))[:limit]
        out.sort(key=lambda c: c["id"])
    return out


def fit_palette(all_concepts, pg_train, n, k, workers, seed=0):
    """Shared palette: k-means over foreground pixels of a fixed sample (600 emoji concepts x all styles +
    4000 PixelGPT train sprites), half the pixels from each family."""
    rng = np.random.default_rng(seed)
    pool = [c for c in all_concepts if len(c["paths"]) >= 2]
    pool = sorted(pool, key=lambda c: _h("pal" + c["id"]))[:600]
    jobs = [(c["id"], s, p, n) for c in pool for s, p in c["paths"].items()]
    em = [r[0] for r in render_all(jobs, workers) if r is not None]
    epx = np.concatenate([a[a[..., 3] > 0][:, :3] for a in em])
    parts = [epx[rng.choice(len(epx), min(len(epx), 100_000), replace=False)]]
    if pg_train:
        idx = rng.choice(len(pg_train), min(4000, len(pg_train)), replace=False)
        ppx = np.concatenate([pg_train[i]["rgba"][pg_train[i]["rgba"][..., 3] > 0][:, :3] for i in sorted(idx)])
        parts.append(ppx[rng.choice(len(ppx), min(len(ppx), 100_000), replace=False)])
    return Palette(kmeans_palette(np.concatenate(parts), k=k, seed=seed))


def choose_test_keywords(concepts, extra_vocab: set[str]):
    """One held-out keyword per concept (when possible): a single-word keyword not in the concept's name,
    used by no other concept (name or keyword), and not a generator / PixelGPT-caption word. It is removed
    from training keywords, so it never appears in a training caption."""
    df = {}
    for c in concepts:
        toks = set(synth.words(c["name"])) | {t for k in c["keywords"] for t in synth.words(k)}
        for t in toks:
            df[t] = df.get(t, 0) + 1
    out = {}
    for c in concepts:
        name_t = set(synth.words(c["name"]))
        cand = [k for k in c["keywords"] if " " not in k and k.isalpha() and k not in name_t
                and df.get(k, 0) == 1 and k not in extra_vocab and k not in synth.COLOR_WORDS]
        if cand:
            k = sorted(cand, key=lambda w: _h(c["id"] + w))[0]
            out[c["id"]] = k
    return out


def arrays_hash(arrs: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build(tier: int, out: str | None = None, use_openmoji: bool = True, limit: int | None = None,
          pixelgpt_limit: int | None = None, palette: Palette | None = None, refit_palette: bool = False,
          workers: int | None = None, write_root: bool = True, quiet: bool = False):
    t0 = time.time()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    out = out or build_dir(tier)
    os.makedirs(out, exist_ok=True)
    log = (lambda *a: None) if quiet else (lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True))

    all_concepts, excl = S.concept_table(use_openmoji)
    styles = S.STYLES if use_openmoji else [s for s in S.STYLES if s != "openmoji"]
    sel = select_concepts(all_concepts, tier, limit)
    log(f"concept table: {len(all_concepts)} usable, selected {len(sel)} for tier {tier}")

    # PixelGPT (tier 1). Captions of the train split always feed the keyword-holdout vocabulary so
    # concept entries are identical across tiers.
    pg = {s: S.pixelgpt_rows(s) for s in ("train", "val", "test")}
    pg_vocab = {t for r in pg.get("train", []) for t in synth.words(r["caption"])}

    # ---- palette (shared)
    root_pal = os.path.join(BUILD, "palette.json")
    if palette is None:
        if os.path.exists(root_pal) and not refit_palette:
            palette = Palette.load(root_pal)
            log("palette: reused", root_pal)
        else:
            palette = fit_palette(all_concepts, pg.get("train", []), N, K, workers)
            os.makedirs(BUILD, exist_ok=True)
            palette.save(root_pal, N)
            log(f"palette: fitted K={palette.K}")

    # ---- emoji bases
    jobs = [(c["id"], s, c["paths"][s], N) for c in sel for s in styles if s in c["paths"]]
    rend = render_all(jobs, workers)
    log(f"rendered {len(jobs)} emoji drawings")
    keep = {}
    for j, r in zip(jobs, rend):
        if r is not None:
            keep.setdefault(j[0], []).append((j[1], r))
    min_styles = 3 if tier == 0 else 2
    sel = [c for c in sel if len(keep.get(c["id"], [])) >= min_styles]
    imgs, small, has_small, concept, style, split, pg_caption, pg_category, key = [], [], [], [], [], [], [], [], []
    for ci, c in enumerate(sel):
        for s, (big, sm) in keep[c["id"]]:
            imgs.append(palette.snap(big))
            small.append(palette.snap(sm))
            has_small.append(1)
            concept.append(ci)
            style.append(styles.index(s))
            split.append(0)
            pg_caption.append("")
            pg_category.append("")
            key.append(f"{c['id']}/{s}")
    n_emoji = len(imgs)

    # ---- PixelGPT bases
    pg_counts = {}
    if tier >= 1:
        for k_split, sname in enumerate(("train", "val", "test")):
            rows = pg[sname][:pixelgpt_limit] if pixelgpt_limit else pg[sname]
            pg_counts[sname] = len(rows)
            for r in rows:
                imgs.append(palette.snap(pad_pixel_art(r["rgba"], N)))
                small.append(np.zeros((N, N), np.uint8))
                has_small.append(0)
                concept.append(-1)
                style.append(styles.index("pixelgpt"))
                split.append(k_split)
                pg_caption.append(r["caption"])
                pg_category.append(r["category"])
                key.append(f"pixelgpt/{r['key']}")
    arrs = {"imgs": np.stack(imgs).astype(np.uint8), "small": np.stack(small).astype(np.uint8),
            "has_small": np.array(has_small, np.uint8), "concept": np.array(concept, np.int32),
            "style": np.array(style, np.int32), "split": np.array(split, np.int8),
            "pg_caption": np.array(pg_caption, dtype=str), "pg_category": np.array(pg_category, dtype=str),
            "key": np.array(key, dtype=str)}
    log(f"bases: {n_emoji} emoji + {len(imgs) - n_emoji} pixelgpt")

    # ---- concepts.json, held-out keywords (computed over ALL usable concepts so tiers agree)
    full = select_concepts(all_concepts, 1)
    gen_vocab = synth.generator_words()
    plurals = {synth._plural(c["name"]).split(" ")[-1] for c in full}  # "two ice skates"
    test_kw = choose_test_keywords(full, gen_vocab | pg_vocab | plurals)

    def entry(c, sty):
        kt = test_kw.get(c["id"])
        ktr = [k for k in c["keywords"] if not (kt and kt in synth.words(k))]
        return {"id": c["id"], "name": c["name"], "keywords_train": ktr, "keywords_test": [kt] if kt else [],
                "group": c["group"], "styles": sty}

    concepts_json = [entry(c, [s for s, _ in keep[c["id"]]]) for c in sel]

    # ---- held-out combos
    sat = synth.saturated_mask(palette)
    fg = arrs["imgs"] > 0
    colorable_base = (sat[arrs["imgs"]] & fg).sum((1, 2)) >= synth.COLORABLE_FRAC * np.maximum(1, fg.sum((1, 2)))
    colorable = {}
    for i in range(n_emoji):
        cid = sel[arrs["concept"][i]]["id"]
        colorable[cid] = colorable.get(cid, False) or bool(colorable_base[i])
    hc, ht = synth.heldout_pairs([c["id"] for c in sel], colorable)

    h = arrays_hash(arrs)
    splits = {
        "heldout_combo": {"colour": hc, "transform": ht, "rule": "sha1(cid|kind|value) % 10 == 0"},
        "heldout_phrase": {"templates_train": synth.TEMPLATES_TRAIN, "templates_heldout": synth.TEMPLATES_HELDOUT},
        "keywords_test": {c["id"]: c["keywords_test"] for c in concepts_json if c["keywords_test"]},
        "pixelgpt": pg_counts,
        "arrays_sha256": h,
    }
    # ---- write
    np.savez_compressed(os.path.join(out, f"bases_{N}.npz"), **arrs)
    palette.save(os.path.join(out, "palette.json"), N)
    json.dump(styles, open(os.path.join(out, "styles.json"), "w"))
    json.dump(concepts_json, open(os.path.join(out, "concepts.json"), "w"), indent=0)
    json.dump(splits, open(os.path.join(out, "splits.json"), "w"), indent=0)
    if write_root and use_openmoji and not limit:
        shutil.copy(os.path.join(out, "styles.json"), os.path.join(BUILD, "styles.json"))
        root_concepts = [entry(c, [s for s in styles if s in c["paths"]]) for c in full]
        json.dump(root_concepts, open(os.path.join(BUILD, "concepts.json"), "w"), indent=0)
        vocab = set(gen_vocab)
        for c in full:
            vocab |= set(synth.words(c["name"])) | {t for k in c["keywords"] for t in synth.words(k)}
            vocab |= {synth._plural(c["name"]).split(" ")[-1]}
        for s in ("train", "val", "test"):
            vocab |= {t for r in pg.get(s, []) for t in synth.words(r["caption"])}
        open(os.path.join(BUILD, "caption_words.txt"), "w").write("\n".join(sorted(vocab)) + "\n")
    style_counts = {s: int((arrs["style"] == i).sum()) for i, s in enumerate(styles)}
    groups = {}
    for c in concepts_json:
        groups[c["group"]] = groups.get(c["group"], 0) + 1
    report = {"tier": tier, "N": N, "K": palette.K, "n_concepts": len(sel), "n_bases": len(imgs),
              "n_emoji_bases": n_emoji, "n_pixelgpt_bases": len(imgs) - n_emoji, "pixelgpt_splits": pg_counts,
              "bases_per_style": style_counts, "concepts_per_group": groups, "usable_concepts_total": len(all_concepts),
              "excluded": excl, "n_heldout_colour_pairs": len(hc), "n_heldout_transform_pairs": len(ht),
              "n_colorable_concepts": sum(colorable.values()),
              "n_concepts_with_keywords_test": len(splits["keywords_test"]), "use_openmoji": use_openmoji,
              "arrays_sha256": h, "build_seconds": round(time.time() - t0, 1)}
    json.dump(report, open(os.path.join(out, "build_report.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in report.items() if k not in ("excluded", "concepts_per_group")}))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[0, 1], default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-openmoji", action="store_true")
    ap.add_argument("--refit-palette", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    build(a.tier, a.out, use_openmoji=not a.no_openmoji, refit_palette=a.refit_palette, workers=a.workers)


if __name__ == "__main__":
    main()
