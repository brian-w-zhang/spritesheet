"""Raw-source loaders (docs/03 "Sources" and "Joining sources into concepts").

Every emoji loader yields (concept_id, style_name, RGBA PIL image). The concept id is the lowercase
codepoint sequence joined by '-' with 'fe0f' removed (e.g. '1f34e' = red apple).
PixelGPT rows are yielded as dicts with the RGBA 24x24 image (tile_bg masked to transparent).
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
from functools import lru_cache

import numpy as np
from PIL import Image

from . import RAW

EMOJI_STYLES = ["noto", "twemoji", "openmoji", "fluent_3d", "fluent_color", "fluent_flat"]
STYLES = EMOJI_STYLES + ["pixelgpt"]
SVG_RENDER_PX = 128

SKIN_TONES = {f"{c:x}" for c in range(0x1F3FB, 0x1F400)}
# Person / family codepoints: ZWJ sequences containing any of these are excluded (gendered, families,
# professions, couples ...). Single-codepoint people (e.g. 'baby') are kept.
PERSON_CPS = {f"{c:x}" for c in list(range(0x1F466, 0x1F46A)) + [0x1F9D1, 0x1F9D2, 0x1F474, 0x1F475,
                                                                 0x1F476, 0x1F46A, 0x1F48F, 0x1F491]}
EXCLUDED_GROUPS = {"Flags", "Component"}
OPENMOJI_GROUPS = {
    "smileys-emotion": "Smileys & Emotion", "people-body": "People & Body", "animals-nature": "Animals & Nature",
    "food-drink": "Food & Drink", "travel-places": "Travel & Places", "activities": "Activities",
    "objects": "Objects", "symbols": "Symbols", "flags": "Flags", "component": "Component",
    "extras-openmoji": "Extras", "extras-unicode": "Extras",
}


def norm_cid(seq: str) -> str:
    """'1F34E-FE0F', '1f34e_fe0f', '1f34e fe0f' -> '1f34e'."""
    cps = [c.lower().lstrip("0") or "0" for c in re.split(r"[-_ ]+", seq.strip()) if c]
    return "-".join(c for c in cps if c != "fe0f")


def cid_of_chars(s: str) -> str:
    return "-".join(f"{ord(ch):x}" for ch in s if ord(ch) != 0xFE0F)


def excluded(cid: str, group: str) -> str | None:
    """Return the exclusion reason, or None if the concept is usable."""
    cps = cid.split("-")
    ints = [int(c, 16) for c in cps]
    if group in EXCLUDED_GROUPS or group == "Extras":
        return f"group:{group}"
    if any(0x1F1E6 <= i <= 0x1F1FF for i in ints) or any(0xE0020 <= i <= 0xE007F for i in ints):
        return "flag"
    if any(c in SKIN_TONES for c in cps):
        return "skin_tone"
    if "20e3" in cps:
        return "keycap"
    if "200d" in cps and (group == "People & Body" or any(c in PERSON_CPS for c in cps)):
        return "zwj_people"
    if len(ints) == 1 and ints[0] < 0x2000:
        return "ascii_symbol"  # (c), (r), digits, '#', '*': tiny symbols
    if any(0x1F9B0 <= i <= 0x1F9B3 for i in ints):
        return "component"
    return None


# ---------------------------------------------------------------- file indexes
@lru_cache(maxsize=1)
def emoji_index() -> dict[str, dict[str, str]]:
    """concept id -> {style: file path}. Unfiltered (exclusion happens in `concept_table`)."""
    idx: dict[str, dict[str, str]] = {}

    def add(cid, style, path):
        idx.setdefault(cid, {}).setdefault(style, path)

    for p in sorted(glob.glob(os.path.join(RAW, "noto-emoji", "2D", "png", "128", "emoji_u*.png"))):
        add(norm_cid(os.path.basename(p)[len("emoji_u"):-4]), "noto", p)
    for p in sorted(glob.glob(os.path.join(RAW, "twemoji", "assets", "72x72", "*.png"))):
        add(norm_cid(os.path.basename(p)[:-4]), "twemoji", p)
    for p in sorted(glob.glob(os.path.join(RAW, "openmoji", "color", "72x72", "*.png"))):
        add(norm_cid(os.path.basename(p)[:-4]), "openmoji", p)
    for cid, (d, _meta) in fluent_meta().items():
        for style, sub, pat in (("fluent_3d", "3D", "*.png"), ("fluent_color", "Color", "*.svg"),
                                ("fluent_flat", "Flat", "*.svg")):  # High Contrast skipped
            for base in (d, os.path.join(d, "Default")):
                files = sorted(glob.glob(os.path.join(base, sub, pat)))
                if files:
                    add(cid, style, files[0])
                    break
    return idx


@lru_cache(maxsize=1)
def fluent_meta() -> dict[str, tuple[str, dict]]:
    out = {}
    for meta in sorted(glob.glob(os.path.join(RAW, "fluentui-emoji", "assets", "*", "metadata.json"))):
        m = json.load(open(meta, encoding="utf-8"))
        if m.get("unicode"):
            out.setdefault(norm_cid(m["unicode"]), (os.path.dirname(meta), m))
    return out


@lru_cache(maxsize=1)
def openmoji_meta() -> dict[str, dict]:
    data = json.load(open(os.path.join(RAW, "openmoji", "data", "openmoji.json"), encoding="utf-8"))
    out = {}
    for e in data:
        out.setdefault(norm_cid(e["hexcode"]), e)
    return out


@lru_cache(maxsize=1)
def cldr() -> dict[str, dict]:
    """concept id -> {'name': tts, 'keywords': [...]}, from annotations-en + annotationsDerived-en."""
    out: dict[str, dict] = {}
    for fn, key in (("annotations-en.json", "annotations"), ("annotationsDerived-en.json", "annotationsDerived")):
        ann = json.load(open(os.path.join(RAW, "cldr", fn), encoding="utf-8"))[key]["annotations"]
        for ch, v in ann.items():
            cid = cid_of_chars(ch)
            e = out.setdefault(cid, {"name": None, "keywords": []})
            if e["name"] is None and v.get("tts"):
                e["name"] = v["tts"][0]
            e["keywords"] += v.get("default", [])
    return out


_KW_OK = re.compile(r"^[a-z][a-z' -]*[a-z]$")


def clean_words(words) -> list[str]:
    seen, out = set(), []
    for w in words:
        w = re.sub(r"\s+", " ", str(w).strip().lower().replace("_", " "))
        if w and _KW_OK.match(w) and len(w) >= 3 and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def concept_table(use_openmoji: bool = True) -> tuple[list[dict], dict[str, int]]:
    """All usable emoji concepts (sorted by id) with name, keywords, group, styles -> paths.
    Returns (concepts, exclusion_counts)."""
    idx, fm, om, cl = emoji_index(), fluent_meta(), openmoji_meta(), cldr()
    concepts, excl = [], {}
    for cid in sorted(idx):
        f = fm.get(cid, (None, {}))[1]
        o = om.get(cid, {})
        c = cl.get(cid, {})
        group = f.get("group") or OPENMOJI_GROUPS.get(o.get("group", ""), "") or ""
        reason = excluded(cid, group)
        name = c.get("name") or f.get("cldr") or o.get("annotation")
        if reason is None and not name:
            reason = "no_name"
        if reason:
            excl[reason] = excl.get(reason, 0) + 1
            continue
        paths = {s: p for s, p in idx[cid].items() if use_openmoji or s != "openmoji"}
        kws = clean_words(list(c.get("keywords", [])) + list(f.get("keywords", []))
                          + str(o.get("tags", "")).split(",") + str(o.get("openmoji_tags", "")).split(","))
        name = name.lower().strip()
        concepts.append({"id": cid, "name": name, "keywords": [k for k in kws if k != name],
                         "group": group or "Other", "paths": {s: paths[s] for s in EMOJI_STYLES if s in paths}})
    return concepts, excl


# ---------------------------------------------------------------- image loading
def load_rgba(path: str) -> Image.Image:
    if path.endswith(".svg"):
        import resvg_py
        png = resvg_py.svg_to_bytes(svg_path=path, width=SVG_RENDER_PX, height=SVG_RENDER_PX)
        return Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    with Image.open(path) as im:
        return im.convert("RGBA")


def iter_emoji(concepts=None, styles=None):
    """Yield (concept_id, style_name, RGBA image) for every (concept, style) file."""
    concepts = concepts if concepts is not None else concept_table()[0]
    for c in concepts:
        for s, p in c["paths"].items():
            if styles is None or s in styles:
                yield c["id"], s, load_rgba(p)


# ---------------------------------------------------------------- PixelGPT
PIXELGPT_SPLITS = {"train": "train", "val": "validation", "test": "test"}


def pixelgpt_rows(split: str) -> list[dict]:
    """Rows of one PixelGPT split: {'key', 'caption', 'category', 'quality', 'rgba' (24x24x4 uint8)}.
    Pixels exactly equal to tile_bg (RGB) become transparent."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(RAW, "pixelgpt-24x24-20k", "data", f"{PIXELGPT_SPLITS[split]}-*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["image", "caption", "category", "tile_bg", "quality_score", "sha256"])
                    for f in files], ignore_index=True)
    rows = []
    for r in df.itertuples(index=False):
        rgb = np.array(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"), dtype=np.uint8)
        bg = np.array(json.loads(r.tile_bg), dtype=np.uint8)
        alpha = np.where((rgb == bg).all(-1), 0, 255).astype(np.uint8)
        rows.append({"key": r.sha256, "caption": str(r.caption), "category": str(r.category),
                     "quality": float(r.quality_score), "rgba": np.dstack([rgb, alpha])})
    return rows
