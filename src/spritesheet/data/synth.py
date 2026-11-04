"""Synthetic captions + exact targets (docs/03 "Synthetic variations", docs/08 "Data -> model").

    Sampler(split, seed, tier=0, mix=None, pixelgpt_n=None).batch(B)
        -> (captions: list[str], targets uint8 [B,N,N], style_ids int64 [B], meta: list[dict])
    eval_set(name, n=300, tier=0) -> list of (caption, target [N,N], style_id, meta)

Every target is a deterministic function of (caption, style id): unspecified attributes take the
canonical default (original colours, centred, normal size, count 1, transparent background).
When the caption has no style word, the target style is uniform over the concept's styles (the
model's seeded style head handles that ambiguity); meta['styles_ok'] lists every style id that
is consistent with the caption.

meta keys: kind, concept (emoji concept id, or None for PixelGPT), concept_idx, category, style,
style_id, styles_ok, colour, transform, size, position, count, bg, template, base_idx.
Unspecified attributes are None.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import os
import re
from functools import lru_cache

import numpy as np

from . import BUILD, build_dir
from .normalize import Palette

# ---------------------------------------------------------------- vocabulary of the generator
STYLE_WORDS = {
    "noto": ["noto style", "in noto style"],
    "twemoji": ["twemoji style", "twitter style"],
    "openmoji": ["openmoji style", "outlined"],
    "fluent_3d": ["3d", "3d style"],
    "fluent_color": ["fluent style", "gradient style"],
    "fluent_flat": ["flat", "flat style"],
    "pixelgpt": ["pixel art"],
}
# caption word -> canonical colour key
COLOR_WORDS = {"red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue",
               "purple": "purple", "violet": "purple", "pink": "pink", "black": "black", "white": "white",
               "gray": "gray", "grey": "gray", "brown": "brown", "golden": "golden", "gold": "golden"}
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white", "gray", "brown", "golden"]
_WORDS_OF = {c: [w for w, k in COLOR_WORDS.items() if k == c] for c in COLORS}
# target hue (degrees) for hue-remap colours
_HUE = {"red": 0, "orange": 28, "yellow": 55, "green": 120, "blue": 220, "purple": 280, "pink": 330,
        "brown": 25, "golden": 45}
BG_COLORS = {"red": (220, 40, 40), "orange": (245, 140, 30), "yellow": (250, 220, 50), "green": (60, 170, 70),
             "blue": (50, 100, 220), "purple": (140, 60, 190), "pink": (245, 150, 190), "black": (20, 20, 20),
             "white": (250, 250, 250), "gray": (128, 128, 128), "brown": (130, 80, 40)}
BG_PHRASES = ["on a {c} background", "with a {c} background"]
TRANSFORMS = {"mirror": (["mirrored", "flipped"], ["mirrored", "flipped"]),  # (prefix words, suffix words)
              "upside_down": (["upside down"], ["upside down"])}
SIZE_WORDS = ["small", "tiny"]
COUNT_WORDS = {2: ["two", "a pair of"]}
POSITIONS = {(0, 0): ["in the top left", "at the top left"], (0, 1): ["at the top", "in the top middle"],
             (0, 2): ["in the top right", "at the top right"], (1, 0): ["on the left", "at the left"],
             (1, 1): ["in the center", "in the middle"], (1, 2): ["on the right", "at the right"],
             (2, 0): ["in the bottom left", "at the bottom left"], (2, 1): ["at the bottom", "in the bottom middle"],
             (2, 2): ["in the bottom right", "at the bottom right"]}
POS_NAMES = {p: v[0].split(" ", 1)[1] for p, v in POSITIONS.items()}  # e.g. 'the top left'
# {a} = phrase with article ("a red apple" / "two red apples"), {p} = bare phrase ("red apple")
TEMPLATES_TRAIN = ["{a}", "{p}", "{a} sprite", "{a} icon", "draw {a}", "a sprite of {a}", "an icon of {a}",
                   "{p} sprite"]
TEMPLATES_HELDOUT = ["a picture of {a}", "make {a}", "show me {a}", "{a} emoji", "an image of {a}"]
MAX_WORDS = 12  # tokenizer.L
DEFAULT_MIX = {1: {"emoji_var": 0.45, "pixelgpt": 0.30, "emoji_plain": 0.25},
               0: {"emoji_var": 0.60, "pixelgpt": 0.0, "emoji_plain": 0.40}}
HELDOUT_MOD = 10  # ~10% of (concept, colour) and (concept, transform) pairs held out
SAT_MIN, VAL_MIN, COLORABLE_FRAC = 0.25, 0.20, 0.15


def generator_words() -> set[str]:
    """Every word the caption generator itself can emit (templates, attributes, style words)."""
    txt = " ".join(TEMPLATES_TRAIN + TEMPLATES_HELDOUT + list(COLOR_WORDS) + BG_PHRASES + SIZE_WORDS
                   + [w for v in COUNT_WORDS.values() for w in v] + [w for v in POSITIONS.values() for w in v]
                   + [w for a, b in TRANSFORMS.values() for w in a + b] + [w for v in STYLE_WORDS.values() for w in v]
                   + ["a", "an", "background"])
    return set(words(txt.replace("{a}", " ").replace("{p}", " ").replace("{c}", " ")))


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def train_template_words() -> set[str]:
    return set(words(" ".join(TEMPLATES_TRAIN).replace("{a}", " ").replace("{p}", " ")))


def _h(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


# ---------------------------------------------------------------- held-out combos (used by build)
def heldout_pairs(concept_ids, colorable: dict[str, bool]):
    """Deterministic ~10% held-out (concept, colour) and (concept, transform) pairs; every concept keeps
    at least one training colour and one training transform."""
    hc, ht = [], []
    for cid in concept_ids:
        if colorable.get(cid):
            held = [c for c in COLORS if _h(cid, "colour", c) % HELDOUT_MOD == 0]
            if len(held) == len(COLORS):
                held = held[1:]
            hc += [[cid, c] for c in held]
        held = [t for t in TRANSFORMS if _h(cid, "transform", t) % HELDOUT_MOD == 0]
        if len(held) == len(TRANSFORMS):
            held = held[1:]
        ht += [[cid, t] for t in held]
    return hc, ht


# ---------------------------------------------------------------- palette lookup tables
def _hsv(rgb):
    return colorsys.rgb_to_hsv(*(np.asarray(rgb) / 255.0))


def saturated_mask(pal: Palette) -> np.ndarray:
    """bool [K+1]: palette classes that a colour variation recolours (not near-grey, not near-black)."""
    m = np.zeros(pal.K + 1, bool)
    for i, rgb in enumerate(pal.rgb):
        _, s, v = _hsv(rgb)
        m[i + 1] = s >= SAT_MIN and v >= VAL_MIN
    return m


def recolor_luts(pal: Palette) -> dict[str, np.ndarray]:
    """colour -> uint8 LUT [K+1] mapping class id -> class id. Saturated colours get the target hue
    (keeping their value; saturation floored), then are snapped to the palette in CIELAB; near-grey /
    outline / background classes map to themselves."""
    sat = saturated_mask(pal)
    luts = {}
    for c in COLORS:
        new = []
        for i, rgb in enumerate(pal.rgb):
            h, s, v = _hsv(rgb)
            if c in _HUE:
                h2 = _HUE[c] / 360.0
                if c == "pink":
                    s2, v2 = min(s, 0.45), max(v, 0.85)
                elif c == "brown":
                    s2, v2 = max(s, 0.6), min(v, 0.6) * 0.85
                elif c == "golden":
                    s2, v2 = max(s, 0.75), max(v, 0.75)
                else:
                    s2, v2 = max(s, 0.6), max(v, 0.45)
            elif c == "black":
                h2, s2, v2 = h, s * 0.3, v * 0.25
            elif c == "white":
                h2, s2, v2 = h, s * 0.08, 0.80 + 0.2 * v
            else:  # gray
                h2, s2, v2 = h, 0.0, 0.25 + 0.6 * v
            new.append(np.round(np.array(colorsys.hsv_to_rgb(h2, s2, v2)) * 255))
        snapped = _hue_snap(pal, np.array(new, dtype=np.uint8), c) + 1
        lut = np.arange(pal.K + 1)
        lut[sat] = snapped[sat[1:]]
        luts[c] = lut.astype(np.uint8)
    return luts


HUE_WINDOW = 30.0  # degrees


def _candidates(pal: Palette, colour: str) -> np.ndarray:
    """bool [K]: palette colours allowed as the snap target for a named colour. Hue colours: saturated
    palette colours within +-HUE_WINDOW (HSV hue) of the target hue; black/white/gray: low-saturation ones."""
    hsv = np.array([_hsv(c) for c in pal.rgb])
    if colour in _HUE:
        dh = np.abs(((hsv[:, 0] * 360 - _HUE[colour]) + 180) % 360 - 180)
        m = (dh <= HUE_WINDOW) & (hsv[:, 1] >= SAT_MIN) & (hsv[:, 2] >= VAL_MIN)
    else:
        m = hsv[:, 1] < 0.4 if colour == "black" else hsv[:, 1] < 0.15
    return m if m.any() else np.ones(pal.K, bool)


def _hue_snap(pal: Palette, rgb: np.ndarray, colour: str) -> np.ndarray:
    """Nearest palette index (0-based, CIELAB) restricted to the colour's candidate set."""
    from .normalize import srgb_to_lab
    d = ((srgb_to_lab(rgb)[:, None, :] - pal.lab[None]) ** 2).sum(-1)
    d[:, ~_candidates(pal, colour)] = np.inf
    return d.argmin(-1)


def bg_classes(pal: Palette) -> dict[str, int]:
    """Named background colour -> palette class (candidate-restricted nearest; distinct per name)."""
    from .normalize import srgb_to_lab
    out, used = {}, set()
    for c, rgb in BG_COLORS.items():
        d = ((srgb_to_lab(np.array(rgb, dtype=np.uint8)) - pal.lab) ** 2).sum(-1)
        d[~_candidates(pal, c)] += 1e6
        for j in np.argsort(d, kind="stable"):
            if int(j) not in used:
                used.add(int(j))
                out[c] = int(j) + 1
                break
    return out


# ---------------------------------------------------------------- data loading
@lru_cache(maxsize=4)
def load_build(tier: int):
    d = build_dir(tier)
    z = np.load(os.path.join(d, "bases_32.npz"))
    data = {k: z[k] for k in z.files}
    data["concepts"] = json.load(open(os.path.join(d, "concepts.json")))
    data["styles"] = json.load(open(os.path.join(d, "styles.json")))
    data["splits"] = json.load(open(os.path.join(d, "splits.json")))
    data["palette"] = Palette.load(os.path.join(d, "palette.json"))
    return data


# ---------------------------------------------------------------- image ops (on class-id grids)
def _transform(g, t):
    if t == "mirror":
        return g[:, ::-1]
    if t == "upside_down":
        return g[::-1, ::-1]
    return g


def _layout_small(small, n, position=None, count=None):
    s = n // 2
    o = (n - s) // 2
    block = small[o:o + s, o:o + s]
    out = np.zeros((n, n), np.uint8)
    if count == 2:
        out[o:o + s, 0:s] = block
        out[o:o + s, n - s:n] = block
    else:
        r, c = position if position is not None else (1, 1)
        y, x = r * (n - s) // 2, c * (n - s) // 2
        out[y:y + s, x:x + s] = block
    return out


def _plural(p):
    w = p.split(" ")
    last = w[-1]
    if re.search(r"(s|x|z|ch|sh)$", last):
        last += "es"
    elif re.search(r"[^aeiou]y$", last):
        last = last[:-1] + "ies"
    else:
        last += "s"
    return " ".join(w[:-1] + [last])


def _article(p):
    return ("an " if p[:1] in "aeiou" else "a ") + p


class Sampler:
    """Deterministic stream of (caption, exact target, style id, meta)."""

    def __init__(self, split: str = "train", seed: int = 0, tier: int = 0, mix: dict | None = None,
                 pixelgpt_n: int | None = None):
        self.split, self.seed, self.tier = split, seed, tier
        self.rng = np.random.default_rng([seed, _h(split) % (2 ** 31)])
        D = self.D = load_build(tier)
        self.N = int(D["imgs"].shape[1])
        self.pal = D["palette"]
        self.styles = D["styles"]
        self.concepts = D["concepts"]
        self.luts = recolor_luts(self.pal)
        self.bgc = bg_classes(self.pal)
        sat = saturated_mask(self.pal)
        imgs = D["imgs"]
        fg = imgs > 0
        self.colorable = (sat[imgs] & fg).sum((1, 2)) >= COLORABLE_FRAC * np.maximum(1, fg.sum((1, 2)))
        conc, split_arr = D["concept"], D["split"]
        # emoji bases per concept index
        self.by_concept = {}
        for i in np.nonzero(conc >= 0)[0]:
            self.by_concept.setdefault(int(conc[i]), []).append(int(i))
        self.cidx = sorted(self.by_concept)
        sp = D["splits"]
        self.held_color = {tuple(p) for p in sp["heldout_combo"]["colour"]}
        self.held_tf = {tuple(p) for p in sp["heldout_combo"]["transform"]}
        pg = np.nonzero(conc < 0)[0]
        self.pg = {s: pg[split_arr[pg] == k] for s, k in (("train", 0), ("val", 1), ("test", 2))}
        if pixelgpt_n is not None:
            self.pg["train"] = self.pg["train"][:pixelgpt_n]
        m = dict(DEFAULT_MIX[1 if len(self.pg["train"]) else 0] if mix is None else mix)
        if not len(self.pg["train"]):
            m["pixelgpt"] = 0.0
        tot = sum(m.values())
        self.mix_keys = [k for k in ("emoji_var", "pixelgpt", "emoji_plain") if m.get(k, 0) > 0]
        self.mix_p = np.array([m[k] for k in self.mix_keys]) / tot
        self.heldout_list = sorted(self.held_color) + sorted(self.held_tf)
        self.kw_test = [i for i in self.cidx if self.concepts[i]["keywords_test"]]
        # keywords usable as the caption noun: only fairly specific ones (shared by <= 2 concepts),
        # so "a food" / "a face" does not become an ambiguous training caption.
        df = {}
        for i in self.cidx:
            for k in self.concepts[i]["keywords_train"]:
                df[k] = df.get(k, 0) + 1
        self.noun_kws = {self.concepts[i]["id"]: [k for k in self.concepts[i]["keywords_train"]
                                                  if df[k] <= 2 and k not in COLOR_WORDS] for i in self.cidx}

    # -------------------------------------------------- helpers
    def _choice(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def _noun(self, c, colour=None, kw_p=0.0, test_kw=False):
        if test_kw:
            return self._choice(c["keywords_test"]), "keyword_test"
        kws = self.noun_kws[c["id"]]
        if kw_p and self.split in ("train", "train_sample") and kws and self.rng.random() < kw_p:
            return self._choice(kws), "keyword"
        name = c["name"]
        if colour is not None:  # "red apple" + blue -> "blue apple", not "blue red apple"
            stripped = " ".join(w for w in name.split(" ") if w not in COLOR_WORDS)
            name = stripped or name
        return name, "name"

    def _template(self, phrase_a, phrase_p):
        t = self._choice(TEMPLATES_HELDOUT if self.split == "heldout_phrase" else TEMPLATES_TRAIN)
        return t.replace("{a}", phrase_a).replace("{p}", phrase_p), t

    def _style_suffix(self, cap, style, p=0.3):
        if self.rng.random() < p:
            return f"{cap}, {self._choice(STYLE_WORDS[style])}", True
        return cap, False

    def _meta(self, **kw):
        m = dict(kind=None, concept=None, concept_idx=None, category=None, style=None, style_id=None,
                 styles_ok=None, colour=None, transform=None, size=None, position=None, count=None, bg=None,
                 template=None, base_idx=None, noun_source=None)
        m.update(kw)
        return m

    # -------------------------------------------------- one emoji sample
    def _emoji(self, variation: bool, forced=None, test_kw=False, style_p=0.3):
        """forced: optional ('colour'|'transform', concept_idx, value) for the heldout_combo split."""
        rng = self.rng
        ci = forced[1] if forced else self._choice(self.cidx if not test_kw else self.kw_test)
        c = self.concepts[ci]
        cid = c["id"]
        bases = self.by_concept[ci]
        colour = transform = position = count = bg = None
        small = False
        if forced:
            if forced[0] == "colour":
                colour = forced[2]
            else:
                transform = forced[2]
        elif variation:
            n_attr = 1 if rng.random() < 0.6 else 2
            kinds = ["colour", "transform", "size", "position", "count", "bg"]
            w = np.array([0.30, 0.20, 0.12, 0.14, 0.09, 0.15])
            chosen = []
            for _ in range(n_attr):
                ww = w * np.array([k not in chosen for k in kinds])
                if {"size", "position", "count"} & set(chosen):  # one layout attribute at most
                    ww *= np.array([k not in ("size", "position", "count") for k in kinds])
                chosen.append(kinds[int(rng.choice(len(kinds), p=ww / ww.sum()))])
            for k in chosen:
                if k == "colour" and any(self.colorable[b] for b in bases):
                    ok = [x for x in COLORS if (cid, x) not in self.held_color]
                    colour = self._choice(ok)
                elif k == "transform":
                    ok = [x for x in TRANSFORMS if (cid, x) not in self.held_tf]
                    transform = self._choice(ok)
                elif k == "size":
                    small = True
                elif k == "position":
                    small, position = True, self._choice(list(POSITIONS))
                elif k == "count" and len(c["name"].split(" ")) <= 2:  # plural of long names reads badly
                    small, count = True, 2
                elif k == "count":
                    small = True
                elif k == "bg":
                    bg = self._choice(list(BG_COLORS))
        if colour is not None:
            bases = [b for b in bases if self.colorable[b]]
        style_ok = sorted({int(self.D["style"][b]) for b in bases})
        b = self._choice(bases)
        sid = int(self.D["style"][b])
        # ---- target
        g = self.D["small"][b] if small else self.D["imgs"][b]
        if colour is not None:
            g = self.luts[colour][g]
        g = _transform(g, transform)
        if small:
            g = _layout_small(g, self.N, position, count)
        g = np.ascontiguousarray(g, dtype=np.uint8)
        if bg is not None:
            g = np.where(g == 0, self.bgc[bg], g).astype(np.uint8)
        # ---- caption
        noun, src = self._noun(c, colour, kw_p=(0.0 if forced else (0.15 if variation else 0.3)), test_kw=test_kw)
        pre = []
        if small and (position is None or rng.random() < 0.5) and count is None:
            pre.append(self._choice(SIZE_WORDS))
        tf_suffix = None
        if transform is not None:
            if rng.random() < 0.5:
                pre.append(self._choice(TRANSFORMS[transform][0]))
            else:
                tf_suffix = self._choice(TRANSFORMS[transform][1])
        if colour is not None:
            pre.append(self._choice(_WORDS_OF[colour]))
        p = " ".join(pre + [noun])
        if count == 2:
            pa = f"{self._choice(COUNT_WORDS[2])} {_plural(p)}"
            p = _plural(p)
        else:
            pa = _article(p)
        if tf_suffix:
            p, pa = f"{p} {tf_suffix}", f"{pa} {tf_suffix}"
        cap, tmpl = self._template(pa, p)
        if position is not None:
            cap += " " + self._choice(POSITIONS[position])
        if bg is not None:
            cap += " " + self._choice(BG_PHRASES).replace("{c}", self._choice(_WORDS_OF[bg]) if bg in _WORDS_OF else bg)
        cap, styled = self._style_suffix(cap, self.styles[sid], p=style_p)
        if len(words(cap)) > MAX_WORDS and styled:
            cap, styled = cap.rsplit(",", 1)[0], False
        kind = "emoji_var" if (variation or forced) else "emoji_plain"
        meta = self._meta(kind=kind, concept=cid, concept_idx=ci, category=c["group"], style=self.styles[sid],
                          style_id=sid, styles_ok=[sid] if styled else style_ok, colour=colour, transform=transform,
                          size="small" if small else None, position=POS_NAMES[position] if position else None,
                          count=count, bg=bg, template=tmpl, base_idx=b, noun_source=src)
        return cap.lower(), g, sid, meta

    def _pixelgpt(self, which="train", variation_p=0.3):
        rng = self.rng
        pool = self.pg[which]
        b = int(pool[int(rng.integers(len(pool)))])
        sid = int(self.D["style"][b])
        g = self.D["imgs"][b]
        cap = str(self.D["pg_caption"][b]).lower().strip()
        transform = bg = None
        n_words = len(words(cap))
        if rng.random() < variation_p and n_words + 4 <= MAX_WORDS:  # suffix must survive tokenizer truncation
            k = rng.integers(3)
            if k < 2:
                transform = list(TRANSFORMS)[int(k)]
                g = _transform(g, transform)
                cap += " " + self._choice(TRANSFORMS[transform][1])
            else:
                bg = self._choice(list(BG_COLORS))
                g = np.where(g == 0, self.bgc[bg], g)
                cap += " " + self._choice(BG_PHRASES).replace("{c}", self._choice(_WORDS_OF[bg]))
        if rng.random() < 0.5 and len(words(cap)) + 2 <= MAX_WORDS:
            cap = "pixel art " + cap
        meta = self._meta(kind="pixelgpt", category=str(self.D["pg_category"][b]), style="pixelgpt", style_id=sid,
                          styles_ok=[sid], transform=transform, bg=bg, base_idx=b, template="verbatim")
        return cap, np.ascontiguousarray(g, dtype=np.uint8), sid, meta

    # -------------------------------------------------- public
    def sample(self):
        s = self.split
        if s in ("train", "train_sample"):
            k = self.mix_keys[int(self.rng.choice(len(self.mix_keys), p=self.mix_p))]
            if k == "pixelgpt":
                return self._pixelgpt("train")
            return self._fit(lambda: self._emoji(variation=(k == "emoji_var")))
        if s == "heldout_combo":
            cid, val = self.heldout_list[int(self.rng.integers(len(self.heldout_list)))]
            ci = self.cid_to_idx[cid]
            return self._emoji(False, forced=("colour" if val in COLORS else "transform", ci, val), style_p=0.0)
        if s == "heldout_phrase":
            v = self.rng.random() < 0.5
            return self._fit(lambda: self._emoji(variation=v))
        if s == "unseen_words":
            return self._emoji(False, test_kw=True, style_p=0.0)
        if s == "underspecified":
            return self._emoji(False, style_p=0.0)
        if s in ("pixelgpt_test", "pixelgpt_val"):
            return self._pixelgpt(s.split("_")[1], variation_p=0.0)
        raise ValueError(f"unknown split {s!r}")

    def _fit(self, make, tries=20):
        """Resample until the caption fits the tokenizer's L words (so no attribute word is truncated)."""
        for _ in range(tries):
            out = make()
            if len(words(out[0])) <= MAX_WORDS:
                return out
        return out

    @property
    def cid_to_idx(self):
        if not hasattr(self, "_c2i"):
            self._c2i = {c["id"]: i for i, c in enumerate(self.concepts)}
        return self._c2i

    def batch(self, B: int):
        caps, tg, st, meta = [], np.zeros((B, self.N, self.N), np.uint8), np.zeros(B, np.int64), []
        for i in range(B):
            c, g, s, m = self.sample()
            caps.append(c)
            tg[i] = g
            st[i] = s
            meta.append(m)
        return caps, tg, st, meta


EVAL_SEEDS = {"train_sample": 101, "heldout_combo": 102, "heldout_phrase": 103, "unseen_words": 104,
              "pixelgpt_test": 105, "underspecified": 106, "pixelgpt_val": 107}


def eval_set(name: str, n: int = 300, tier: int = 0):
    """Fixed evaluation list [(caption, target [N,N], style_id, meta)] for a split name."""
    s = Sampler(name, seed=EVAL_SEEDS[name], tier=tier)
    if name.startswith("pixelgpt") and not len(s.pg[name.split("_")[1]]):
        return []
    if name == "unseen_words" and not s.kw_test:
        return []
    caps, tg, st, meta = s.batch(n)
    return [(caps[i], tg[i], int(st[i]), meta[i]) for i in range(n)]
