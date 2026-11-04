"""Normalisation to N x N class-id grids (docs/03 "Normalising different native sizes").

* Emoji / SVG: crop to the alpha bbox (alpha > 16), area-downsample (Image.BOX, premultiplied by
  Pillow) to fit (size - 2), centre, alpha threshold 128, snap to the palette. No outline is added.
* PixelGPT 24x24: padded 1:1 into the N canvas (never resampled).
* Small variant: the same render at size round(N * 0.5), centred on the N canvas (emoji only).
* Palette: K colours fitted by k-means in CIELAB (D65) over foreground pixels; snapping is nearest
  colour by Euclidean distance in CIELAB (CIE76). Class 0 = background / transparent,
  class i (1..K) = colors[i-1].
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

ALPHA_BBOX = 16
ALPHA_ON = 128


# ---------------------------------------------------------------- colour space
def srgb_to_lab(rgb) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * np.array([0.95047, 1.0, 1.08883])
    minv = np.array([[3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]])
    c = np.clip(xyz @ minv.T, 0, 1)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- palette
class Palette:
    def __init__(self, colors_hex: list[str]):
        self.hex = list(colors_hex)
        self.rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in self.hex], dtype=np.uint8)
        self.lab = srgb_to_lab(self.rgb)
        self.K = len(self.hex)
        assert len(set(self.hex)) == self.K, "duplicate palette colours break snap idempotence"

    @classmethod
    def load(cls, path: str) -> "Palette":
        return cls(json.load(open(path))["colors"])

    def save(self, path: str, n: int):
        json.dump({"n": n, "colors": self.hex, "space": "CIELAB-D65", "snap": "nearest CIE76",
                   "class0": "background/transparent"}, open(path, "w"), indent=1)

    def nearest(self, rgb) -> np.ndarray:
        """Nearest palette index (0-based) for each RGB colour in [..., 3]."""
        lab = srgb_to_lab(rgb)
        d = ((lab[..., None, :] - self.lab) ** 2).sum(-1)
        return d.argmin(-1)

    def snap(self, rgba: np.ndarray) -> np.ndarray:
        """RGBA uint8 [H, W, 4] -> class ids uint8 [H, W] (0 = transparent)."""
        rgba = np.asarray(rgba)
        ids = np.zeros(rgba.shape[:2], dtype=np.uint8)
        on = rgba[..., 3] >= ALPHA_ON
        if on.any():
            ids[on] = self.nearest(rgba[..., :3][on]).astype(np.uint8) + 1
        return ids

    def to_rgba(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        out = np.zeros(ids.shape + (4,), dtype=np.uint8)
        on = ids > 0
        out[on, :3] = self.rgb[ids[on] - 1]
        out[on, 3] = 255
        return out

    def to_rgb(self, ids: np.ndarray, bg=(246, 244, 239)) -> np.ndarray:
        out = np.empty(np.shape(ids) + (3,), dtype=np.uint8)
        out[:] = bg
        on = np.asarray(ids) > 0
        out[on] = self.rgb[np.asarray(ids)[on] - 1]
        return out


def kmeans_palette(rgb_pixels: np.ndarray, k: int = 32, seed: int = 0, iters: int = 60,
                   max_points: int = 200_000) -> list[str]:
    """Plain k-means (k-means++ init, Lloyd iterations) in CIELAB. Deterministic for a seed.
    Returns K distinct '#rrggbb' strings sorted by lightness."""
    rng = np.random.default_rng(seed)
    px = np.asarray(rgb_pixels, dtype=np.uint8).reshape(-1, 3)
    if len(px) > max_points:
        px = px[rng.choice(len(px), max_points, replace=False)]
    x = srgb_to_lab(px)
    cent = [x[rng.integers(len(x))]]
    d2 = ((x - cent[0]) ** 2).sum(-1)
    for _ in range(1, k):
        cent.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - cent[-1]) ** 2).sum(-1))
    c = np.array(cent)
    for _ in range(iters):
        a = ((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]).argmin(1)
        newc = np.array([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(newc, c, atol=1e-6):
            break
        c = newc
    rgb = lab_to_srgb(c)
    order = np.argsort(c[:, 0], kind="stable")
    hexes = []
    for i in order:
        r, g, b = (int(v) for v in rgb[i])
        h = f"#{r:02x}{g:02x}{b:02x}"
        while h in hexes:  # vanishingly rare rounding collision: nudge blue channel
            b = (b + 1) % 256
            h = f"#{r:02x}{g:02x}{b:02x}"
        hexes.append(h)
    return hexes


# ---------------------------------------------------------------- rendering
def render_emoji(img: Image.Image, n: int, size: int | None = None) -> np.ndarray | None:
    """High-res RGBA emoji -> RGBA uint8 [n, n, 4] with binary alpha, content fitted to (size - 2)
    and centred. `size` defaults to n; a smaller size gives the 'small' variant (centred on n)."""
    size = n if size is None else size
    im = img.convert("RGBA")
    bbox = im.getchannel("A").point(lambda a: 255 if a > ALPHA_BBOX else 0).getbbox()
    if bbox is None:
        return None
    im = im.crop(bbox)
    inner = size - 2
    s = inner / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BOX)
    canvas = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    canvas.paste(im, ((n - im.width) // 2, (n - im.height) // 2))
    a = np.array(canvas, dtype=np.uint8)
    on = a[..., 3] >= ALPHA_ON
    a[..., 3] = np.where(on, 255, 0)
    a[~on, :3] = 0
    return a


def small_size(n: int) -> int:
    return round(n * 0.5)


def pad_pixel_art(rgba: np.ndarray, n: int) -> np.ndarray:
    """Pad native pixel art 1:1 (no resampling) into the centre of a transparent n x n canvas."""
    h, w = rgba.shape[:2]
    assert h <= n and w <= n, "pixel art larger than the canvas; integer-downscale is not allowed"
    out = np.zeros((n, n, 4), dtype=np.uint8)
    y, x = (n - h) // 2, (n - w) // 2
    core = rgba.copy()
    core[core[..., 3] < ALPHA_ON] = 0
    core[..., 3] = np.where(core[..., 3] >= ALPHA_ON, 255, 0)
    out[y:y + h, x:x + w] = core
    return out
