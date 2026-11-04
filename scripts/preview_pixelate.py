"""Setup-pass prototype: pixelate the same concepts from 4 emoji sets at 32 and 48 px.
Writes artifacts/figures/preview_pixelate.png. Not the final pipeline (the overnight agent builds
src/spritesheet/pixelate.py with a fixed corpus palette); this is only to judge the look.
Run: uv run python scripts/preview_pixelate.py
"""
import glob, json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CONCEPTS = [("1f34e", "red apple"), ("1f431", "cat face"), ("1f415", "dog"), ("1f680", "rocket"),
            ("1f355", "pizza"), ("1f333", "tree"), ("1f697", "car"), ("1f47b", "ghost")]


def fluent_index():
    idx = {}
    for meta in glob.glob(os.path.join(RAW, "fluentui-emoji", "assets", "*", "metadata.json")):
        m = json.load(open(meta))
        u = m.get("unicode", "").lower()
        pngs = glob.glob(os.path.join(os.path.dirname(meta), "3D", "*.png"))
        if u and pngs:
            idx[u] = pngs[0]
    return idx


FL = fluent_index()


def sources(cp):
    return {
        "noto": os.path.join(RAW, "noto-emoji", "2D", "png", "128", f"emoji_u{cp}.png"),
        "twemoji": os.path.join(RAW, "twemoji", "assets", "72x72", f"{cp}.png"),
        "fluent-3d": FL.get(cp, ""),
        "openmoji": os.path.join(RAW, "openmoji", "color", "72x72", f"{cp.upper()}.png"),
    }


def pixelate(path, n, colors=24, outline=True):
    im = Image.open(path).convert("RGBA")
    bbox = im.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox()
    im = im.crop(bbox)
    inner = n - 2
    s = inner / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BOX)
    canvas = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    canvas.paste(im, ((n - im.width) // 2, (n - im.height) // 2))
    a = canvas.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
    rgb = Image.new("RGB", (n, n), (255, 255, 255))
    rgb.paste(canvas.convert("RGB"), mask=a)
    q = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
    out = Image.new("RGB", (n, n), (246, 244, 239))
    out.paste(q, mask=a)
    if outline:
        px, am = out.load(), a.load()
        for y in range(n):
            for x in range(n):
                if am[x, y] == 0 and any(0 <= x + dx < n and 0 <= y + dy < n and am[x + dx, y + dy]
                                         for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    px[x, y] = (40, 36, 48)
    return out


def main():
    cell, pad = 160, 12
    sets = list(sources("1f34e").keys())
    cols = len(CONCEPTS)
    rows = len(sets) * 2
    W, H = 150 + cols * (cell + pad), 40 + rows * (cell + pad)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for c, (_, name) in enumerate(CONCEPTS):
        d.text((150 + c * (cell + pad), 12), name, fill=(30, 30, 30), font=font)
    r = 0
    for n in (32, 48):
        for sname in sets:
            d.text((8, 40 + r * (cell + pad) + cell // 2), f"{sname} {n}px", fill=(30, 30, 30), font=font)
            for c, (cp, _) in enumerate(CONCEPTS):
                p = sources(cp)[sname]
                if p and os.path.exists(p):
                    img = pixelate(p, n).resize((cell, cell), Image.NEAREST)
                    sheet.paste(img, (150 + c * (cell + pad), 40 + r * (cell + pad)))
                else:
                    d.text((150 + c * (cell + pad) + 50, 40 + r * (cell + pad) + 70), "missing", fill=(200, 0, 0))
            r += 1
    os.makedirs(os.path.join(ROOT, "artifacts", "figures"), exist_ok=True)
    out = os.path.join(ROOT, "artifacts", "figures", "preview_pixelate.png")
    sheet.save(out)
    print(out, sheet.size, "fluent concepts indexed:", len(FL))


if __name__ == "__main__":
    main()
