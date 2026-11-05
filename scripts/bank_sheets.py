"""Contact sheets of the generation bank for visual review (LOCAL ONLY: rows include real PixelGPT sprites).

  uv run python scripts/bank_sheets.py --rows 15
Writes artifacts/bank/sheets/sheet_NNN_local.png. Each row is one prompt: its number and name, then
v3 seeds 1-8, v1 seeds 1-8, then 4 real training sprites (reference only). artifacts/bank/ is gitignored.
"""
import argparse, os, re, sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_figures import font  # noqa: E402

BANK = os.path.join(ROOT, "artifacts", "bank")
CELL, GAP, LABEL = 72, 6, 210


def hexrgb(pal):
    return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=15)
    a = ap.parse_args()
    import json
    html = open(os.path.join(BANK, "index_local.html")).read()
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    import base64
    dec = lambda s: np.frombuffer(base64.b64decode(s), np.uint8)
    S = len(data["seeds"])
    mods = {}
    for k, m in data["models"].items():
        mods[k] = (dec(m["sprites"]).reshape(-1, m["n"], m["n"]), hexrgb(m["palette"]))
    rpal = hexrgb(data["real"]["palette"])
    out = os.path.join(BANK, "sheets")
    os.makedirs(out, exist_ok=True)
    P = len(data["prompts"])
    f, fs = font(17), font(13)
    ncols = 2 * S + 4
    W = LABEL + ncols * (CELL + GAP) + 3 * 18
    for s0 in range(0, P, a.rows):
        idx = range(s0, min(P, s0 + a.rows))
        H = 34 + len(idx) * (CELL + 12)
        im = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(im)
        x = LABEL
        for lab, n in (("v3 seeds 1-8", S), ("v1 seeds 1-8", S), ("real (reference)", 4)):
            d.text((x, 8), lab, fill=(60, 60, 60), font=fs)
            x += n * (CELL + GAP) + 18
        for r, i in enumerate(idx):
            y = 34 + r * (CELL + 12)
            p = data["prompts"][i]
            d.text((6, y + 18), f"#{i}", fill=(130, 130, 130), font=fs)
            d.text((52, y + 16), p["prompt"][:20], fill=(0, 0, 0), font=f)
            x = LABEL
            for k in ("v3", "v1"):
                g, pal = mods[k]
                for j in range(S):
                    t = Image.fromarray(pal[g[i * S + j]]).resize((CELL, CELL), Image.NEAREST)
                    im.paste(t, (x, y))
                    d.text((x + 2, y + CELL - 1), str(j + 1), fill=(150, 150, 150), font=font(10))
                    x += CELL + GAP
                x += 18
            rs = data["real"]["sprites"][i]
            if rs:
                rr = dec(rs).reshape(-1, 24, 24)
                for t in rr:
                    im.paste(Image.fromarray(rpal[t]).resize((CELL, CELL), Image.NEAREST), (x, y))
                    x += CELL + GAP
        im.save(os.path.join(out, f"sheet_{s0 // a.rows:03d}_local.png"))
    print(f"{(P + a.rows - 1) // a.rows} sheets in {out}")


if __name__ == "__main__":
    main()
