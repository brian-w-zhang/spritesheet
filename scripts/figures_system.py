"""F (systems figures): pipeline.svg, latency_cells.png, network_tabs.png, data_scaling.png.
Inputs: artifacts/metrics/{systems,parity_sheets,abl_*}.json and the model dir.

uv run python scripts/figures_system.py --model artifacts/model/v32
"""
import argparse, json, os, sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import reference as R, tokenizer as T  # noqa: E402
from make_figures import font, INK, MUTED, PAPER, ACCENT  # noqa: E402

M = os.path.join(ROOT, "artifacts", "metrics")
FIG = os.path.join(ROOT, "artifacts", "figures")
STAGES = [("Tokens", "tokenizer", lambda t: t.startswith("Tokens")),
          ("Encoder", "CLIP word table + attention", lambda t: t in ("X", "Q", "K", "VT", "Scores", "Softmax", "Ctx", "Resid", "MLP", "H")),
          ("Latent", "seeded style sampler", lambda t: t == "Latent"),
          ("Decoder", "dense + 3 conv blocks", lambda t: t in ("Dense", "FiLM") or t.startswith("Up_") or t.startswith("Conv_")),
          ("Logits → Pixels", "33-class argmax", lambda t: t in ("Logits", "Pixels", "Hex"))]


def stage_counts(sysm):
    out = []
    for name, desc, pred in STAGES:
        f = sum(v["formulas"] for t, v in sysm["per_tab"].items() if pred(t))
        out.append((name, desc, f))
    return out


def pipeline_svg(sysm, path):
    st = stage_counts(sysm)
    W, H, bw, gap = 1700, 360, 270, 60
    x0 = (W - (len(st) * bw + (len(st) - 1) * gap)) // 2
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{x0}" y="54" font-size="30" fill="#201e26">Prompt cell → sprite cells: one recalculation, {sysm["formula_cells"]:,} formulas</text>',
             f'<text x="{x0}" y="88" font-size="19" fill="#78748a">{sysm["total_cells"]:,} cells in {sysm["tabs"]} tabs ({100 * sysm["share_of_cell_limit"]:.1f}% of the 20M-cell Google Sheets limit)</text>']
    for i, (name, desc, f) in enumerate(st):
        x = x0 + i * (bw + gap)
        parts.append(f'<rect x="{x}" y="140" width="{bw}" height="150" rx="14" fill="#f6f4ef" stroke="#d9d5e0"/>')
        parts.append(f'<text x="{x + 20}" y="185" font-size="24" fill="#201e26">{name}</text>')
        parts.append(f'<text x="{x + 20}" y="215" font-size="16" fill="#78748a">{desc}</text>')
        parts.append(f'<text x="{x + 20}" y="265" font-size="26" fill="#e8705b">{f:,}</text>')
        parts.append(f'<text x="{x + 20 + 14 * len(f"{f:,}") + 8}" y="265" font-size="16" fill="#78748a">formulas</text>')
        if i < len(st) - 1:
            ax = x + bw + 8
            parts.append(f'<path d="M{ax} 215 L{ax + gap - 16} 215" stroke="#201e26" stroke-width="2.5"/>')
            parts.append(f'<path d="M{ax + gap - 24} 207 L{ax + gap - 14} 215 L{ax + gap - 24} 223" fill="none" stroke="#201e26" stroke-width="2.5"/>')
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))
    print("wrote", path)


def bars(path, title, sub, rows, unit, color=ACCENT):
    W, rh, top, left = 1700, 64, 130, 420
    H = top + rh * len(rows) + 70
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    d.text((40, 28), title, fill=INK, font=font(36))
    d.text((40, 76), sub, fill=MUTED, font=font(20))
    mx = max(v for _, v, _ in rows) or 1
    f = font(22)
    for i, (lab, v, txt) in enumerate(rows):
        y = top + i * rh
        d.text((40, y + 12), lab, fill=INK, font=f)
        w = int((W - left - 260) * v / mx)
        d.rectangle([left, y + 8, left + max(2, w), y + rh - 16], fill=color)
        d.text((left + w + 14, y + 12), txt, fill=MUTED, font=f)
    im.save(path)
    print("wrote", path)


def network_tabs(model_dir, path, prompt="a blue cat"):
    w, cfg = R.load(model_dir)
    vocab = T.load_vocab(os.path.join(ROOT, "data", "build", "vocab.txt"))
    out = R.forward(T.encode(prompt, vocab), 1, w, cfg)
    acts = out["acts"]
    tiles = []
    for a in acts:
        H, Wd, C = a.shape
        k = min(C, 8)
        for c in range(k):
            m = a[:, :, c]
            m = (m - m.min()) / (m.max() - m.min() + 1e-12)
            tiles.append(((m * 255).astype(np.uint8), H))
    cell = 120
    cols = 16
    rows = (len(tiles) + cols - 1) // cols
    im = Image.new("RGB", (40 + cols * (cell + 10), 130 + rows * (cell + 10) + 20), PAPER)
    d = ImageDraw.Draw(im)
    d.text((40, 28), f'Inside the sheet: decoder activation tabs for "{prompt}"', fill=INK, font=font(34))
    d.text((40, 76), "first 8 channels of Dense (4×4), Conv_1 (8×8), Conv_2 (16×16), Conv_3 (32×32); values from the reference, which the sheet matches to ≤1e-12",
           fill=MUTED, font=font(18))
    lut = np.stack([np.linspace(246, 232, 256), np.linspace(244, 112, 256), np.linspace(239, 91, 256)], 1).astype(np.uint8)
    for i, (m, H) in enumerate(tiles):
        r, c = divmod(i, cols)
        im.paste(Image.fromarray(lut[m]).resize((cell, cell), Image.NEAREST), (40 + c * (cell + 10), 130 + r * (cell + 10)))
    im.save(path)
    print("wrote", path)


def data_scaling(path):
    runs = [("PixelGPT 5k", "abl_pg5k"), ("PixelGPT 10k", "abl_pg10k"), ("PixelGPT 18k (all)", "abl_base")]
    rows = []
    for lab, key in runs:
        p = os.path.join(M, f"{key}.json")
        if os.path.exists(p):
            s = json.load(open(p))["sets"].get("pixelgpt_test", {})
            if "clip_score_rel" in s:
                rows.append((lab, s["clip_score_rel"], f'{s["clip_score_rel"]:.3f} of target CLIP score'))
    if rows:
        bars(path, "Does more pixel-art data help? (C5)", "PixelGPT test split; CLIP score relative to the ground-truth sprite; 8k-step runs", rows, "")
        json.dump([{"run": r[0], "clip_score_rel": r[1]} for r in rows], open(os.path.join(M, "data_scaling.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "model", "v32"))
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    sysm = json.load(open(os.path.join(M, "systems.json")))
    pipeline_svg(sysm, os.path.join(FIG, "pipeline.svg"))
    st = stage_counts(sysm)
    rows = [(n, f, f"{f:,} formulas") for n, _, f in st]
    sub = "formula cells per stage"
    if "sheets" in sysm and sysm["sheets"].get("same_model"):
        L = sysm["sheets"]["latency_s"]
        sub += f'; Google Sheets prompt→sprite latency median {L["median"]:.1f} s (n={sysm["sheets"]["cases"]})'
    bars(os.path.join(FIG, "latency_cells.png"), "Where the cells (and the recalculation time) go", sub, rows, "")
    network_tabs(a.model, os.path.join(FIG, "network_tabs.png"))
    data_scaling(os.path.join(FIG, "data_scaling.png"))


if __name__ == "__main__":
    main()
