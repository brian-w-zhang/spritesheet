"""Gallery from REAL Google Sheets output: types each prompt into the live sheet, waits for the recalculation,
reads the Pixels tab that Google computed, and renders it (plus the local reference for comparison).

uv run python scripts/sheets_gallery.py --sheet v3 --model artifacts/model/v3_var16 --tag v3_var16
→ artifacts/figures/sheets_gallery_<tag>.png, artifacts/metrics/sheets_gallery_<tag>.json
"""
import argparse, json, os, sys, time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import reference as R, tokenizer as T  # noqa: E402
from spritesheet.pixcell import gates as G, sheets as SH  # noqa: E402
from make_figures import font, INK, MUTED, PAPER  # noqa: E402

PROMPTS = ["a frog", "a red frog", "a fox", "a cat", "a ghost", "a dragon", "a wizard", "a knight",
           "a sword", "a treasure chest", "a red potion", "a mushroom", "a tree", "a castle", "a golden crown", "a lantern"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--prompts", default="|".join(PROMPTS))
    a = ap.parse_args()
    w, cfg, vocab, pal, _ = G.load_model(a.model)
    wb = G.compile_any(w, cfg, vocab, pal, render=True)
    cl = SH.Client(a.sheet)
    got = SH.read_ranges(cl, ["'About'!B2"])[0]
    if not got or got[0][0] != wb.meta["model_hash"]:
        sys.exit("the sheet holds a different model (About!B2 hash); push it first")
    vi = T.index_vocab(vocab)
    rgb = np.array([[0xD8, 0xD4, 0xDD]] + [[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        for p in a.prompts.split("|"):
            r = SH.set_prompt_and_read(p, s, cl, wb, timeout=900)
            ref = R.forward(T.encode(p, vi), s, w, cfg)["classes"]
            g = np.asarray(r["classes"], dtype=np.uint8)
            rows.append({"prompt": p, "seed": s, "sheet_classes": g.tolist(), "matches_reference": bool((g == ref).all()),
                         "seconds": r["elapsed_s"], "variant_or_style": r.get("style")})
            print(p, s, "match" if rows[-1]["matches_reference"] else "MISMATCH", round(r["elapsed_s"], 1), flush=True)
    json.dump({"sheet": a.sheet, "model": a.model, "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows},
              open(os.path.join(ROOT, "artifacts", "metrics", f"sheets_gallery_{a.tag}.json"), "w"))
    cs, pad, cols = 120, 10, 8
    n = len(rows)
    H = 60 + ((n + cols - 1) // cols) * (cs + 40)
    im = Image.new("RGB", (pad + cols * (cs + pad), H), PAPER)
    d = ImageDraw.Draw(im)
    d.text((pad, 12), f"{a.tag}: read back from the live Google Sheet ({a.sheet})", fill=INK, font=font(22))
    for i, r in enumerate(rows):
        y, x = 50 + (i // cols) * (cs + 40), pad + (i % cols) * (cs + pad)
        im.paste(Image.fromarray(rgb[np.array(r["sheet_classes"])]).resize((cs, cs), Image.NEAREST), (x, y))
        d.text((x, y + cs + 4), f"{r['prompt'][:18]} · s{r['seed']}", fill=MUTED, font=font(12))
    im.save(os.path.join(ROOT, "artifacts", "figures", f"sheets_gallery_{a.tag}.png"))
    print("matches:", sum(r["matches_reference"] for r in rows), "/", n)


if __name__ == "__main__":
    main()
