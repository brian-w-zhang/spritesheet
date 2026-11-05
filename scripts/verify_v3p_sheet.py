"""A4 for the deployed v3-P sheet: set the live guidance cell (Prompt!B4), then for each prompt/seed write the inputs,
read Pixels back from Google Sheets and compare with the spec (reference_p.forward at that guidance).

uv run python scripts/verify_v3p_sheet.py --model artifacts/model/v3p_a_g1_sheet --guidance 0.75
→ artifacts/metrics/parity_sheets_v3p.json, artifacts/figures/v3_sheet_readback.png (pixels as read from Sheets)
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from spritesheet import reference_p as RP, tokenizer as T  # noqa: E402
from spritesheet.pixcell import gates as G, sheets  # noqa: E402

CASES = [("a red potion", 1), ("a frog", 3), ("a fox", 2), ("a mushroom", 1), ("a golden crown", 4), ("an owl", 6),
         ("a treasure chest", 1), ("a blue slime", 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sheet", default="v3")
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--n", type=int, default=len(CASES))
    a = ap.parse_args()
    if not sheets.available():
        print("NOT RUN: no service-account key/config")
        sys.exit(2)
    w, cfg, vocab, pal, src = G.load_model(a.model)
    wb = G.compile_any(w, cfg, vocab, pal, render=True)
    cl = sheets.Client(a.sheet)
    g = cfg.get("guidance", 0.0) if a.guidance is None else a.guidance
    if a.guidance is not None:
        cl.values_batch_update([{"range": "'Prompt'!B4", "values": [[float(g)]]}], "RAW")
    cfg = dict(cfg, guidance=g)
    vix = T.index_vocab(vocab)
    rgb = np.concatenate([np.array([[216, 212, 221]], np.uint8),
                          np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])
    res, tiles = [], []
    for prompt, seed in CASES[: a.n]:
        got = sheets.set_prompt_and_read(prompt, seed, cl, wb, timeout=1800)
        ref = RP.forward(T.encode(prompt, vix), seed, w, cfg, keep=False)["pixels"]
        px = np.asarray(got["classes"])
        ok = bool((px == ref).all())
        res.append({"prompt": prompt, "seed": seed, "match": ok, "first_consistent_s": round(got["first_consistent_s"], 1),
                    "stable_s": round(got["elapsed_s"], 1)})
        print(res[-1], flush=True)
        tiles.append(np.pad(rgb[px.astype(int)], ((1, 1), (1, 1), (0, 0)), constant_values=255))
    out = {"model": src.get("source"), "sheet": a.sheet, "guidance": g, "tau": cfg.get("tau"), "cases": res,
           "all_match": all(r["match"] for r in res), "n": len(res), "time": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "metrics", "parity_sheets_v3p.json"), "w"), indent=1)
    img = np.concatenate(tiles, 1)
    Image.fromarray(img).resize((img.shape[1] * 6, img.shape[0] * 6), Image.NEAREST).save(
        os.path.join(ROOT, "artifacts", "figures", "v3_sheet_readback.png"))
    print("all match:", out["all_match"])


if __name__ == "__main__":
    main()
