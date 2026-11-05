"""Assets for the MP2 deck (docs/slides/deck/): sprite grids from our own models (never PixelGPT images) and the
formula counts of the deployed v3 workbook.

uv run python scripts/slide_assets.py
→ docs/slides/deck/sprites.js (window.SPRITES = {...}), artifacts/metrics/systems_v3p.json
Sprites for the final model use the same sampler as the sheet (codes equal the spec; tests/test_patchgpt.py), and the
8 prompt/seed pairs are the ones read back from Google Sheets in parity_sheets_v3p.json.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np  # noqa: E402

from eval_v3 import Model  # noqa: E402
from spritesheet.pixcell import gates as G  # noqa: E402
from spritesheet.pixcell.workbook import is_formula  # noqa: E402

CASES = [(c["prompt"], c["seed"]) for c in json.load(open(os.path.join(ROOT, "artifacts", "metrics", "parity_sheets_v3p.json")))["cases"]]
MODELS = {"final": "v3p_a@g=0.75:t=0.8:u=v3p_u48", "unguided": "v3p_a@t=0.8", "v1xl": "v32"}
EXTRA = [("a frog", s) for s in range(1, 9)] + [("a fox", s) for s in range(1, 9)]


def hexes(rgb):
    return ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in rgb]


def main():
    out = {"cases": [{"prompt": p, "seed": s} for p, s in CASES], "models": {}}
    for key, spec in MODELS.items():
        m = Model(spec)
        caps, seeds = [p for p, _ in CASES], [s for _, s in CASES]
        if key == "final":
            caps += [p for p, _ in EXTRA]
            seeds += [s for _, s in EXTRA]
        g, _ = m.generate(caps, seeds)
        out["models"][key] = {"spec": spec, "palette": hexes(m.rgb), "grids": [x.astype(int).tolist() for x in g]}
        print(key, len(g), g[0].shape, flush=True)
    out["extra"] = [{"prompt": p, "seed": s} for p, s in EXTRA]
    os.makedirs(os.path.join(ROOT, "docs", "slides", "deck"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "slides", "deck", "sprites.js"), "w") as f:
        f.write("window.SPRITES = " + json.dumps(out, separators=(",", ":")) + ";\n")

    # formula counts of the deployed workbook
    md = os.path.join(ROOT, "artifacts", "model", "v3p_a_g1_sheet")
    w, cfg, vocab, pal, src = G.load_model(md)
    wb = G.compile_any(w, cfg, vocab, pal, render=True)
    per = {n: sum(1 for v in s.values() if is_formula(v)) for n, s in wb.sheets.items()}
    cells = sum(len(s) for s in wb.sheets.values())
    sysm = {"model_dir": "artifacts/model/v3p_a_g1_sheet", "formulas": sum(per.values()), "cells": cells,
            "tabs": len(wb.sheets), "formulas_guidance_chain": sum(v for k, v in per.items() if k.startswith("U")),
            "formulas_score": per.get("Score"), "per_tab_formulas": per,
            "params_main": json.load(open(os.path.join(ROOT, "artifacts", "model", "v3p_a", "config.json")))["params"],
            "params_uncond": json.load(open(os.path.join(ROOT, "artifacts", "model", "v3p_u48", "config.json")))["params"]}
    json.dump(sysm, open(os.path.join(ROOT, "artifacts", "metrics", "systems_v3p.json"), "w"), indent=1)
    print({k: v for k, v in sysm.items() if k != "per_tab_formulas"})


if __name__ == "__main__":
    main()
