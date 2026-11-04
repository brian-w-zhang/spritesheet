#!/usr/bin/env python3
"""Gate A3: end-to-end parity of the compiled workbook vs reference.forward on HyperFormula and LibreOffice.

uv run python scripts/verify_e2e.py --model artifacts/model/v32 --prompts 200 --seeds 2 [--engines hyperformula,libreoffice]
Works with random weights when the model dir has no weights.npz. Writes artifacts/metrics/parity_local.json.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet.pixcell import gates as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts/model/v32"))
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--engines", default="hyperformula,libreoffice")
    ap.add_argument("--lo-max", type=int, default=None, help="cap LibreOffice cases (each is a full recalc)")
    ap.add_argument("--all-tabs", action="store_true", help="read and compare every tab, not only the key ones")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/metrics/parity_local.json"))
    a = ap.parse_args()
    t0 = time.time()
    res = G.gate_a3(a.model, a.prompts, a.seeds, tuple(a.engines.split(",")), a.lo_max, a.all_tabs)
    res["wall_s"] = time.time() - t0
    res["when"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=G.np_default)
    st = res["stats"]
    print(f"model={res['model']['source']} cells={st['cells']} formulas={st['formulas']} "
          f"formula_chars={st['formula_chars']} lint_violations={res['lint_violations']}")
    for eng, e in res["engines"].items():
        print(f"A3 {eng}: cases={e['cases']} max_rel={e['max_rel']:.3g} fail={e['n_fail']} errors={e['error_cells']} "
              f"stale={e['stale_cells']} style_mismatch={e['style_mismatch']} pixel_mismatch={e['pixels_mismatch']} "
              f"wall={e['timing']['wall_s']:.1f}s")
    print(f"min_margin={res['min_margin']:.3g} styles={res['styles']}  -> {'PASS' if res['pass'] else 'FAIL'}  ({a.out})")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
