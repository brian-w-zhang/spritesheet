#!/usr/bin/env python3
"""Spike R1 for v2 (masked discrete diffusion) on the shared sheet sheet_ids.diffusion: upload cost and
prompt->pixels latency of a RANDOM-weight full-size v2 workbook (mdm.MDM_CFG) for each step count T.

uv run python scripts/spike_sheets_v2.py [--steps 4,6] [--width full|half|quarter] [--prompts 3] [--append]
The last T in --steps stays pushed (verify_sheets.py --model-type mdm --random-cfg '<same json>' --skip-push reuses it).
width: full = MDM_CFG (E/C1/C2/C3 = 24/24/48/96), half = 12/12/24/48, quarter = 8/8/16/32.
Writes artifacts/metrics/spikes_v2.json after every config. Never prints the key or the spreadsheet id.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from spritesheet import mdm as D, tokenizer as T  # noqa: E402
from spritesheet.pixcell import emit_mdm, gates as G, sheets as SH, trace as TR  # noqa: E402

PROMPTS = ["a red apple", "a tiny blue rocket facing left", "green cat with a hat"]
WIDTHS = {"full": {}, "half": dict(E=12, C1=12, C2=24, C3=48), "quarter": dict(E=8, C1=8, C2=16, C3=32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="4,6")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--width", default="full", choices=sorted(WIDTHS))
    ap.add_argument("--append", action="store_true", help="keep earlier runs in the output file")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/metrics/spikes_v2.json"))
    a = ap.parse_args()
    if not SH.available():
        sys.exit("NOT RUN: no service-account key / config in ~/.config/spritesheet")
    report = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sheet": "diffusion", "weights": "random-init MDMNet",
              "runs": []}
    if a.append and os.path.exists(a.out):
        report["runs"] = json.load(open(a.out)).get("runs", [])
    for TT in [int(x) for x in a.steps.split(",")]:
        cfg = dict(D.MDM_CFG, T=TT, **WIDTHS[a.width])
        w, cfg, vocab, pal = G.random_mdm_full(cfg)
        t0 = time.time()
        wb = emit_mdm.compile_mdm(w, cfg, vocab, pal, render=True)
        st = wb.stats()
        entry = {"T": TT, "width": a.width, "config": cfg, "cells": st["cells"], "formulas": st["formulas"],
                 "formula_chars": st["formula_chars"], "tabs": st["sheets"], "compile_s": time.time() - t0,
                 "model_hash": wb.meta["model_hash"], "errors": []}
        print(f"[T={TT}] cells={st['cells']} formulas={st['formulas']} tabs={st['sheets']} pushing ...", flush=True)
        cl = SH.Client("diffusion")
        try:
            up = SH.push(wb, cl)
            entry["upload"] = up
            print(f"[T={TT}] upload {up['upload_s']:.1f}s, {up['requests']} requests, "
                  f"{up['bytes_sent'] / 1e6:.1f} MB", flush=True)
            vi = T.index_vocab(vocab)
            lat = []
            for k, p in enumerate(PROMPTS[: a.prompts]):
                seed = k + 1
                r = SH.set_prompt_and_read(p, seed, cl, wb, timeout=a.timeout)
                f = TR.trace_mdm(p, seed, w, cfg, vi)["forward"]
                same = bool(np.array_equal(np.array(r["classes"]), f["classes"]))
                lat.append({"prompt": p, "seed": seed, "elapsed_s": r["elapsed_s"], "write_s": r["write_s"],
                            "first_consistent_s": r["first_consistent_s"], "reads": r["reads"],
                            "pixels_match_reference": same, "style_sheet": r["style"], "style_ref": f["style"],
                            "committed_per_step": f["committed"], "min_top2_margin": f["margins"],
                            "min_conf_tau_gap": f["tau_gaps"]})
                print(f"[T={TT}] prompt {k + 1}: stable {r['elapsed_s']:.1f}s (consistent at "
                      f"{r['first_consistent_s']:.1f}s) match={same}", flush=True)
            entry["latency"] = lat
            entry["latency_summary"] = {
                "first_consistent_median_s": float(np.median([x["first_consistent_s"] for x in lat])),
                "stable_median_s": float(np.median([x["elapsed_s"] for x in lat])),
                "stable_max_s": float(np.max([x["elapsed_s"] for x in lat]))}
        except Exception as ex:  # noqa: BLE001
            entry["errors"].append(f"{type(ex).__name__}: {ex}")
            print(f"[T={TT}] ERROR {type(ex).__name__}: {ex}", flush=True)
        entry["requests_total"] = cl.requests
        entry["retries_total"] = cl.retries
        report["runs"].append(entry)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1, default=G.np_default)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
