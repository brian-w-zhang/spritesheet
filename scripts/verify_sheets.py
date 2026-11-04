#!/usr/bin/env python3
"""Gate A4: end-to-end parity on real Google Sheets vs reference.forward (any model dir; random weights if absent).

uv run python scripts/verify_sheets.py --model artifacts/model/v32 --prompts 20 [--skip-push]
Pushes the compiled workbook (render=True) to sheet_ids.v32 unless --skip-push (then About!B2 must hold the same
model hash). For each prompt: write B2/B3, wait for Pixels, read Logits/Latent, compare. Writes
artifacts/metrics/parity_sheets.json. Exit 2 = NOT RUN (no credentials).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from spritesheet import reference as R, tokenizer as T  # noqa: E402
from spritesheet.pixcell import emit, gates as G, sheets as SH  # noqa: E402
from spritesheet.pixcell.workbook import rng  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts/model/v32"))
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/metrics/parity_sheets.json"))
    a = ap.parse_args()
    if not SH.available():
        print("A4 NOT RUN: no service-account key/config in ~/.config/spritesheet")
        sys.exit(2)
    w, cfg, vocab, pal, src = G.load_model(a.model)
    wb = emit.compile_model(w, cfg, vocab, pal, render=True)
    st = wb.stats()
    cl = SH.Client()
    res = {"gate": "A4", "model": src, "model_hash": wb.meta["model_hash"], "config": cfg,
           "cells": st["cells"], "formulas": st["formulas"], "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if a.skip_push:
        got = SH.read_ranges(cl, ["'About'!B2"])[0]
        if not got or got[0][0] != wb.meta["model_hash"]:
            sys.exit("sheet holds a different model (About!B2 hash mismatch); run without --skip-push")
        res["upload"] = None
    else:
        res["upload"] = SH.push(wb, cl)
    vi = T.index_vocab(vocab)
    N, K = wb.meta["N"], cfg["K"]
    cases = G.e2e_cases(a.prompts, vocab, a.seeds, rng_seed=1)
    rows, worst, n_px_bad, n_style_bad, lat = [], 0.0, 0, 0, []
    for p, s in cases:
        r = SH.set_prompt_and_read(p, s, cl, wb)
        ref = R.forward(T.encode(p, vi), s, w, cfg)
        lg, latv = SH.read_ranges(cl, [f"'Logits'!{rng(1, 1, N, N * (K + 1))}", "'Latent'!A1:" + "ZZ1"])
        L = np.array(lg, dtype=float).reshape(N, N, K + 1)
        rel = float((np.abs(L - ref["logits"]) / np.maximum(1.0, np.abs(ref["logits"]))).max())
        c_rel = float((np.abs(np.array(latv[0][: cfg["d"]], float) - ref["c"]) / np.maximum(1, np.abs(ref["c"]))).max())
        px_ok = bool(np.array_equal(np.array(r["classes"]), ref["classes"]))
        st_ok = r["style"] == ref["style"]
        worst = max(worst, rel, c_rel)
        n_px_bad += not px_ok
        n_style_bad += not st_ok
        lat.append(r["first_consistent_s"])
        rows.append({"prompt": p, "seed": s, "pixels_match": px_ok, "style_match": st_ok, "logits_max_rel": rel,
                     "c_max_rel": c_rel, "latency_s": r["first_consistent_s"], "stable_s": r["elapsed_s"],
                     "min_margin": ref["min_margin"]})
        print(f"{p[:40]!r:44} seed={s:<8} px={px_ok} style={st_ok} rel={rel:.2g} t={r['first_consistent_s']:.1f}s",
              flush=True)
    res.update({"cases": len(cases), "pixels_mismatch": n_px_bad, "style_mismatch": n_style_bad, "max_rel": worst,
                "latency_s": {"median": float(np.median(lat)), "max": float(np.max(lat)), "min": float(np.min(lat))},
                "min_margin": float(min(x["min_margin"] for x in rows)), "rows": rows,
                "requests": cl.requests, "retries": cl.retries})
    res["pass"] = n_px_bad == 0 and n_style_bad == 0 and worst <= 1e-12
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=G.np_default)
    print(f"A4 sheets: cases={len(cases)} max_rel={worst:.3g} pixel_mismatch={n_px_bad} style_mismatch={n_style_bad} "
          f"latency median={res['latency_s']['median']:.1f}s -> {'PASS' if res['pass'] else 'FAIL'} ({a.out})")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
