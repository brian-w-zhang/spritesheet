#!/usr/bin/env python3
"""Gate A4: end-to-end parity on real Google Sheets vs the spec (any model dir; random weights if absent).

uv run python scripts/verify_sheets.py --model artifacts/model/v32 --prompts 20 [--skip-push]
uv run python scripts/verify_sheets.py --model artifacts/model/v2_mdm_a --sheet diffusion --prompts 8
v1 spec = reference.forward; v2 (config.json "model": "mdm", or --model-type mdm) spec = mdm.ref_generate.
Pushes the compiled workbook (render=True) to sheet_ids[--sheet] (default: v32 for v1, diffusion for v2) unless
--skip-push (then About!B2 must hold the same model hash). For each prompt: write B2/B3, wait for Pixels, read the
logits (v2: every step's logits, conf and canvas) + Latent, compare. Writes artifacts/metrics/parity_sheets.json
(v1) or parity_sheets_mdm.json (v2). Exit 2 = NOT RUN (no credentials).
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
from spritesheet.pixcell import emit_mdm, gates as G, sheets as SH, trace as TR  # noqa: E402
from spritesheet.pixcell.workbook import rng  # noqa: E402


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((np.abs(a - b) / np.maximum(1.0, np.abs(b))).max())


def check_v1(cl, wb, w, cfg, vi, p, s, r):
    N, K = wb.meta["N"], cfg["K"]
    ref = R.forward(T.encode(p, vi), s, w, cfg)
    lg, latv = SH.read_ranges(cl, [f"'Logits'!{rng(1, 1, N, N * (K + 1))}", "'Latent'!A1:" + "ZZ1"])
    rel = rel_err(np.array(lg, dtype=float).reshape(N, N, K + 1), ref["logits"])
    c_rel = rel_err(latv[0][: cfg["d"]], ref["c"])
    return ref["classes"], ref["style"], {"logits_max_rel": rel, "c_max_rel": c_rel, "min_margin": ref["min_margin"]}


def check_mdm(cl, wb, w, cfg, vi, p, s, r):
    N, K, TT = wb.meta["N"], cfg["K"], cfg["T"]
    tr = TR.trace_mdm(p, s, w, cfg, vi)
    f = tr["forward"]
    ranges = ["'Latent'!A1:" + "ZZ1"]
    for t in range(1, TT + 1):
        ranges += [f"'{emit_mdm.step_sheet(t, 'Logits')}'!{rng(1, 1, N, N * (K + 1))}",
                   f"'{emit_mdm.step_sheet(t, 'Conf')}'!{rng(1, 1, N, 3 * N + 2)}",
                   f"'{emit_mdm.step_sheet(t, 'Canvas')}'!{rng(1, 1, N, N)}"]
    got = SH.read_ranges(cl, ranges)
    c_rel = rel_err(got[0][0][: cfg["d"]], f["c"])
    lrel, crel, canvas_bad = 0.0, 0.0, []
    for t in range(TT):
        lg, cf, cv = got[1 + 3 * t: 4 + 3 * t]
        st = f["steps"][t]
        lrel = max(lrel, rel_err(np.array(lg, float).reshape(N, N, K + 1), st["logits"]))
        cf = np.array(cf, dtype=object)
        crel = max(crel, rel_err(cf[:, N + 1: 2 * N + 1].astype(float), st["conf"]))
        if not np.array_equal(np.array(cv, float), st["canvas"]):
            canvas_bad.append(t + 1)
    return f["classes"], f["style"], {"logits_max_rel": lrel, "conf_max_rel": crel, "c_max_rel": c_rel,
                                      "canvas_mismatch_steps": canvas_bad, "min_margin": f["min_margin"],
                                      "margins": f["margins"], "tau_gaps": f["tau_gaps"], "committed": f["committed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts/model/v32"))
    ap.add_argument("--model-type", choices=["auto", "v1", "mdm"], default="auto")
    ap.add_argument("--sheet", default=None, help="key in config.json sheet_ids (default v32 for v1, diffusion for v2)")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--random-cfg", default=None, help="JSON overrides of mdm.MDM_CFG for a random-weight v2 "
                    "(e.g. the spike's half width), used only when the model dir has no weights.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not SH.available():
        print("A4 NOT RUN: no service-account key/config in ~/.config/spritesheet")
        sys.exit(2)
    mt = G.model_type_of(a.model, a.model_type)
    base = None
    if a.random_cfg:
        from spritesheet import mdm as D
        base = dict(D.MDM_CFG, **json.loads(a.random_cfg))
    w, cfg, vocab, pal, src = G.load_model(a.model, random_base=base, model_type=mt)
    mdm = G.is_mdm(cfg)
    sheet = a.sheet or ("diffusion" if mdm else "v32")
    if mdm and sheet == "v32":
        sys.exit("refusing to push a v2 model into the v1 demo sheet (v32); use --sheet diffusion")
    out = a.out or os.path.join(ROOT, "artifacts/metrics", "parity_sheets_mdm.json" if mdm else "parity_sheets.json")
    wb = G.compile_any(w, cfg, vocab, pal, render=True)
    st = wb.stats()
    cl = SH.Client(sheet)
    res = {"gate": "A4", "model": src, "model_type": "mdm" if mdm else "v1", "sheet": sheet,
           "model_hash": wb.meta["model_hash"], "config": cfg,
           "cells": st["cells"], "formulas": st["formulas"], "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if a.skip_push:
        got = SH.read_ranges(cl, ["'About'!B2"])[0]
        if not got or got[0][0] != wb.meta["model_hash"]:
            sys.exit("sheet holds a different model (About!B2 hash mismatch); run without --skip-push")
        res["upload"] = None
    else:
        res["upload"] = SH.push(wb, cl)
    vi = T.index_vocab(vocab)
    cases = G.e2e_cases(a.prompts, vocab, a.seeds, rng_seed=1)
    rows, worst, n_px_bad, n_style_bad, lat = [], 0.0, 0, 0, []
    check = check_mdm if mdm else check_v1
    for p, s in cases:
        r = SH.set_prompt_and_read(p, s, cl, wb, timeout=a.timeout)
        ref_cls, ref_style, info = check(cl, wb, w, cfg, vi, p, s, r)
        px_ok = bool(np.array_equal(np.array(r["classes"]), ref_cls)) and not info.get("canvas_mismatch_steps")
        st_ok = r["style"] == ref_style
        worst = max(worst, info["logits_max_rel"], info["c_max_rel"], info.get("conf_max_rel", 0.0))
        n_px_bad += not px_ok
        n_style_bad += not st_ok
        lat.append(r["first_consistent_s"])
        rows.append({"prompt": p, "seed": s, "pixels_match": px_ok, "style_match": st_ok,
                     "latency_s": r["first_consistent_s"], "stable_s": r["elapsed_s"], **info})
        print(f"{p[:40]!r:44} seed={s:<8} px={px_ok} style={st_ok} rel={info['logits_max_rel']:.2g} "
              f"t={r['first_consistent_s']:.1f}s stable={r['elapsed_s']:.1f}s", flush=True)
    res.update({"cases": len(cases), "pixels_mismatch": n_px_bad, "style_mismatch": n_style_bad, "max_rel": worst,
                "latency_s": {"median": float(np.median(lat)), "max": float(np.max(lat)), "min": float(np.min(lat))},
                "stable_s": {"median": float(np.median([x["stable_s"] for x in rows])),
                             "max": float(np.max([x["stable_s"] for x in rows]))},
                "min_margin": float(min(x["min_margin"] for x in rows)), "rows": rows,
                "requests": cl.requests, "retries": cl.retries})
    if mdm:
        res["steps"] = [{"step": t + 1,
                         "min_top2_margin": min((x["margins"][t] for x in rows if x["margins"][t] is not None),
                                                default=None),
                         "min_conf_tau_gap": min((x["tau_gaps"][t] for x in rows if x["tau_gaps"][t] is not None),
                                                 default=None)} for t in range(cfg["T"])]
    res["pass"] = n_px_bad == 0 and n_style_bad == 0 and worst <= 1e-12
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=G.np_default)
    for s_ in res.get("steps", []):
        print(f"  step {s_['step']}: min|conf-tau|={G._g(s_['min_conf_tau_gap'])} "
              f"min top1-top2={G._g(s_['min_top2_margin'])}")
    print(f"A4 sheets[{sheet}]: cases={len(cases)} max_rel={worst:.3g} pixel_mismatch={n_px_bad} "
          f"style_mismatch={n_style_bad} latency median={res['latency_s']['median']:.1f}s "
          f"stable median={res['stable_s']['median']:.1f}s -> {'PASS' if res['pass'] else 'FAIL'} ({out})")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
