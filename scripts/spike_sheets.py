#!/usr/bin/env python3
"""Spikes R1 + R3 on the real shared Google Sheet (sheet_ids.v32): upload cost and prompt->pixels latency.

uv run python scripts/spike_sheets.py [--configs full,half] [--prompts 3]
R1: push a random-weight workbook per config, time 3 prompts. R3: SPARKLINE cells read back without errors,
achieved pixelSize. Writes artifacts/metrics/spikes.json. Never prints the key or the spreadsheet id.
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
from spritesheet.pixcell import emit, fixtures as F, sheets as SH  # noqa: E402
from spritesheet.pixcell.workbook import rng  # noqa: E402

PROMPTS = ["a red apple", "a tiny blue rocket facing left", "green cat with a hat"]


def config(name):
    base = dict(F.V32)
    if name == "half":
        for k in ("C0", "C1", "C2", "C3"):
            base[k] //= 2
    elif name == "small":
        base = dict(F.SMALL)
    return F.make_cfg(base)


def r3_check(cl, wb):
    d = wb.meta["display"]
    rg = f"'Prompt'!{rng(d['r0'], d['c0'], d['r0'] + d['n'] - 1, d['c0'] + d['n'] - 1)}"
    f = SH.read_ranges(cl, [rg], "FORMULA")[0]
    v = SH.read_ranges(cl, [rg], "FORMATTED_VALUE")[0]
    flat_f = [x for row in f for x in row]
    flat_v = [x for row in v for x in row]
    errs = [x for x in flat_v if isinstance(x, str) and x.startswith("#")]
    hexv = SH.read_ranges(cl, ["'Hex'!A1:AF32"])[0]
    return {"sparkline_cells": sum(1 for x in flat_f if isinstance(x, str) and x.startswith("=SPARKLINE")),
            "error_cells": len(errs), "error_examples": errs[:3],
            "hex_ok": all(isinstance(x, str) and x.startswith("#") and len(x) == 7 for row in hexv for x in row),
            "sample_formula": flat_f[0] if flat_f else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="half,full")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/metrics/spikes.json"))
    a = ap.parse_args()
    if not SH.available():
        sys.exit("NOT RUN: no service-account key / config in ~/.config/spritesheet")
    report = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": []}
    if os.path.exists(a.out):
        try:
            old = json.load(open(a.out))
            report["previous"] = old.get("runs", [])[-4:]
        except Exception:  # noqa: BLE001
            pass
    for name in a.configs.split(","):
        cfg = config(name)
        w, vocab, pal = F.random_model(cfg)
        wb = emit.compile_model(w, cfg, vocab, pal, render=True)
        st = wb.stats()
        entry = {"config_name": name, "config": cfg, "cells": st["cells"], "formulas": st["formulas"],
                 "formula_chars": st["formula_chars"], "errors": []}
        print(f"[{name}] cells={st['cells']} formulas={st['formulas']} pushing ...", flush=True)
        cl = SH.Client()
        try:
            up = SH.push(wb, cl)
            entry["upload"] = up
            print(f"[{name}] upload {up['upload_s']:.1f}s, {up['requests']} requests, {up['bytes_sent'] / 1e6:.1f} MB",
                  flush=True)
            vi = T.index_vocab(vocab)
            lat = []
            for k, p in enumerate(PROMPTS[: a.prompts]):
                seed = k + 1
                r = SH.set_prompt_and_read(p, seed, cl, wb)
                ref = R.forward(T.encode(p, vi), seed, w, cfg)
                same = bool(np.array_equal(np.array(r["classes"]), ref["classes"]))
                lat.append({"prompt": p, "seed": seed, "elapsed_s": r["elapsed_s"], "write_s": r["write_s"],
                            "first_consistent_s": r["first_consistent_s"], "reads": r["reads"],
                            "pixels_match_reference": same, "style_sheet": r["style"], "style_ref": ref["style"]})
                print(f"[{name}] prompt {k + 1}: {r['elapsed_s']:.1f}s (consistent at {r['first_consistent_s']:.1f}s) "
                      f"match={same}", flush=True)
            entry["latency"] = lat
            entry["r3"] = r3_check(cl, wb)
            entry["r3"]["pixel"] = up["format"]
            print(f"[{name}] R3 {entry['r3']}", flush=True)
        except Exception as ex:  # noqa: BLE001
            entry["errors"].append(f"{type(ex).__name__}: {ex}")
            print(f"[{name}] ERROR {type(ex).__name__}: {ex}", flush=True)
        entry["requests_total"] = cl.requests
        entry["retries_total"] = cl.retries
        report["runs"].append(entry)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
