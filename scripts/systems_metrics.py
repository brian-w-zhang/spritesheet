"""D: systems numbers for a model dir → artifacts/metrics/systems.json.
Combines: a fresh compile (cell/formula/char counts, xlsx bytes, value vs formula cells per tab group),
parity_local.json (local engine timings) and parity_sheets.json (upload + latency) when they refer to the
same model hash, plus a numpy forward timing.

uv run python scripts/systems_metrics.py --model artifacts/model/v32
"""
import argparse, json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet import reference as R, tokenizer as T  # noqa: E402
from spritesheet.pixcell.emit import compile_model, model_hash  # noqa: E402
from spritesheet.pixcell.xlsx import to_xlsx  # noqa: E402

M = os.path.join(ROOT, "artifacts", "metrics")
BUILD = os.path.join(ROOT, "data", "build")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "model", "v32"))
    a = ap.parse_args()
    w, cfg = R.load(a.model)
    vocab_words = [l.rstrip("\n") for l in open(os.path.join(BUILD, "vocab.txt"), encoding="utf-8")]
    pal = json.load(open(os.path.join(BUILD, "palette.json")))["colors"]
    h = model_hash(w, cfg, vocab_words, pal)
    t0 = time.time()
    wb = compile_model(w, cfg, vocab_words, pal, render=True)
    compile_s = time.time() - t0
    per_tab, tot_f, tot_v, chars = {}, 0, 0, 0
    for name, cells in wb.sheets.items() if hasattr(wb, "sheets") else wb.items():
        f = sum(1 for v in cells.values() if isinstance(v, str) and v.startswith("="))
        v = len(cells) - f
        per_tab[name] = {"formulas": f, "values": v}
        tot_f += f
        tot_v += v
        chars += sum(len(x) for x in cells.values() if isinstance(x, str) and x.startswith("="))
    groups = {"weights (W_*, WordVec, style/palette tables)": 0, "vocabulary": 0}
    for name, c in per_tab.items():
        if name.startswith("W_") or name in ("WordVec",):
            groups["weights (W_*, WordVec, style/palette tables)"] += c["values"]
        if name == "Vocab":
            groups["vocabulary"] += c["values"]
    os.makedirs(os.path.join(ROOT, "artifacts", "workbooks"), exist_ok=True)
    xp = os.path.join(ROOT, "artifacts", "workbooks", "spritesheet-32.xlsx")
    to_xlsx(wb, xp)
    params = int(sum(np.asarray(v).size for k, v in w.items() if k != "wordvec"))
    vocab = T.load_vocab(os.path.join(BUILD, "vocab.txt"))
    ids = T.encode("a tiny blue rocket in the top left", vocab)
    R.forward(ids, 1, w, cfg)
    t0 = time.time()
    for s in range(20):
        R.forward(ids, s + 1, w, cfg)
    numpy_s = (time.time() - t0) / 20
    out = {"model_hash": h, "model_dir": os.path.relpath(a.model, ROOT), "config": cfg,
           "trainable_params_excl_wordtable": params, "wordtable_values": int(np.asarray(w["wordvec"]).size),
           "total_cells": tot_f + tot_v, "formula_cells": tot_f, "value_cells": tot_v, "formula_chars": chars,
           "tabs": len(per_tab), "cell_limit_google_sheets": 20_000_000,
           "share_of_cell_limit": (tot_f + tot_v) / 20_000_000, "value_cell_groups": groups,
           "xlsx_bytes": os.path.getsize(xp), "compile_s": compile_s, "numpy_forward_s": numpy_s, "per_tab": per_tab}
    for f, key in (("parity_local.json", "local"), ("parity_sheets.json", "sheets")):
        p = os.path.join(M, f)
        if os.path.exists(p):
            d = json.load(open(p))
            same = d.get("model_hash") == h or (d.get("model") or {}).get("hash") == h
            if key == "local":
                hf = d["engines"]["hyperformula"]["timing"]
                lo = d["engines"]["libreoffice"]["timing"]
                out["local"] = {"same_model": same, "hyperformula_build_s": hf.get("build_s"),
                                "hyperformula_median_prompt_s": float(np.median(hf["per_run_eval_s"])) if hf.get("per_run_eval_s") else None,
                                "libreoffice_wall_s": lo.get("wall_s"), "libreoffice_cases": d["engines"]["libreoffice"].get("cases")}
            else:
                up = d.get("upload") or {}  # None when A4 ran with --skip-push (model already in the sheet)
                out["sheets"] = {"same_model": same, "upload_s": up.get("upload_s"), "upload_bytes": up.get("bytes_sent"),
                                 "upload_requests": up.get("requests"), "latency_s": d["latency_s"], "cases": d["cases"]}
                if same and out["numpy_forward_s"] > 0:
                    out["sheets"]["slowdown_vs_numpy_median"] = d["latency_s"]["median"] / out["numpy_forward_s"]
    json.dump(out, open(os.path.join(M, "systems.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_tab", "config")}, indent=1))


if __name__ == "__main__":
    main()
