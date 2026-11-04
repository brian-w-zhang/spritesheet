"""Parity gates shared by tests/test_pixcell.py and scripts/verify_*.py.

A1 tokenizer parity, A2 op parity (every intermediate block, random weights), A3 end-to-end (any model dir).
Each returns a JSON-able dict with a boolean "pass".
"""
import time

import numpy as np

from .. import reference as R
from .. import tokenizer as T
from ..oracles import hyperformula as HF
from ..oracles import libreoffice as LO
from ..oracles import scan_errors
from . import emit, fixtures as F, lint
from .trace import compare, trace

ENGINES = ("hyperformula", "libreoffice")
TOL = 1e-12


def _engine_run(engine, wb, runs):
    t0 = time.time()
    if engine == "hyperformula":
        out = HF.run(wb, runs)
        timing = {"build_s": out["build_s"], "node_s": out["node_s"],
                  "per_run_eval_s": [r["eval_s"] for r in out["runs"]]}
        version = "HyperFormula " + out["version"]
    else:
        out = LO.run(wb, runs)
        timing = {k: out[k] for k in ("xlsx_s", "variants_s", "convert_s", "read_s")}
        version = out["version"]
    timing["wall_s"] = time.time() - t0
    return out["runs"], timing, version


# ------------------------------------------------------------------ A1
def gate_a1(n=320, engines=ENGINES, vocab=None, seed=0, long_words=True) -> dict:
    vocab = vocab or F.vocab_words(400, seed)
    vi = T.index_vocab(vocab)
    prompts = F.adversarial_prompts(n, vocab, seed, long_words=long_words)
    wb = emit.tokenizer_rows(prompts, vocab)
    L, c0 = T.L, wb.meta["tok"]["id_c0"]
    expect = [T.encode(p, vi) for p in prompts]
    res = {"gate": "A1", "prompts": len(prompts), "vocab": len(vocab), "engines": {}}
    for eng in engines:
        runs, timing, version = _engine_run(eng, wb, [{"read": ["Tok"]}])
        g = runs[0]["values"]["Tok"]
        mism = []
        for p, (prompt, ids) in enumerate(zip(prompts, expect)):
            got = [g[p][c0 - 1 + i] for i in range(L)]
            got = [int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in got]
            if got != ids:
                mism.append({"prompt": prompt, "sheet": got, "python": ids})
        err = scan_errors(wb, runs[0]["values"])
        res["engines"][eng] = {"version": version, "mismatches": len(mism), "examples": mism[:5],
                               "error_cells": err["errors"], "timing": timing}
    res["pass"] = all(e["mismatches"] == 0 and e["error_cells"] == 0 for e in res["engines"].values())
    return res


# ------------------------------------------------------------------ A2 / A3 core
def parity(wb, w, cfg, vocab, cases, engines=ENGINES, read=None) -> dict:
    """Run cases [(prompt, seed)] on each engine; compare blocks to the reference trace."""
    vi = T.index_vocab(vocab)
    traces = [trace(p, s, w, cfg, vi) for p, s in cases]
    runs = [{"prompt": p, "seed": s, "read": read} for p, s in cases]
    out = {"cases": len(cases), "engines": {}, "stats": {k: v for k, v in wb.stats().items() if k != "per_sheet"}}
    out["min_margin"] = float(min(t["forward"]["min_margin"] for t in traces))
    out["styles"] = sorted({int(t["forward"]["style"]) for t in traces})
    lt = lint.lint(wb)
    out["lint_violations"] = len(lt["violations"])
    out["functions"] = lt["functions"]
    for eng in engines:
        rr, timing, version = _engine_run(eng, wb, runs)
        per_block, fails, errors, stale, style_bad, class_bad = {}, [], 0, 0, 0, 0
        for (p, s), t, r in zip(cases, traces, rr):
            cmp = compare(wb, r["values"], t["expected"])
            for b, v in cmp["blocks"].items():
                pb = per_block.setdefault(b, {"max_rel": 0.0, "bad": 0})
                pb["max_rel"] = max(pb["max_rel"], v["max_rel"])
                pb["bad"] += v["bad"]
            if cmp["fail"]:
                fails.append({"prompt": p, "seed": s, "blocks": cmp["fail"][:8]})
            se = scan_errors(wb, r["values"])
            errors += se["errors"]
            stale += se["stale"]
            style_bad += int(cmp["blocks"].get("style", {}).get("bad", 0) > 0)
            class_bad += int(cmp["blocks"].get("Pixels", {}).get("bad", 0) > 0)
        mx = max((v["max_rel"] for v in per_block.values()), default=0.0)
        out["engines"][eng] = {"version": version, "max_rel": mx, "per_block": per_block, "failures": fails[:10],
                               "n_fail": len(fails), "error_cells": errors, "stale_cells": stale,
                               "style_mismatch": style_bad, "pixels_mismatch": class_bad, "timing": timing}
    out["pass"] = out["lint_violations"] == 0 and all(
        e["n_fail"] == 0 and e["error_cells"] == 0 and e["stale_cells"] == 0 and e["max_rel"] <= TOL
        for e in out["engines"].values())
    return out


A2_CASES = [("a red apple", 1), ("Big blue CAT with a hat!!", 7), ("", 3), ("dog " * 14, 123456789),
            ("zzzunknown", 2), ("the small green tree of the sun moon star", 11), ("3d emoji heart", 5),
            ("   pixel   art   sword  ", 2147483646), ("apple", 0), ("a", -4.7)]


def gate_a2(engines=ENGINES, cases=A2_CASES) -> dict:
    res = {"gate": "A2", "models": []}
    for wseed, scale in ((0, 0.3), (1, 0.6)):
        cfg = F.make_cfg()
        w = R.random_weights(cfg, seed=wseed, scale=scale)
        vocab, pal = F.vocab_words(cfg["V"], wseed), F.palette(cfg["K"], wseed)
        wb = emit.compile_model(w, cfg, vocab, pal)
        r = parity(wb, w, cfg, vocab, cases, engines)
        r.update({"weights_seed": wseed, "scale": scale, "config": cfg})
        res["models"].append(r)
    res["pass"] = all(m["pass"] for m in res["models"])
    return res


def summarize(res) -> str:
    lines = []
    for m in res.get("models", [res]):
        for eng, e in m["engines"].items():
            lines.append(f"{res.get('gate', 'A3')} {eng}: cases={m['cases']} max_rel={e['max_rel']:.3g} "
                         f"fail={e['n_fail']} errors={e['error_cells']} stale={e['stale_cells']} "
                         f"style_mismatch={e['style_mismatch']} pixel_mismatch={e['pixels_mismatch']} "
                         f"min_margin={m['min_margin']:.3g} styles={m['styles']}")
    return "\n".join(lines)


def np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
