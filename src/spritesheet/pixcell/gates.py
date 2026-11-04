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
BATCH = {"hyperformula": 40, "libreoffice": 8}


def _batched_run(engine, wb, runs, batch=None):
    """Run in batches (bounded memory / disk); merge timings (sums, plus per-batch lists)."""
    b = batch or BATCH[engine]
    allr, timing, version = [], {}, None
    for i in range(0, len(runs), b):
        rr, tm, version = _engine_run(engine, wb, runs[i:i + b])
        allr.extend(rr)
        for k, v in tm.items():
            if isinstance(v, list):
                timing.setdefault(k, []).extend(v)
            else:
                timing[k] = timing.get(k, 0.0) + v
    timing["batches"] = (len(runs) + b - 1) // b
    timing["runs"] = len(runs)
    return allr, timing, version


def parity(wb, w, cfg, vocab, cases, engines=ENGINES, read=None, batch=None) -> dict:
    """Run cases [(prompt, seed)] on each engine; compare blocks to the reference trace.
    read: sheets to read back (None = all; blocks on unread sheets are skipped)."""
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
        rr, timing, version = _batched_run(eng, wb, runs, batch)
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


# ------------------------------------------------------------------ model loading + A3
def load_model(model_dir=None, random_base=None, seed=0):
    """Trained model dir (config.json + weights.npz [+ vocab.txt, palette.json]) or random weights.

    Vocab: <model_dir>/vocab.txt, else data/build/vocab.txt. Palette: <model_dir>/palette.json, else
    data/build/palette.json. Falls back to random weights (fixtures.V32 or random_base) when the dir is absent.
    """
    import json
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if model_dir and os.path.exists(os.path.join(model_dir, "weights.npz")):
        w, cfg = R.load(model_dir)
        cfg.setdefault("L", T.L)
        vp = next(p for p in (os.path.join(model_dir, "vocab.txt"), os.path.join(root, "data/build/vocab.txt"))
                  if os.path.exists(p))
        vocab = [x.rstrip("\n") for x in open(vp, encoding="utf-8")]
        pp = next(p for p in (os.path.join(model_dir, "palette.json"), os.path.join(root, "data/build/palette.json"))
                  if os.path.exists(p))
        pal = json.load(open(pp))["colors"]
        return w, cfg, vocab, pal, {"source": "trained", "model_dir": model_dir, "vocab": vp, "palette": pp}
    cfg = F.make_cfg(random_base or F.V32)
    w, vocab, pal = F.random_model(cfg, seed)
    return w, cfg, vocab, pal, {"source": "random", "model_dir": model_dir, "weights_seed": seed}


TEMPLATES = ["a {a} {n}", "{n}", "the {a} {n} with a {n2}", "{a} {n} in {s} style", "two {a} {n}s", "{n} and {n2}",
             "a big {a} {n}", "{A} {N}!!", "a tiny {a} {n}, facing left"]


def e2e_cases(n, vocab, seeds=2, rng_seed=0):
    import random
    rnd = random.Random(rng_seed)
    words = [x for x in vocab[2:] if x.isalnum()] or ["apple"]
    prompts = []
    adv = F.adversarial_prompts(max(1, n // 4), vocab, rng_seed, long_words=False)
    while len(prompts) < n - len(adv):
        t = rnd.choice(TEMPLATES)
        a, nn, n2, s = (rnd.choice(words) for _ in range(4))
        prompts.append(t.format(a=a, n=nn, n2=n2, s=s, A=a.upper(), N=nn.capitalize()))
    prompts = (prompts + adv)[:n]
    cases = []
    for i, p in enumerate(prompts):
        for j in range(seeds):
            cases.append((p, 1 + j if i == 0 else rnd.randint(0, 10 ** 6)))
    return cases


E2E_READ = ["Tokens", "Latent", "H", "FiLM", "Dense", "Conv_3", "Logits", "Pixels"]


def gate_a3(model_dir=None, n_prompts=20, seeds=2, engines=ENGINES, lo_max=None, all_tabs=False, log=print):
    w, cfg, vocab, pal, src = load_model(model_dir)
    t0 = time.time()
    wb = emit.compile_model(w, cfg, vocab, pal)
    t_compile = time.time() - t0
    cases = e2e_cases(n_prompts, vocab, seeds)
    res = {"gate": "A3", "model": src, "config": cfg, "compile_s": t_compile, "prompts": n_prompts, "seeds": seeds,
           "engines": {}}
    read = None if all_tabs else E2E_READ
    for eng in engines:
        cs = cases if (eng != "libreoffice" or not lo_max) else cases[: lo_max]
        log(f"[A3] {eng}: {len(cs)} cases ...")
        r = parity(wb, w, cfg, vocab, cs, (eng,), read=read)
        e = r["engines"][eng]
        e["cases"] = len(cs)
        res["engines"][eng] = e
        for k in ("stats", "min_margin", "styles", "lint_violations", "functions"):
            res[k] = r[k]
        res.setdefault("cases", r["cases"])
    res["max_rel_per_tab"] = {}
    for eng, e in res["engines"].items():
        for b, v in e["per_block"].items():
            res["max_rel_per_tab"][b] = max(res["max_rel_per_tab"].get(b, 0.0), v["max_rel"])
    res["pass"] = res["lint_violations"] == 0 and all(
        e["n_fail"] == 0 and e["error_cells"] == 0 and e["stale_cells"] == 0 and e["max_rel"] <= TOL
        for e in res["engines"].values())
    return res
