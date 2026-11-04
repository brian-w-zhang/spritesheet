"""Does the inert-text upload (formulas as "#=…" text, then per-tab find-replace) get a full-width v2 into Sheets?
uv run python scripts/spike_inert_upload.py --T 4 [--model artifacts/model/v2_mdm_b]
→ artifacts/metrics/spike_inert_upload.json
"""
import argparse, json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet import mdm as D, tokenizer as T  # noqa: E402
from spritesheet.pixcell import gates as G, sheets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--model", default=None, help="trained v2 dir; default: random full-width weights")
    ap.add_argument("--prompts", default="a red apple|a blue fox|a tiny house in the top left")
    a = ap.parse_args()
    if a.model:
        w, cfg, vocab, pal, src = G.load_model(a.model, model_type="mdm")
    else:
        cfg0 = dict(D.MDM_CFG, T=a.T)
        w, cfg, vocab, pal = G.random_mdm_full(cfg0)
        src = {"source": "random full-width"}
    wb = G.compile_any(w, cfg, vocab, pal, render=True)
    n_f = sum(1 for c in wb.sheets.values() for v in c.values() if isinstance(v, str) and v.startswith("="))
    print(f"{src['source']}: T={cfg['T']} formulas={n_f}", flush=True)
    cl = sheets.Client("diffusion")
    t0 = time.time()
    rep = sheets.push(wb, cl, inert=True)
    up = time.time() - t0
    print(f"uploaded+activated in {up:.0f}s (values {rep['values_s']:.0f}s, activate {rep['activate_s']:.0f}s)", flush=True)
    vidx = T.index_vocab(vocab)
    rows = []
    for p in a.prompts.split("|"):
        got = sheets.set_prompt_and_read(p, 1, cl, wb, timeout=1800)
        ref = D.ref_generate(T.encode(p, vidx), 1, w, cfg)["classes"]
        ok = bool((np.asarray(got["classes"]) == ref).all())
        rows.append({"prompt": p, "match": ok, "first_consistent_s": got["first_consistent_s"], "stable_s": got["elapsed_s"]})
        print(rows[-1], flush=True)
    out = {"source": src["source"], "T": cfg["T"], "formulas": n_f, "upload_activate_s": up, "push": rep, "renders": rows}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "metrics", "spike_inert_upload.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
