"""Compile a model and push it to a shared Google Sheet (the demo build), then render one prompt.

uv run python scripts/push_sheets.py --model artifacts/model/v32 --prompt "a red apple" --seed 1
uv run python scripts/push_sheets.py --model artifacts/model/v2_mdm_a --sheet diffusion
The model type (v1 / v2 masked diffusion) is auto-detected from config.json "model" (or --model-type). The sheet
defaults to v32 for v1 and diffusion for v2. Needs ~/.config/spritesheet/{sa-key.json,config.json}. Never prints
the key or the spreadsheet id.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from spritesheet import mdm as D, reference as R, tokenizer as T  # noqa: E402
from spritesheet.pixcell import gates as G, sheets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "model", "v32"))
    ap.add_argument("--model-type", choices=["auto", "v1", "mdm"], default="auto")
    ap.add_argument("--sheet", default=None, help="key in config.json sheet_ids (default v32 / diffusion)")
    ap.add_argument("--prompt", default="a red apple")
    ap.add_argument("--seed", type=float, default=1)
    a = ap.parse_args()
    if not sheets.available():
        print("NOT RUN: no service-account key/config")
        sys.exit(2)
    w, cfg, vocab, pal, src = G.load_model(a.model, model_type=G.model_type_of(a.model, a.model_type))
    mdm = G.is_mdm(cfg)
    key = a.sheet or ("diffusion" if mdm else "v32")
    if mdm and key == "v32":
        sys.exit("refusing to push a v2 model into the v1 demo sheet (v32); use --sheet diffusion")
    wb = G.compile_any(w, cfg, vocab, pal, default_prompt=a.prompt, default_seed=a.seed, render=True)
    cl = sheets.Client(key)
    t0 = time.time()
    rep = sheets.push(wb, cl)
    print(f"pushed {src['source']} {'v2' if mdm else 'v1'} model to sheet '{key}' in {time.time() - t0:.1f} s",
          {k: rep[k] for k in ("requests", "bytes_sent") if k in rep})
    got = sheets.set_prompt_and_read(a.prompt, a.seed, cl, wb, timeout=1800)
    ids = T.encode(a.prompt, T.index_vocab(vocab))
    ref = (D.ref_generate(ids, a.seed, w, cfg) if mdm else R.forward(ids, a.seed, w, cfg))["classes"]
    ok = bool((np.asarray(got["classes"]) == ref).all())
    print("render", repr(a.prompt), "seed", a.seed, "matches reference:", ok,
          f"latency: {got['first_consistent_s']:.1f}s (stable {got['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
