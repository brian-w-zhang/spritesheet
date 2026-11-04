"""Compile a model and push it to the shared Google Sheet (the demo build), then render one prompt.

uv run python scripts/push_sheets.py --model artifacts/model/v32 --prompt "a red apple" --seed 1
Needs ~/.config/spritesheet/{sa-key.json,config.json}. Never prints the key or the spreadsheet id.
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet import reference as R, tokenizer as T  # noqa: E402
from spritesheet.pixcell import sheets  # noqa: E402
from spritesheet.pixcell.emit import compile_model  # noqa: E402

BUILD = os.path.join(ROOT, "data", "build")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "model", "v32"))
    ap.add_argument("--prompt", default="a red apple")
    ap.add_argument("--seed", type=float, default=1)
    a = ap.parse_args()
    if not sheets.available():
        print("NOT RUN: no service-account key/config")
        sys.exit(2)
    w, cfg = R.load(a.model)
    vocab = [l.rstrip("\n") for l in open(os.path.join(BUILD, "vocab.txt"), encoding="utf-8")]
    pal = json.load(open(os.path.join(BUILD, "palette.json")))["colors"]
    wb = compile_model(w, cfg, vocab, pal, default_prompt=a.prompt, default_seed=a.seed, render=True)
    t0 = time.time()
    rep = sheets.push(wb)
    print("pushed in %.1f s" % (time.time() - t0), {k: rep[k] for k in ("requests", "bytes_sent") if k in rep})
    got = sheets.set_prompt_and_read(a.prompt, a.seed, wb=wb)
    ref = R.forward(T.encode(a.prompt, T.load_vocab(os.path.join(BUILD, "vocab.txt"))), a.seed, w, cfg)
    cls = got[0] if isinstance(got, tuple) else got.get("classes")
    import numpy as np
    ok = cls is not None and (np.asarray(cls) == ref["classes"]).all()
    print("render", a.prompt, "seed", a.seed, "matches reference:", bool(ok), "latency:", got[1] if isinstance(got, tuple) else got.get("elapsed_s"))


if __name__ == "__main__":
    main()
