"""Resume an inert push whose activation was cut off (sheets.activate is idempotent), then format and render once.

uv run python scripts/activate_sheet.py --model artifacts/model/v3p_random --sheet v3 [--start Out]
"""
import argparse, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from spritesheet import tokenizer as T  # noqa: E402
from spritesheet.pixcell import gates as G, sheets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--prompt", default="a red frog")
    ap.add_argument("--seed", type=float, default=1)
    a = ap.parse_args()
    w, cfg, vocab, pal, src = G.load_model(a.model)
    wb = G.compile_any(w, cfg, vocab, pal, default_prompt=a.prompt, default_seed=a.seed, render=True)
    cl = sheets.Client(a.sheet)
    t0 = time.time()
    sheets.activate(wb, cl, start=a.start)
    ids = {x["properties"]["title"]: x["properties"]["sheetId"] for x in cl.meta().get("sheets", [])}
    sheets.format_prompt(cl, wb, ids["Prompt"])
    print(f"activated in {time.time() - t0:.0f}s", flush=True)
    got = sheets.set_prompt_and_read(a.prompt, a.seed, cl, wb, timeout=1800)
    ids_ = T.encode(a.prompt, T.index_vocab(vocab))
    if cfg.get("model") == "v3p":
        from spritesheet import reference_p as RP
        ref = RP.forward(ids_, a.seed, w, cfg, keep=False)["pixels"]
    else:
        from spritesheet import reference as R
        ref = R.forward(ids_, a.seed, w, cfg)["classes"]
    print("render", repr(a.prompt), "matches reference:", bool((np.asarray(got["classes"]) == ref).all()),
          f"latency: {got['first_consistent_s']:.1f}s (stable {got['elapsed_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
