"""Export the generation bank for the public website (tools/bank-site, Next.js static export on Vercel).

  uv run python scripts/export_bank_site.py
Reads artifacts/bank/index.html (the PUBLIC viewer: our models' sprites only, never index_local.html, which embeds
licensed PixelGPT sprites) and writes tools/bank-site/public/data/:
  meta.json            prompts, seeds, per-model palette/labels/scores, visual-review verdicts
  sprites_<model>.gz   gzip of uint8 palette indices, [prompt*8 + seed-1][N*N] (the browser inflates it)
"""
import base64, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "artifacts", "bank", "index.html")
OUT = os.path.join(ROOT, "tools", "bank-site", "public", "data")


def main():
    html = open(SRC).read()
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    if "real" in data:
        sys.exit("refusing: this bank page embeds real training sprites")
    os.makedirs(OUT, exist_ok=True)
    meta = {"prompts": [{k: p.get(k) for k in ("prompt", "category", "base", "axis", "n_train")} for p in data["prompts"]],
            "seeds": data["seeds"], "models": {}, "review": {}}
    for k, m in data["models"].items():
        raw = base64.b64decode(m["sprites"])
        if raw[:2] != b"\x1f\x8b":
            sys.exit(f"{k}: sprites are not gzip; regenerate the bank with scripts/gen_bank.py")
        open(os.path.join(OUT, f"sprites_{k}.gz"), "wb").write(raw)
        meta["models"][k] = {"label": m["label"], "n": m["n"], "palette": m["palette"],
                             "score": [round(x, 2) for x in m["score"]]}
    for k, rv in (data.get("review") or {}).items():
        meta["review"][k] = {i: {f: r.get(f) for f in ("best_seed", "verdict", "good_seeds", "note", "shortlist") if r.get(f) is not None}
                             for i, r in rv.items()}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), separators=(",", ":"))
    for f in sorted(os.listdir(OUT)):
        print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
