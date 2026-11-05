"""Add a "Gallery" tab to a demo spreadsheet: the best reviewed generations of that sheet's model, as cells.

  uv run python scripts/push_gallery.py --sheet v3 --model v3 [--max 400] [--bank DIR] [--dry]
  (--sheet probe to try the layout on the probe sheet)

Picks come from the visual review (artifacts/bank/review/*.json + shortlist.json, merged by bank_merge_review.py
into DIR/index.html): shortlist first, then every seed the reviewers called "good". Each sprite is a square block of
square cells holding palette indices; 49 conditional-format rules paint them and a ";;;" number format hides the
digits. Default: static background fills, one repeatCell per run of one colour (~25 KB per sprite to upload, but the
tab opens as fast as ordinary formatting); --cf instead paints with 49 conditional-format rules (tiny upload, but the
browser re-evaluates the rules over every cell, which got slow at ~800 sprites).
A merged label row under each tile gives prompt, seed and verdict. Constants only: nothing here recalculates.
Only touches the tab named "Gallery" (deleted and rebuilt if it carries our marker in A1).
"""
import argparse, base64, gzip, json, os, re, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet.pixcell import sheets as SH  # noqa: E402

TAB = "Gallery"
MARK = "Gallery: best generations"
LABEL_PX, GAP_COLS, HEAD = 34, 2, 3


def load_bank(d):
    html = open(os.path.join(d, "index.html")).read()
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    out = {}
    for k, m in data["models"].items():
        raw = base64.b64decode(m["sprites"])
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass
        out[k] = {"n": m["n"], "palette": m["palette"], "label": m["label"],
                  "grids": np.frombuffer(raw, np.uint8).reshape(-1, m["n"], m["n"])}
    return data, out


def picks(data, key, limit, grids, include_ok=False):
    S, seeds = len(data["seeds"]), data["seeds"]
    rev = (data.get("review") or {}).get(key, {})
    first, rest = [], []
    for i, r in sorted(rev.items(), key=lambda t: int(t[0])):
        i = int(i)
        good = list(r.get("good_seeds") or [])
        if r.get("best_seed") and r.get("verdict") == "good" and r["best_seed"] not in good:
            good.insert(0, r["best_seed"])
        for s in good:
            item = (i, s, r)
            (first if r.get("shortlist") and s == r.get("best_seed") else rest).append(item)
    rest.sort(key=lambda t: (t[1] != t[2].get("best_seed"), t[0]))   # best seeds before alternates
    if include_ok:   # after every good one: each "ok" row's best seed (for models with few goods, e.g. v2)
        rest += [(int(i), r["best_seed"], r) for i, r in sorted(rev.items(), key=lambda t: int(t[0]))
                 if r.get("verdict") == "ok" and r.get("best_seed")]
    out, seen = [], set()
    for i, s, r in first + rest:   # v1/v2 seeds often give the identical sprite: keep one copy per prompt
        j = i * S + seeds.index(s)
        key_ = (i, grids[j].tobytes())
        if key_ not in seen:
            seen.add(key_)
            out.append((i, s, r, j))
    return out[:limit]


def rgb(h):
    return {"red": int(h[1:3], 16) / 255, "green": int(h[3:5], 16) / 255, "blue": int(h[5:7], 16) / 255}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, help="key in config.json sheet_ids (v32, diffusion, v3, probe)")
    ap.add_argument("--model", required=True, choices=["v1", "v2", "v3"])
    ap.add_argument("--bank", default=os.path.join(ROOT, "artifacts", "bank"))
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--per-row", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--cf", action="store_true", help="paint with 49 conditional-format rules instead of static fills")
    ap.add_argument("--include-ok", action="store_true", help="append 'ok' best seeds after the good ones")
    a = ap.parse_args()
    data, models = load_bank(a.bank)
    m = models[a.model]
    N, pal = m["n"], m["palette"]
    px = 6 if N <= 24 else 5
    items = picks(data, a.model, a.max, m["grids"], a.include_ok)
    if not items:
        sys.exit(f"no reviewed picks for {a.model}")
    T = a.per_row
    tw, th = N + GAP_COLS, N + 2            # tile: N sprite rows, 1 label row, 1 gap row
    bands = (len(items) + T - 1) // T
    ncols, nrows = T * tw, HEAD + bands * th
    print(f"{len(items)} sprites -> {nrows} x {ncols} cells on '{a.sheet}'", flush=True)
    if a.dry:
        return

    cl = SH.Client(a.sheet)
    meta = cl.meta()
    old = [s["properties"] for s in meta["sheets"] if s["properties"]["title"] == TAB]
    if old:
        a1 = SH.read_ranges(cl, [f"'{TAB}'!A1"])[0]
        if not (a1 and str(a1[0][0]).startswith(MARK)):
            sys.exit(f"a '{TAB}' tab exists that this script did not make; not touching it")
        cl.batch_update([{"deleteSheet": {"sheetId": old[0]["sheetId"]}}])
    pi = [x["properties"]["index"] for x in meta["sheets"] if x["properties"]["title"] == "Prompt"]
    at = pi[0] + 1 if pi else 0   # right after the Prompt tab
    rep = cl.batch_update([{"addSheet": {"properties": {"title": TAB, "index": at,
                                                        "gridProperties": {"rowCount": nrows, "columnCount": ncols,
                                                                           "frozenRowCount": HEAD}}}}])
    gid = rep["replies"][0]["addSheet"]["properties"]["sheetId"]
    G = lambda r0, r1, c0, c1: {"sheetId": gid, "startRowIndex": r0, "endRowIndex": r1, "startColumnIndex": c0,
                                "endColumnIndex": c1}

    # values: header + every band (palette indices; label text in each tile's first label cell)
    grid = [[""] * ncols for _ in range(nrows)]
    grid[0][0] = f"{MARK} from {m['label']}, picked by visual review ({len(items)} sprites; shortlist first, then good{', then ok' if a.include_ok else ''})"
    grid[1][0] = "Each tile: prompt, seed and the reviewer's verdict. Type the prompt in Prompt!B2 and the seed in B3 to redraw it live."
    for n, (i, s, r, j) in enumerate(items):
        r0, c0 = HEAD + (n // T) * th, (n % T) * tw
        g = m["grids"][j]
        for y in range(N):
            grid[r0 + y][c0:c0 + N] = [int(v) for v in g[y]]
        tag = r.get("verdict", "")
        if r.get("shortlist") and s == r.get("best_seed"):
            tag += ", shortlist"
        grid[r0 + N][c0] = f"{data['prompts'][i]['prompt']} · seed {s}\n{tag}"
    data_req, chunk, size = [], [], 0
    for r0 in range(0, nrows, 50):
        rows = grid[r0:r0 + 50]
        sz = len(json.dumps(rows))
        if size + sz > 1_500_000 and chunk:
            data_req.append(chunk); chunk, size = [], 0
        chunk.append({"range": f"'{TAB}'!A{r0 + 1}", "values": rows}); size += sz
    if chunk:
        data_req.append(chunk)

    # layout: square cells, tall label rows, merged labels, hidden digits, one colour rule per palette entry
    reqs = [{"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": "COLUMNS", "startIndex": 0,
                                                     "endIndex": ncols}, "properties": {"pixelSize": px},
                                           "fields": "pixelSize"}},
            {"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": "ROWS", "startIndex": HEAD,
                                                     "endIndex": nrows}, "properties": {"pixelSize": px},
                                           "fields": "pixelSize"}},
            {"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": "ROWS", "startIndex": 0,
                                                     "endIndex": HEAD}, "properties": {"pixelSize": 26},
                                           "fields": "pixelSize"}}]
    reqs.append({"updateSheetProperties": {"properties": {"sheetId": gid, "gridProperties": {"hideGridlines": True}},
                                           "fields": "gridProperties.hideGridlines"}})
    hide = {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": ";;;"}}}
    lab = {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP",
                                 "numberFormat": {"type": "TEXT"},
                                 "textFormat": {"fontSize": 9, "fontFamily": "Roboto Mono"}}}
    reqs.append({"repeatCell": {"range": G(HEAD, nrows, 0, ncols), "cell": hide, "fields": "userEnteredFormat.numberFormat"}})
    reqs.append({"repeatCell": {"range": G(0, 2, 0, ncols), "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                                "fields": "userEnteredFormat.textFormat.bold"}})
    for b in range(bands):
        lr = HEAD + b * th + N
        reqs.append({"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": "ROWS", "startIndex": lr,
                                                             "endIndex": lr + 1}, "properties": {"pixelSize": LABEL_PX},
                                                   "fields": "pixelSize"}})
        reqs.append({"repeatCell": {"range": G(lr, lr + 1, 0, ncols), "cell": lab,
                                    "fields": "userEnteredFormat(wrapStrategy,verticalAlignment,numberFormat,textFormat)"}})
        for t in range(T):
            if b * T + t < len(items):
                reqs.append({"mergeCells": {"range": G(lr, lr + 1, t * tw, t * tw + N), "mergeType": "MERGE_ALL"}})
    fills = []
    if not a.cf:   # static fills: one repeatCell per horizontal run of one colour; renders like ordinary formatting
        cols_ = [rgb(h) for h in pal]
        for n, (i, s_, r, j) in enumerate(items):
            r0, c0 = HEAD + (n // T) * th, (n % T) * tw
            g = m["grids"][j]
            for y in range(N):
                x = 0
                while x < N:
                    v, x1 = int(g[y][x]), x
                    while x1 < N and g[y][x1] == v:
                        x1 += 1
                    fills.append({"repeatCell": {"range": G(r0 + y, r0 + y + 1, c0 + x, c0 + x1),
                                                 "cell": {"userEnteredFormat": {"backgroundColor": cols_[v]}},
                                                 "fields": "userEnteredFormat.backgroundColor"}})
                    x = x1
    for k, h in enumerate(pal if a.cf else []):
        reqs.append({"addConditionalFormatRule": {"index": k, "rule": {"ranges": [G(HEAD, nrows, 0, ncols)], "booleanRule": {
            "condition": {"type": "CUSTOM_FORMULA", "values": [{"userEnteredValue": f"=AND(ISNUMBER(A{HEAD + 1}),A{HEAD + 1}={k})"}]},
            "format": {"backgroundColor": rgb(h), "textFormat": {"foregroundColor": rgb(h)}}}}}})
    for d in data_req:
        cl.values_batch_update(d, "RAW")
    for i in range(0, len(reqs), 400):
        cl.batch_update(reqs[i:i + 400])
    chunk, size = [], 0
    for q in fills:   # <= ~1.8 MB per request
        sz = len(json.dumps(q))
        if size + sz > 1_800_000 and chunk:
            cl.batch_update(chunk); chunk, size = [], 0
        chunk.append(q); size += sz
    if chunk:
        cl.batch_update(chunk)
    print(f"done: {len(items)} sprites, {cl.requests} requests, {cl.bytes_sent / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
