"""Does MMULT recalculate faster than per-cell SUMPRODUCT in Google Sheets? (MP3 speed experiment, docs/09)

  uv run python scripts/bench_mmult_sheets.py [--sheet probe] [--rows 64] [--dm 128] [--layers 6] [--reps 3]

The same computation is built three ways on separate tabs of the probe (scratch) spreadsheet, each with its own input
cell, so changing one input recalculates only that version:
  A  "BenchA":  one SUMPRODUCT per output cell (how Pixcell emits linear layers today): rows*dm*layers formulas
  B  "BenchB":  one MMULT per row per layer (row vector x W, spills dm cells): rows*layers formulas. Keeps rows
                independent, so it also works for the causal (row-by-row) model.
  C  "BenchC":  one MMULT per layer over the whole block: layers formulas (upper bound; not usable for the causal chain)
Layer l+1 = layer l x W (W = random dm x dm / sqrt(dm), values on "BenchW"). Layer 0 row n, col j = in * (n+1+j)/1000.
Timing: write a new input value via the API, then poll the version's checksum cell (SUM of the last layer) until it
equals the numpy value (rel 1e-9). This is API-observed server recalculation; the A/B/C comparison is like for like.
Writes artifacts/metrics/bench_mmult_sheets.json. Leaves the Bench* tabs in place (delete with --cleanup).
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402

from spritesheet.pixcell import sheets  # noqa: E402
from spritesheet.pixcell.workbook import a1, rng  # noqa: E402

TABS = ["BenchW", "BenchWT", "BenchA", "BenchB", "BenchC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="probe")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--dm", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--versions", default="ABC")
    ap.add_argument("--print-expect", action="store_true", help="print expected checksums (browser timing) and stop")
    ap.add_argument("--cleanup", action="store_true")
    a = ap.parse_args()
    if a.sheet not in ("probe",):
        sys.exit("benchmark tabs go on the probe scratch sheet only")
    cl = sheets.Client(a.sheet)
    have = {s["properties"]["title"]: s["properties"]["sheetId"] for s in cl.meta()["sheets"]}
    if a.cleanup:
        cl.batch_update([{"deleteSheet": {"sheetId": have[t]}} for t in TABS if t in have])
        print("deleted", [t for t in TABS if t in have])
        return
    N, dm, L = a.rows, a.dm, a.layers
    W = np.random.default_rng(0).normal(0, 1, (dm, dm)) / np.sqrt(dm)
    blk_r0 = lambda l: 3 + l * (N + 1)          # layer l occupies rows blk_r0(l) .. +N-1 (1-based), cols 1..dm

    if not a.skip_build:
        add = [{"addSheet": {"properties": {"title": t, "gridProperties": {"rowCount": max(dm, blk_r0(L) + N) + 2,
                                                                             "columnCount": dm + 2}}}}
               for t in TABS if t not in have]
        if add:
            cl.batch_update(add)
        data = [{"range": f"BenchW!{rng(1, 1, dm, dm)}", "values": W.tolist()},
                {"range": f"BenchWT!{rng(1, 1, dm, dm)}", "values": W.T.tolist()}]   # row j = column j of W (for A)
        cl.values_batch_update(data, "RAW")
        for v in a.versions:
            t = f"Bench{v}"
            rows = [["in", 1.0, "checksum", f"=SUM({rng(blk_r0(L), 1, blk_r0(L) + N - 1, dm)})", "expect", 0.0,
                     "=IF(ABS(D1-F1)<=0.000000001*MAX(1,ABS(F1)),1,0)"]]
            # layer 0
            l0 = [[f"=$B$1*{n + 1 + j}/1000" for j in range(dm)] for n in range(N)]
            data = [{"range": f"{t}!A1:G1", "values": rows},
                    {"range": f"{t}!{rng(blk_r0(0), 1, blk_r0(0) + N - 1, dm)}", "values": l0}]
            for l in range(1, L + 1):
                r0, p0 = blk_r0(l), blk_r0(l - 1)
                if v == "A":
                    blk = [[f"=SUMPRODUCT({rng(p0 + n, 1, p0 + n, dm)},{rng(1 + j, 1, 1 + j, dm, 'BenchWT', True)})"
                            for j in range(dm)] for n in range(N)]
                    data.append({"range": f"{t}!{rng(r0, 1, r0 + N - 1, dm)}", "values": blk})
                elif v == "B":
                    col = [[f"=MMULT({rng(p0 + n, 1, p0 + n, dm)},{rng(1, 1, dm, dm, 'BenchW', True)})"] for n in range(N)]
                    data.append({"range": f"{t}!{rng(r0, 1, r0 + N - 1, 1)}", "values": col})
                else:
                    data.append({"range": f"{t}!{a1(r0, 1)}",
                                 "values": [[f"=MMULT({rng(p0, 1, p0 + N - 1, dm)},{rng(1, 1, dm, dm, 'BenchW', True)})"]]})
            # send in chunks (<= ~1.5 MB each)
            chunk, size = [], 0
            for d in data:
                s = len(json.dumps(d))
                if chunk and size + s > 1_500_000:
                    cl.values_batch_update(chunk, "USER_ENTERED"); chunk, size = [], 0
                chunk.append(d); size += s
            if chunk:
                cl.values_batch_update(chunk, "USER_ENTERED")
            sid = {x["properties"]["title"]: x["properties"]["sheetId"] for x in cl.meta()["sheets"]}[t]
            cell = {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 6, "endColumnIndex": 7}
            cl.batch_update([{"addConditionalFormatRule": {"index": 0, "rule": {"ranges": [cell], "booleanRule": {
                "condition": {"type": "NUMBER_EQ", "values": [{"userEnteredValue": str(k)}]},
                "format": {"backgroundColor": col}}}}} for k, col in
                ((1, {"green": 0.8, "red": 0.1, "blue": 0.1}), (0, {"red": 0.9, "green": 0.1, "blue": 0.1}))])
            print("built", t, flush=True)

    def expect(x):
        X = np.array([[x * (n + 1 + j) / 1000 for j in range(dm)] for n in range(N)])
        for _ in range(L):
            X = X @ W
        return float(X.sum())

    if a.print_expect:
        for x in [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]:
            print(x, repr(expect(x)))
        return
    res = {v: [] for v in "ABC"}
    vals = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0][: a.reps]
    for rep, x in enumerate(vals):
        for v in "ABC":
            t = f"Bench{v}"
            want = expect(x)
            t0 = time.time()
            cl.values_batch_update([{"range": f"{t}!B1", "values": [[x]]}], "RAW")
            reads = 0
            while True:
                got = cl.values_batch_get([f"{t}!D1"])["valueRanges"][0].get("values", [[None]])[0][0]
                reads += 1
                if isinstance(got, (int, float)) and abs(got - want) <= 1e-9 * max(1.0, abs(want)):
                    break
                if time.time() - t0 > 900:
                    raise SystemExit(f"timeout {t}")
                time.sleep(0.25)
            dt = time.time() - t0
            res[v].append({"x": x, "seconds": round(dt, 2), "reads": reads})
            print(v, res[v][-1], flush=True)
    out = {"rows": N, "dm": dm, "layers": L, "formulas": {"A": N * dm * L, "B": N * L, "C": L},
           "runs": res, "median_s": {v: float(np.median([r["seconds"] for r in res[v]])) for v in "ABC"},
           "method": "API write of the input cell, then API polling of SUM(last layer) until it matches numpy",
           "time": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "metrics", "bench_mmult_sheets.json"), "w"), indent=1)
    print(json.dumps(out["median_s"]))


if __name__ == "__main__":
    main()
