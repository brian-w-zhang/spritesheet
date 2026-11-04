"""HyperFormula oracle: Workbook -> JSON sheets -> tools/hf_eval.js (one node process for many prompts)."""
import json
import os
import subprocess
import tempfile
import time

from ..pixcell.workbook import Text, Workbook, is_formula

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNNER = os.path.join(ROOT, "tools", "hf_eval.js")


def _hf_value(v):
    if v is None:
        return None
    if is_formula(v):
        return v
    if isinstance(v, str):
        return "'" + v          # apostrophe = literal text in HyperFormula (no number/date/bool parsing)
    return float(v)


def to_hyperformula_json(wb: Workbook) -> dict:
    sheets = {}
    for name in wb.sheets:
        g = wb.grid(name)
        sheets[name] = [[_hf_value(v) for v in row] for row in g]
    return sheets


def run(wb: Workbook, runs: list[dict], memory_mb=12000, workdir=None) -> dict:
    """runs: [{"prompt": str|None, "seed": num|None, "set": [(sheet,row,col,value)], "read": [sheet,...]}].

    Rows/cols in "set" are 1-based. Returns the runner output (values grids are 0-based) plus wall time.
    """
    inputs = wb.meta.get("inputs", {})
    jr = []
    for r in runs:
        sets = []
        if r.get("prompt") is not None:
            s, row, col = inputs["prompt"]
            sets.append([s, row - 1, col - 1, "'" + str(r["prompt"])])
        if r.get("seed") is not None:
            s, row, col = inputs["seed"]
            sets.append([s, row - 1, col - 1, float(r["seed"])])
        for s, row, col, v in r.get("set", []):
            sets.append([s, row - 1, col - 1, _hf_value(v if not isinstance(v, str) or is_formula(v) else Text(v))])
        jr.append({"set": sets, "read": list(r.get("read") or wb.sheets.keys())})
    tmp = workdir or tempfile.mkdtemp(prefix="hf_")
    os.makedirs(tmp, exist_ok=True)
    ip, op = os.path.join(tmp, "hf_in.json"), os.path.join(tmp, "hf_out.json")
    t0 = time.time()
    with open(ip, "w") as f:
        json.dump({"sheets": to_hyperformula_json(wb), "runs": jr}, f)
    t_write = time.time() - t0
    t1 = time.time()
    p = subprocess.run(["node", f"--max-old-space-size={memory_mb}", RUNNER, ip, op], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"hf_eval.js failed: {p.stderr[-2000:]}")
    out = json.load(open(op))
    out["node_s"] = time.time() - t1
    out["json_write_s"] = t_write
    if workdir is None:
        for fp in (ip, op):
            os.remove(fp)
        os.rmdir(tmp)
    return out
