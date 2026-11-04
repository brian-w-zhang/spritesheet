"""Evaluate tests.xlsx with Python engines. Usage: py_run.py cases.json tests.xlsx (pycel|formulas) [--xlfn]"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
C = json.load(open(sys.argv[1]))
path, eng = sys.argv[2], sys.argv[3]
column = "xlfn" if "--xlfn" in sys.argv else "plain"
targets = [("F", p[column], p["name"]) for p in C["meta"]["probes"]]
targets.append(("F", C["meta"]["spill"]["read"], C["meta"]["spill"]["name"]))
ys = [("M", r) for r in C["meta"]["mlp"]["ycells"]]
yms = [("M", r) for r in C["meta"]["mlp"]["ycells_mmult"]]
out = {"engine": f"{eng} ({column})", "probes": {}, "mlp": {}}
t0 = time.time()


def safe(f):
    try:
        v = f()
        if hasattr(v, "tolist"):
            v = v.tolist()
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        return v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)
    except Exception as e:
        return f"EXC:{type(e).__name__}:{str(e)[:80]}"


if eng == "pycel":
    from pycel import ExcelCompiler
    xc = ExcelCompiler(filename=path)
    ev = lambda s, r: safe(lambda: xc.evaluate(f"{s}!{r}"))
else:
    import formulas
    xl = formulas.ExcelModel().loads(path).finish()
    sol = xl.calculate()
    import os
    book = os.path.basename(path)
    def get(s, r):
        key = f"'[{book}]{s}'!{r}"
        v = sol[key].value
        return v
    ev = lambda s, r: safe(lambda: get(s, r))
for s, r, n in targets:
    out["probes"][n] = ev(s, r)
out["mlp"]["y"] = [ev(s, r) for s, r in ys]
out["mlp"]["y_mmult"] = [ev(s, r) for s, r in yms]
out["seconds"] = time.time() - t0
print(json.dumps(out, default=str))
