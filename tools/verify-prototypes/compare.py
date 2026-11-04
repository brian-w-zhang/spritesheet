"""Compare an engine result JSON (stdin or file) against expected values in cases.json."""
import json, sys
C = json.load(open(sys.argv[1]))
R = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else json.load(sys.stdin)
exp = {p["name"]: p["expected"] for p in C["meta"]["probes"]}
exp[C["meta"]["spill"]["name"]] = C["meta"]["spill"]["expected"]


def ok(e, g):
    if isinstance(e, bool) or isinstance(g, bool):
        return str(e).upper() == str(g).upper()
    if isinstance(e, (int, float)):
        try: return abs(float(g) - e) < 1e-12 * max(1, abs(e))
        except Exception: return False
    return str(e) == str(g)


print("##", R["engine"])
for k in exp:
    g = R["probes"].get(k, "missing")
    print(f"  {'PASS' if ok(exp[k], g) else 'FAIL'} {k:16s} got={g!r}")
for key in [k for k in R["mlp"]]:
    ys = R["mlp"][key]
    try:
        err = max(abs(float(a) - b) for a, b in zip(ys, C["meta"]["mlp"]["y"]))
        print(f"  MLP {key}: max abs err vs numpy = {err:.3e}")
    except Exception:
        print(f"  MLP {key}: FAILED values={ys}")
