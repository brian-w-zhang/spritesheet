"""H: every number inside <!-- claims --> ... <!-- /claims --> blocks of README.md / PROGRESS.md must be
backed by a numeric value in artifacts/metrics/*.json (same value at the displayed precision; percentages
may be shown as value×100; thousands separators allowed). Exit 1 on any unbacked number.

Numbers that are not claims (dates, IDs) belong outside the claims blocks.
"""
import glob, json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["README.md", "PROGRESS.md"]
NUM = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+|-?\d+(?:\.\d+)?)(%|×|x)?(?![\w])")


def leaves(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        if math.isfinite(o):
            yield float(o)
    elif isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)


def backed(txt, suffix, values):
    x = float(txt.replace(",", ""))
    dec = len(txt.split(".")[1]) if "." in txt else 0
    tol = 0.5 * 10 ** (-dec) + 1e-12
    cands = []
    for v in values:
        cands.append(v)
        if suffix == "%" or dec <= 2:
            cands.append(v * 100)
    return any(abs(c - x) <= tol for c in cands)


def main():
    values = []
    for p in glob.glob(os.path.join(ROOT, "artifacts", "metrics", "*.json")):
        try:
            values.extend(leaves(json.load(open(p))))
        except Exception as e:
            print("unreadable", p, e)
    bad, n = [], 0
    for f in FILES:
        p = os.path.join(ROOT, f)
        if not os.path.exists(p):
            continue
        for block in re.findall(r"<!-- claims -->(.*?)<!-- /claims -->", open(p).read(), flags=re.S):
            for m in NUM.finditer(block):
                n += 1
                if not backed(m.group(1), m.group(2), values):
                    bad.append(f"{f}: {m.group(0)}")
    print(f"checked {n} numbers in claims blocks against {len(values)} metric values")
    if bad:
        print("UNBACKED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("PASS: all claimed numbers are backed by artifacts/metrics")


if __name__ == "__main__":
    main()
