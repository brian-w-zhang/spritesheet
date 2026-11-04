"""Generate shared test cases: function probes + tiny MLP, as xlsx (openpyxl) and JSON (for HyperFormula)."""
import json, sys
import numpy as np
from openpyxl import Workbook
from openpyxl.utils import get_column_letter as L

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------- function probes ----------
# Data block on sheet "F": A1:B2=[[1,2],[3,4]], D1:E2=[[5,6],[7,8]], G1 text, G2 csv
DATA = {"A1": 1, "B1": 2, "A2": 3, "B2": 4, "D1": 5, "E1": 6, "D2": 7, "E2": 8,
        "G1": "Hello World", "G2": "a,b,c"}
# (name, formula, expected). Formulas in Sheets/HyperFormula syntax.
PROBES = [
    ("MMULT", "=INDEX(MMULT(A1:B2,D1:E2),2,2)", 50),
    ("SUMPRODUCT", "=SUMPRODUCT(A1:B2,D1:E2)", 70),
    ("EXP", "=EXP(1)", float(np.exp(1))),
    ("TANH", "=TANH(0.5)", float(np.tanh(0.5))),
    ("MAX", "=MAX(A1:B2,D1:E2)", 8),
    ("MAX0_relu", "=MAX(0,-2.5)", 0),
    ("LET", "=LET(x,2,y,3,x*y)", 6),
    ("LAMBDA", "=LAMBDA(x,x*2)(5)", 10),
    ("MAP", "=SUM(MAP(A1:B2,LAMBDA(x,x*x)))", 30),
    ("MAKEARRAY", "=SUM(MAKEARRAY(2,3,LAMBDA(r,c,r*c)))", 18),
    ("BYROW", "=SUM(BYROW(A1:B2,LAMBDA(r,SUM(r))))", 10),
    ("INDEX", "=INDEX(D1:E2,2,1)", 7),
    ("MATCH", "=MATCH(7,D1:D2,0)", 2),
    ("XLOOKUP", "=XLOOKUP(3,A1:A2,B1:B2)", 4),
    ("SPLIT", "=INDEX(SPLIT(G2,\",\"),1,2)", "b"),
    ("LOWER", "=LOWER(G1)", "hello world"),
    ("REGEXMATCH", "=REGEXMATCH(G1,\"W.r\")", True),
    ("ARRAYFORMULA", "=ARRAYFORMULA(SUM(A1:B2*D1:E2))", 70),
    ("ARRAY_SUM_noAF", "=SUM(A1:B2*D1:E2)", 70),
    ("SUMPRODUCT_cond", "=SUMPRODUCT((A1:B2>2)*D1:E2)", 15),
    ("ROWDOT_MMULT", "=INDEX(MMULT(A1:B1,TRANSPOSE(D1:E1)),1,1)", 17),
]
# Spill probe: K1 holds a bare MMULT; L2 should read the spilled bottom-right value.
SPILL = ("SPILL_MMULT", "K1", "=MMULT(A1:B2,D1:E2)", "L2", 50)

XLFN = ["LET", "LAMBDA", "MAP", "MAKEARRAY", "BYROW", "XLOOKUP"]


def to_xlsx_formula(f, name):
    """Excel file-format spelling: future functions need _xlfn., LET/LAMBDA params need _xlpm."""
    import re
    g = f
    for fn in XLFN:
        g = re.sub(rf"\b{fn}\(", f"_xlfn.{fn}(", g)
    if name in ("LET",):
        g = g.replace("(x,", "(_xlpm.x,").replace(",y,", ",_xlpm.y,").replace("x*y", "_xlpm.x*_xlpm.y")
    if name in ("LAMBDA", "MAP", "BYROW"):
        v = "r" if name == "BYROW" else "x"
        g = re.sub(rf"\b{v}\b", f"_xlpm.{v}", g)
    if name == "MAKEARRAY":
        g = g.replace("(r,c,r*c)", "(_xlpm.r,_xlpm.c,_xlpm.r*_xlpm.c)")
    return g


# ---------- tiny MLP 8 -> 16 -> 4, ReLU ----------
rng = np.random.default_rng(0)
NI, NH, NO = 8, 16, 4
x = rng.standard_normal(NI)
W1 = rng.standard_normal((NH, NI)) / np.sqrt(NI)
b1 = rng.standard_normal(NH) * 0.1
W2 = rng.standard_normal((NO, NH)) / np.sqrt(NH)
b2 = rng.standard_normal(NO) * 0.1
h = np.maximum(0, W1 @ x + b1)
y = W2 @ h + b2


def mlp_cells():
    """Sheet 'M' layout. Row 1: x in A1:H1. W1 rows 3..18 cols A..H, b1 col J rows 3..18.
    W2 rows 21..24 cols A..P, b2 col R rows 21..24. h in col L rows 3..18. y in col T rows 21..24.
    Hidden uses SUMPRODUCT of row with x; output uses SUMPRODUCT of row with TRANSPOSE(h) is avoided:
    instead store h also as a row (row 27, A..P) via INDEX so SUMPRODUCT is same-shape."""
    c = {}
    for i in range(NI):
        c[f"{L(i+1)}1"] = float(x[i])
    for r in range(NH):
        for k in range(NI):
            c[f"{L(k+1)}{3+r}"] = float(W1[r, k])
        c[f"J{3+r}"] = float(b1[r])
        c[f"L{3+r}"] = f"=MAX(0,SUMPRODUCT(A{3+r}:H{3+r},$A$1:$H$1)+J{3+r})"
    for j in range(NH):  # h as a row, so output SUMPRODUCT has matching shapes
        c[f"{L(j+1)}27"] = f"=INDEX($L$3:$L$18,{j+1})"
    for o in range(NO):
        for j in range(NH):
            c[f"{L(j+1)}{21+o}"] = float(W2[o, j])
        c[f"R{21+o}"] = float(b2[o])
        c[f"T{21+o}"] = f"=SUMPRODUCT(A{21+o}:P{21+o},$A$27:$P$27)+R{21+o}"
    # Alternative single-cell vectorised output with MMULT (array result in V21:V24 via INDEX)
    for o in range(NO):
        c[f"V{21+o}"] = f"=INDEX(MMULT($A$21:$P$24,$L$3:$L$18),{o+1},1)+R{21+o}"
    return c


def main():
    wb = Workbook()
    ws = wb.active
    ws.title = "F"
    for k, v in DATA.items():
        ws[k] = v
    meta = {"probes": [], "spill": None}
    for i, (name, f, exp) in enumerate(PROBES):
        r = i + 1
        ws[f"N{r}"] = name
        ws[f"O{r}"] = f                       # verbatim (Sheets spelling)
        ws[f"P{r}"] = to_xlsx_formula(f, name)  # Excel file-format spelling
        meta["probes"].append({"name": name, "plain": f"O{r}", "xlfn": f"P{r}", "formula": f, "expected": exp})
    ws[SPILL[1]] = SPILL[2]
    meta["spill"] = {"name": SPILL[0], "cell": SPILL[1], "formula": SPILL[2], "read": SPILL[3], "expected": SPILL[4]}
    m = wb.create_sheet("M")
    cells = mlp_cells()
    for k, v in cells.items():
        m[k] = v
    meta["mlp"] = {"y": y.tolist(), "h": h.tolist(), "ycells": [f"T{21+o}" for o in range(NO)],
                   "ycells_mmult": [f"V{21+o}" for o in range(NO)], "hcells": [f"L{3+r}" for r in range(NH)]}
    wb.save(f"{OUT}/tests.xlsx")
    json.dump({"data": DATA, "meta": meta, "mlp_cells": cells}, open(f"{OUT}/cases.json", "w"), indent=1)
    print("wrote tests.xlsx, cases.json; y =", y)


main()
