"""Scale test: A sheet 'act' (NA rows x K), 'W' (R rows x K), 'out' NA x R grid of
=SUMPRODUCT(W!$A{r}:${K}{r}, act!$A{a}:${K}{a}). Writes big.xlsx, big.json (formulas + numpy ref)."""
import json, sys
import numpy as np
from openpyxl import Workbook
from openpyxl.utils import get_column_letter as L

NA, R, K = (int(v) for v in (sys.argv[1:4] if len(sys.argv) > 3 else (400, 256, 64)))
rng = np.random.default_rng(1)
A = rng.standard_normal((NA, K)); W = rng.standard_normal((R, K))
ref = A @ W.T
kc = L(K)
wb = Workbook(write_only=True)
sa = wb.create_sheet("act"); [sa.append(list(map(float, row))) for row in A]
sw = wb.create_sheet("W"); [sw.append(list(map(float, row))) for row in W]
so = wb.create_sheet("out")
formulas = []
for a in range(NA):
    row = [f"=SUMPRODUCT(W!$A${r+1}:${kc}${r+1},act!$A${a+1}:${kc}${a+1})" for r in range(R)]
    formulas.append(row); so.append(row)
wb.save("big.xlsx")
json.dump({"act": A.tolist(), "W": W.tolist(), "out": formulas, "ref": ref.tolist()}, open("big.json", "w"))
print(f"{NA*R} formula cells, {NA*R*K/1e6:.1f}M MACs, {NA*K+R*K} data cells")
