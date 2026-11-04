"""openpyxl ArrayFormula (CSE) + float round-trip check; then evaluate via LO (forced recalc) and `formulas`."""
import json, os, subprocess, sys, zipfile, re, warnings
import numpy as np, openpyxl
from openpyxl.worksheet.formula import ArrayFormula
warnings.filterwarnings("ignore")
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "S"
for r, row in enumerate([[1, 2], [3, 4]], 1):
    for c, v in enumerate(row, 1): ws.cell(r, c, v)
for r, row in enumerate([[5, 6], [7, 8]], 1):
    for c, v in enumerate(row, 4): ws.cell(r, c, v)
ws["K1"] = ArrayFormula("K1:L2", "=MMULT(A1:B2,D1:E2)")
ws["N1"] = ArrayFormula("N1", "=SUM(A1:B2*D1:E2)")
ws["P1"] = 0.1 + 0.2; ws["P2"] = float(np.exp(1)) / 3; ws["P3"] = 1e-300
wb.save("arr.xlsx")
xml = zipfile.ZipFile("arr.xlsx").read("xl/worksheets/sheet1.xml").decode()
print("cell XML:", re.findall(r'<c r="(?:K1|N1|P1|P2)".*?</c>', xml))
raw = openpyxl.load_workbook("arr.xlsx")["S"]
print("float roundtrip exact:", raw["P1"].value == 0.1 + 0.2, raw["P2"].value == float(np.exp(1)) / 3)
out = "lo_out_arr"; os.makedirs(out, exist_ok=True)
subprocess.run(["/opt/homebrew/bin/soffice", f"-env:UserInstallation=file://{os.path.abspath('lo_profile_always')}",
                "--headless", "--convert-to", "xlsx", "--outdir", out, "arr.xlsx"], check=True, capture_output=True)
s = openpyxl.load_workbook(f"{out}/arr.xlsx", data_only=True)["S"]
print("LO:", [s[c].value for c in ("K1", "L1", "K2", "L2", "N1")], "P2 back:", s["P2"].value, s["P2"].value == float(np.exp(1)) / 3)
import formulas
sol = formulas.ExcelModel().loads("arr.xlsx").finish().calculate()
print("formulas:", {k.split("!")[1]: v.value.tolist() for k, v in sol.items() if k.split("!")[-1] in ("K1:L2", "N1")})
