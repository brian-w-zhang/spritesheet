"""LibreOffice forced-recalc on big*.xlsx; compare to numpy ref. Usage: lo_big.py big100k.xlsx big100k.json
Uses make_stale-style injection first so a non-recalculating path would be caught (all cached = 999)."""
import json, os, subprocess, sys, time
import numpy as np, openpyxl
HERE = os.path.dirname(os.path.abspath(__file__))
src, ref = os.path.abspath(sys.argv[1]), json.load(open(sys.argv[2]))["ref"]
stale = src.replace(".xlsx", "_stale.xlsx")
subprocess.run([sys.executable, os.path.join(HERE, "make_stale.py"), src, stale], check=True)
prof = os.path.join(HERE, "lo_profile_always")  # created by lo_convert.py --recalc always
out = os.path.join(HERE, "lo_out_big"); os.makedirs(out, exist_ok=True)
t0 = time.time()
subprocess.run(["/opt/homebrew/bin/soffice", f"-env:UserInstallation=file://{prof}", "--headless", "--norestore",
                "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", out, stale], check=True, capture_output=True)
t1 = time.time()
ws = openpyxl.load_workbook(os.path.join(out, os.path.basename(stale)), data_only=True, read_only=True)["out"]
got = np.array([[c for c in row] for row in ws.iter_rows(values_only=True)], dtype=float)
print(json.dumps({"soffice_s": round(t1 - t0, 2), "read_s": round(time.time() - t1, 2), "n999": int((got == 999).sum()),
                  "maxAbsErr": float(np.abs(got - np.array(ref)).max())}))
