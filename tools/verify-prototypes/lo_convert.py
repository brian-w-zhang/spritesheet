"""LibreOffice headless oracle without UNO (bundled LO python is SIGKILLed on this Mac).
Usage: .venv/bin/python lo_convert.py cases.json tests.xlsx [--xlfn] [--recalc always|never|default]
Writes an isolated LO profile whose registrymodifications.xcu forces OOXML/ODS recalc-on-load,
converts xlsx->xlsx with soffice, reads the computed cached values back with openpyxl(data_only)."""
import json, os, shutil, subprocess, sys, time
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(sys.argv[1]))
src = os.path.abspath(sys.argv[2])
column = "xlfn" if "--xlfn" in sys.argv else "plain"
mode = sys.argv[sys.argv.index("--recalc") + 1] if "--recalc" in sys.argv else "always"
MODE = {"always": 0, "never": 1, "default": None}[mode]

prof = os.path.join(HERE, f"lo_profile_{mode}")
xcu_dir = os.path.join(prof, "user")
os.makedirs(xcu_dir, exist_ok=True)
items = ""
if MODE is not None:
    for key in ("OOXMLRecalcMode", "ODFRecalcMode"):
        items += (f'<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="{key}" oor:op="fuse">'
                  f'<value>{MODE}</value></prop></item>\n')
open(os.path.join(xcu_dir, "registrymodifications.xcu"), "w").write(
    '<?xml version="1.0" encoding="UTF-8"?>\n<oor:items xmlns:oor="http://openoffice.org/2001/registry" '
    'xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
    + items + "</oor:items>\n")

outdir = os.path.join(HERE, f"lo_out_{mode}")
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)
t0 = time.time()
subprocess.run(["/opt/homebrew/bin/soffice", f"-env:UserInstallation=file://{prof}", "--headless", "--norestore",
                "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", outdir, src],
               check=True, capture_output=True)
dt = time.time() - t0
wb = openpyxl.load_workbook(os.path.join(outdir, os.path.basename(src)), data_only=True)


def val(sheet, ref):
    v = wb[sheet][ref].value
    return v


out = {"engine": f"libreoffice-convert ({column}, recalc={mode})", "probes": {}, "mlp": {}, "seconds": dt}
for p in C["meta"]["probes"]:
    out["probes"][p["name"]] = val("F", p[column])
out["probes"][C["meta"]["spill"]["name"]] = val("F", C["meta"]["spill"]["read"])
out["mlp"]["y"] = [val("M", r) for r in C["meta"]["mlp"]["ycells"]]
out["mlp"]["y_mmult"] = [val("M", r) for r in C["meta"]["mlp"]["ycells_mmult"]]
print(json.dumps(out, default=str))
