"""Run with LibreOffice's bundled python:
  /Applications/LibreOffice.app/Contents/Resources/python lo_run.py cases.json tests.xlsx [--no-recalc]
Starts a private headless soffice (isolated profile), loads the xlsx, calculateAll(), reads values."""
import json, os, subprocess, sys, time, uuid
import uno
from com.sun.star.beans import PropertyValue

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(sys.argv[1]))
path = os.path.abspath(sys.argv[2])
recalc = "--no-recalc" not in sys.argv
column = "xlfn" if "--xlfn" in sys.argv else "plain"
pipe = "lo" + uuid.uuid4().hex[:8]
prof = "file://" + os.path.join(HERE, "lo_profile")
proc = subprocess.Popen(["/opt/homebrew/bin/soffice", f"-env:UserInstallation={prof}", "--headless", "--invisible",
                         "--nologo", "--norestore", f"--accept=pipe,name={pipe};urp;"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ctx = uno.getComponentContext()
resolver = ctx.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", ctx)
for _ in range(100):
    try:
        rctx = resolver.resolve(f"uno:pipe,name={pipe};urp;StarOffice.ComponentContext"); break
    except Exception:
        time.sleep(0.2)
desktop = rctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", rctx)


def pv(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v; return p


t0 = time.time()
doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(path), "_blank", 0, (pv("Hidden", True),))
t1 = time.time()
if recalc:
    doc.calculateAll()
t2 = time.time()
sheets = doc.Sheets


def val(sheet, ref):
    c = sheets.getByName(sheet).getCellRangeByName(ref)
    if c.getError():
        return f"ERR:{c.getError()}:{c.getFormula()}"
    t = c.Type.value  # EMPTY VALUE TEXT FORMULA
    if t == "FORMULA":
        rt = c.FormulaResultType2.value
        return c.getString() if rt == "TEXT" else c.getValue()
    return c.getValue() if t == "VALUE" else c.getString()


out = {"engine": f"libreoffice-uno ({column}, recalc={recalc})", "probes": {}, "mlp": {},
       "timing": {"load": t1 - t0, "calc": t2 - t1}}
for p in C["meta"]["probes"]:
    out["probes"][p["name"]] = val("F", p[column])
out["probes"][C["meta"]["spill"]["name"]] = val("F", C["meta"]["spill"]["read"])
out["mlp"]["y"] = [val("M", r) for r in C["meta"]["mlp"]["ycells"]]
out["mlp"]["y_mmult"] = [val("M", r) for r in C["meta"]["mlp"]["ycells_mmult"]]
if "--formulas" in sys.argv:  # show how LO parsed the formulas
    out["parsed"] = {p["name"]: sheets.getByName("F").getCellRangeByName(p[column]).getFormula() for p in C["meta"]["probes"]}
doc.close(True)
try:
    desktop.terminate()
except Exception:
    pass
proc.wait(timeout=20)
print(json.dumps(out))
