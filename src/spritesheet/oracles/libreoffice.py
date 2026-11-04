"""LibreOffice oracle: xlsx -> forced-recalc headless convert -> computed values (openpyxl data_only).

LibreOffice trusts cached values in openpyxl files by default (docs/05), so we
  1. use an isolated profile whose registrymodifications.xcu sets OOXML/ODF RecalcMode = 0 (always), and
  2. plant a stale cached value 999 in every formula cell and assert none survive (scan_errors -> stale).
Many prompts: the base xlsx is written once; each variant only rewrites the Prompt sheet XML (B2/B3), and
all variants are converted in one soffice call.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from xml.sax.saxutils import escape

import openpyxl

from ..pixcell.workbook import Workbook, a1
from ..pixcell.xlsx import to_xlsx

SOFFICE = "/opt/homebrew/bin/soffice"
STALE = 999


def _profile(root: str, recalc=0) -> str:
    """recalc: 0 = always (required), 1 = never (used only to prove the stale check catches it)."""
    prof = os.path.join(root, f"lo_profile_{recalc}")
    os.makedirs(os.path.join(prof, "user"), exist_ok=True)
    items = "".join(
        f'<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="{k}" oor:op="fuse">'
        f"<value>{recalc}</value></prop></item>\n" for k in ("OOXMLRecalcMode", "ODFRecalcMode"))
    with open(os.path.join(prof, "user", "registrymodifications.xcu"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<oor:items xmlns:oor="http://openoffice.org/2001/registry" '
                'xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
                + items + "</oor:items>\n")
    return prof


_F = re.compile(r"(<f>[^<]*</f>)(<v\s*/>|<v>[^<]*</v>)?")


def plant_stale(src: str, dst: str, prompt_sheet_index=None, cells=None) -> int:
    """Copy xlsx, writing <v>999</v> into every formula cell. Optionally replace input cells of one sheet:
    cells = {"B2": ("str", text) | ("num", x)} in xl/worksheets/sheet<prompt_sheet_index>.xml."""
    n = 0
    zin = zipfile.ZipFile(src)
    zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=1)
    target = f"xl/worksheets/sheet{prompt_sheet_index}.xml" if prompt_sheet_index else None
    for it in zin.infolist():
        data = zin.read(it.filename)
        if it.filename.startswith("xl/worksheets/sheet"):
            s = data.decode()
            s, k = _F.subn(rf"\g<1><v>{STALE}</v>", s)
            n += k
            if it.filename == target and cells:
                for addr, (kind, v) in cells.items():
                    if kind == "str":
                        new = f'<c r="{addr}" t="inlineStr"><is><t xml:space="preserve">{escape(v)}</t></is></c>'
                    else:
                        new = f'<c r="{addr}" t="n"><v>{repr(float(v))}</v></c>'
                    s, k2 = re.subn(rf'<c r="{addr}"[^>]*?(/>|>.*?</c>)', lambda _m: new, s, count=1)
                    assert k2 == 1, f"input cell {addr} not found"
            data = s.encode()
        zout.writestr(it, data)
    zout.close()
    return n


def read_values(path: str, sheets=None) -> dict:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    out = {}
    for name in (sheets or wb.sheetnames):
        ws = wb[name]
        out[name] = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return out


def run(wb: Workbook, runs: list[dict], workdir=None, keep=False, timeout=3600, recalc=0) -> dict:
    """runs: [{"prompt", "seed", "read": [sheets]}]. Returns {"runs": [{"values"}], timings, stale_planted}."""
    tmp = workdir or tempfile.mkdtemp(prefix="lo_")
    os.makedirs(tmp, exist_ok=True)
    prof = _profile(tmp, recalc)
    t0 = time.time()
    base = os.path.join(tmp, "base.xlsx")
    to_xlsx(wb, base)
    t_xlsx = time.time() - t0
    names = list(wb.sheets)
    inputs = wb.meta.get("inputs")
    if inputs:
        ps, pr, pc = inputs["prompt"]
        ss, sr, sc = inputs["seed"]
        assert ps == ss
        idx = names.index(ps) + 1
    else:
        idx = None
    indir, outdir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    for dd in (indir, outdir):
        shutil.rmtree(dd, ignore_errors=True)
        os.makedirs(dd)
    t1 = time.time()
    files, planted = [], 0
    for k, r in enumerate(runs):
        cells = {}
        if r.get("prompt") is not None or r.get("seed") is not None:
            assert inputs, "workbook has no Prompt inputs"
        if r.get("prompt") is not None:
            cells[a1(pr, pc)] = ("str", str(r["prompt"]))
        if r.get("seed") is not None:
            cells[a1(sr, sc)] = ("num", r["seed"])
        fp = os.path.join(indir, f"run{k:04d}.xlsx")
        planted = plant_stale(base, fp, idx, cells)
        files.append(fp)
    t_variants = time.time() - t1
    t2 = time.time()
    p = subprocess.run([SOFFICE, f"-env:UserInstallation=file://{prof}", "--headless", "--norestore",
                        "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", outdir] + files,
                       capture_output=True, text=True, timeout=timeout)
    t_convert = time.time() - t2
    t3 = time.time()
    res = []
    for k, (r, fp) in enumerate(zip(runs, files)):
        op = os.path.join(outdir, os.path.basename(fp))
        if not os.path.exists(op):
            raise RuntimeError(f"LibreOffice produced no output for run {k}: {p.stderr[-1500:]}")
        res.append({"values": read_values(op, r.get("read"))})
    t_read = time.time() - t3
    if not keep and workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"runs": res, "stale_planted": planted, "xlsx_s": t_xlsx, "variants_s": t_variants,
            "convert_s": t_convert, "read_s": t_read, "version": _version()}


_V = None


def _version():
    global _V
    if _V is None:
        try:
            _V = subprocess.run([SOFFICE, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as ex:  # noqa: BLE001
            _V = f"unknown ({ex})"
    return _V
