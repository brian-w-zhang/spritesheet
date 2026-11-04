"""Inject bogus cached values (<v>999</v>) into every formula cell of an xlsx, to prove an engine recalculates.
Usage: make_stale.py in.xlsx out.xlsx"""
import re, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
n = 0
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename.startswith("xl/worksheets/sheet"):
        s = data.decode()
        # openpyxl writes <c r="X"><f>...</f><v></v></c> (or <v/>); replace/insert value with 999
        s, k = re.subn(r"(<f>[^<]*</f>)(<v\s*/>|<v></v>)?", r"\g<1><v>999</v>", s)
        n += k
        data = s.encode()
    zout.writestr(it, data)
zout.close()
print(f"injected stale 999 into {n} formula cells")
