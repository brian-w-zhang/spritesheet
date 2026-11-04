"""Write a Pixcell Workbook to .xlsx with openpyxl (write-only). Text stays text, numbers stay numbers."""
from openpyxl import Workbook as XWorkbook
from openpyxl.cell import WriteOnlyCell

from .workbook import Text, Workbook, is_formula


def to_xlsx(wb: Workbook, path: str, overrides: dict | None = None) -> str:
    """overrides: {(sheet, row, col): value} applied on top (e.g. a different prompt/seed)."""
    xw = XWorkbook(write_only=True)
    ov = {}
    for (s, r, c), v in (overrides or {}).items():
        ov.setdefault(s, {})[(r, c)] = v
    for name in wb.sheets:
        ws = xw.create_sheet(title=name)
        cells = dict(wb.sheets[name])
        cells.update(ov.get(name, {}))
        if not cells:
            continue
        nr = max(r for r, _ in cells)
        rows = [[] for _ in range(nr)]
        for (r, c), v in cells.items():
            rows[r - 1].append((c, v))
        for row in rows:
            row.sort()
            out, last = [], 0
            for c, v in row:
                out.extend([None] * (c - last - 1))
                last = c
                if is_formula(v):
                    out.append(v)
                elif isinstance(v, str):
                    cell = WriteOnlyCell(ws, value=str(v))
                    cell.data_type = "s"          # never parsed as a formula
                    out.append(cell)
                else:
                    out.append(float(v))
            ws.append(out)
    xw.save(path)
    return path
