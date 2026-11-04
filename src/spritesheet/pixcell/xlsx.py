"""Write a Pixcell Workbook to .xlsx with openpyxl (write-only). Text stays text, numbers stay numbers."""
from openpyxl import Workbook as XWorkbook
from openpyxl.cell import WriteOnlyCell

from .workbook import Text, Workbook, col_letter, is_formula, rng

PIXEL_COL_WIDTH = 1.71   # Excel width units (max digit width 7 px): 12 px / 7 = 1.71
PIXEL_ROW_PT = 9.0       # 12 px


def _render_prompt(ws, wb: Workbook):
    """Coloured display (+ v2 filmstrip) in the xlsx: square 12 px cells, digits hidden (';;;'), one CellIsRule per
    class, the same colours as the Sheets build. Must run before any row is appended (write-only mode)."""
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import PatternFill

    from .emit import render_spec
    sp = render_spec(wb)
    for a, b in sp["pixel_cols"]:
        for c in range(a, b + 1):
            ws.column_dimensions[col_letter(c)].width = PIXEL_COL_WIDTH
    ws.column_dimensions["B"].width = 45
    ws.sheet_view.showGridLines = False
    sqref = " ".join(rng(r0, c0, r0 + n - 1, c0 + n - 1) for r0, c0, n in sp["grids"])
    for k, hx in sp["classes"]:
        h = hx.lstrip("#").upper()
        ws.conditional_formatting.add(sqref, CellIsRule(operator="equal", formula=[str(k)], fill=PatternFill(
            start_color="FF" + h, end_color="FF" + h, fill_type="solid")))
    hide = set()
    for r0, c0, n in sp["grids"]:
        hide.update((r, c) for r in range(r0, r0 + n) for c in range(c0, c0 + n))
        for r in range(r0, r0 + n):
            ws.row_dimensions[r].height = PIXEL_ROW_PT
    return hide


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
        hide = _render_prompt(ws, wb) if (name == "Prompt" and wb.meta.get("render") and "display" in wb.meta) else ()
        nr = max(r for r, _ in cells)
        rows = [[] for _ in range(nr)]
        for (r, c), v in cells.items():
            rows[r - 1].append((c, v))
        for ri, row in enumerate(rows, start=1):
            row.sort()
            out, last = [], 0
            for c, v in row:
                out.extend([None] * (c - last - 1))
                last = c
                if is_formula(v) and (ri, c) in hide:
                    cell = WriteOnlyCell(ws, value=v)
                    cell.number_format = ";;;"
                    out.append(cell)
                elif is_formula(v):
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
