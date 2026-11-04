"""Local formula-engine oracles (HyperFormula, LibreOffice) for Pixcell workbooks."""
from ..pixcell.workbook import is_formula

ERROR_PREFIXES = ("#ERR", "#NAME", "#VALUE", "#REF", "#NUM", "#DIV/0", "#N/A", "#NULL", "Err:")


def scan_errors(wb, values: dict) -> dict:
    """Count formula cells whose engine value is an error or a planted stale 999. values grids are 0-based."""
    errors, stale, examples = 0, 0, []
    for name, cells in wb.sheets.items():
        g = values.get(name)
        if g is None:
            continue
        for (r, c), v in cells.items():
            if not is_formula(v):
                continue
            try:
                x = g[r - 1][c - 1]
            except IndexError:
                x = None
            if isinstance(x, str) and x.startswith(ERROR_PREFIXES):
                errors += 1
                if len(examples) < 5:
                    examples.append((name, r, c, x, v[:120]))
            elif isinstance(x, (int, float)) and not isinstance(x, bool) and x == 999:
                stale += 1
    return {"errors": errors, "stale": stale, "examples": examples}
