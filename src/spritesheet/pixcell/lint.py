"""Static checks on a compiled Workbook: only the allowed function subset (docs/05), MATCH always exact (…,0),
SPARKLINE only in the Prompt display, no array literals outside it, sheet names that cannot be read as cell refs."""
import re

from .workbook import is_formula

CORE = {"SUMPRODUCT", "INDEX", "MATCH", "MAX", "MIN", "SUM", "EXP", "IF", "IFERROR", "MOD", "INT", "ABS"}
TOKENIZER = {"LOWER", "SUBSTITUTE", "TRIM", "MID", "REPT"}
RENDER = {"SPARKLINE"}
_STR = re.compile(r'"(?:[^"]|"")*"')
_FN = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\(")
_CELLISH = re.compile(r"^[A-Za-z]{1,3}[0-9]+$")


def _args(s: str, start: int) -> list[str]:
    """Top-level comma-separated args of the call whose '(' is at s[start]."""
    depth, cur, out = 0, [], []
    for ch in s[start:]:
        if ch == "(":
            depth += 1
            if depth == 1:
                continue
        elif ch == ")":
            depth -= 1
            if depth == 0:
                out.append("".join(cur))
                return out
        elif ch == "," and depth == 1:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    raise ValueError("unbalanced parentheses")


def lint(wb) -> dict:
    used, bad = {}, []
    disp = wb.blocks.get("Display")
    for sheet, cells in wb.sheets.items():
        if _CELLISH.match(sheet) or sheet.upper() in ("R", "C"):
            bad.append((sheet, None, "sheet name looks like a cell reference"))
        for (r, c), v in cells.items():
            if not is_formula(v):
                continue
            in_disp = disp is not None and sheet == disp.sheet and disp.r0 <= r < disp.r0 + disp.nrows \
                and disp.c0 <= c < disp.c0 + disp.ncols
            body = _STR.sub('""', v)
            allowed = CORE | TOKENIZER | (RENDER if in_disp else set())
            if "{" in body and not in_disp:
                bad.append((sheet, (r, c), "array literal"))
            for m in _FN.finditer(body):
                fn = m.group(1).upper()
                used[fn] = used.get(fn, 0) + 1
                if fn not in allowed:
                    bad.append((sheet, (r, c), f"function {fn} not allowed here"))
                if fn == "MATCH":
                    a = _args(body, m.end() - 1)
                    if len(a) != 3 or a[2].strip() != "0":
                        bad.append((sheet, (r, c), "MATCH without exact match type 0"))
    return {"functions": used, "violations": bad}
