"""Static checks on a compiled Workbook: only the allowed function subset (docs/05), MATCH always exact (…,0),
SPARKLINE only in the Prompt display, LN only in v2 S{t}_Score tabs, no array literals outside it, sheet names that cannot be read as cell refs."""
import re

from .workbook import is_formula

CORE = {"SUMPRODUCT", "INDEX", "MATCH", "MAX", "MIN", "SUM", "EXP", "IF", "IFERROR", "MOD", "INT", "ABS"}
TOKENIZER = {"LOWER", "SUBSTITUTE", "TRIM", "MID", "REPT"}
RENDER = {"SPARKLINE"}
SAMPLING = {"LN"}                     # v2 Gumbel noise, only in S{t}_Score tabs (cfg temp > 0; parity D-43)
_SCORE_TAB = re.compile(r"^(S[0-9]+_)?Score$")   # v2 S{t}_Score, v3-P Score (D-52)
NORM = {"SQRT"}                       # v3-P RMSNorm 1/SQRT(mean(x^2)+eps) (D-52)
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
    render_blocks = [b for k, b in wb.blocks.items() if k in ("Display", "Progress")]
    for sheet, cells in wb.sheets.items():
        if _CELLISH.match(sheet) or sheet.upper() in ("R", "C"):
            bad.append((sheet, None, "sheet name looks like a cell reference"))
        for (r, c), v in cells.items():
            if not is_formula(v):
                continue
            in_disp = any(sheet == b.sheet and b.r0 <= r < b.r0 + b.nrows and b.c0 <= c < b.c0 + b.ncols
                          for b in render_blocks)
            body = _STR.sub('""', v)
            allowed = CORE | TOKENIZER | (NORM if wb.meta.get("model") in ("v3p", "v3pg") else set()) | (RENDER if in_disp else set()) | (SAMPLING if _SCORE_TAB.match(sheet) else set())
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
