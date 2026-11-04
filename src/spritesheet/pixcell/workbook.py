"""In-memory workbook for Pixcell: {sheet_name: {(row, col): value}} with 1-based rows/cols.

Cell values:
  * float / int        -> a number
  * Text(str)          -> literal text (never parsed: Vocab words, palette hex, the prompt)
  * str starting "="   -> a formula
Blocks record where each tensor lives, so oracles can compare a whole tab region to numpy.
"""
from dataclasses import dataclass, field


class Text(str):
    """A literal string cell. Written RAW / as a shared string / with a leading apostrophe in HyperFormula."""


def col_letter(c: int) -> str:
    s = ""
    while c > 0:
        c, r = divmod(c - 1, 26)
        s = chr(65 + r) + s
    return s


def col_index(letters: str) -> int:
    c = 0
    for ch in letters:
        c = c * 26 + ord(ch) - 64
    return c


def a1(row: int, col: int, abs_row=False, abs_col=False) -> str:
    return f"{'$' if abs_col else ''}{col_letter(col)}{'$' if abs_row else ''}{row}"


def rng(r0: int, c0: int, r1: int, c1: int, sheet: str | None = None, absolute=False) -> str:
    """Range string r0,c0 .. r1,c1 inclusive (1-based); optional sheet prefix."""
    s = f"{a1(r0, c0, absolute, absolute)}:{a1(r1, c1, absolute, absolute)}"
    return f"{sheet}!{s}" if sheet else s


def ref(row: int, col: int, sheet: str | None = None, absolute=False) -> str:
    s = a1(row, col, absolute, absolute)
    return f"{sheet}!{s}" if sheet else s


@dataclass
class Block:
    """A rectangular region holding one tensor (2-D view) on one sheet."""
    sheet: str
    r0: int
    c0: int
    nrows: int
    ncols: int

    def cell(self, i: int, j: int) -> tuple[int, int]:  # 0-based within the block
        return self.r0 + i, self.c0 + j

    def ref(self, i: int, j: int, sheet: str | None = None, absolute=False) -> str:
        r, c = self.cell(i, j)
        return ref(r, c, sheet, absolute)

    def row_range(self, i: int, j0=0, j1=None, sheet=None, absolute=False) -> str:
        j1 = self.ncols - 1 if j1 is None else j1
        return rng(self.r0 + i, self.c0 + j0, self.r0 + i, self.c0 + j1, sheet, absolute)

    def col_range(self, j: int, sheet=None, absolute=False) -> str:
        return rng(self.r0, self.c0 + j, self.r0 + self.nrows - 1, self.c0 + j, sheet, absolute)

    def range(self, sheet=None, absolute=False) -> str:
        return rng(self.r0, self.c0, self.r0 + self.nrows - 1, self.c0 + self.ncols - 1, sheet, absolute)


@dataclass
class Workbook:
    sheets: dict = field(default_factory=dict)          # name -> {(r, c): value}
    blocks: dict = field(default_factory=dict)          # name -> Block
    meta: dict = field(default_factory=dict)            # free-form: config, render cells, formatting hints

    def sheet(self, name: str) -> dict:
        return self.sheets.setdefault(name, {})

    def set(self, sheet: str, row: int, col: int, value):
        self.sheet(sheet)[(row, col)] = value

    def add_block(self, name: str, sheet: str, r0: int, c0: int, nrows: int, ncols: int) -> Block:
        b = Block(sheet, r0, c0, nrows, ncols)
        self.sheet(sheet)
        self.blocks[name] = b
        return b

    def fill(self, block: Block, fn):
        """fn(i, j) -> cell value, for every cell of the block."""
        s = self.sheet(block.sheet)
        for i in range(block.nrows):
            for j in range(block.ncols):
                s[(block.r0 + i, block.c0 + j)] = fn(i, j)

    def put_matrix(self, block: Block, M):
        s = self.sheet(block.sheet)
        for i in range(block.nrows):
            row = M[i]
            for j in range(block.ncols):
                s[(block.r0 + i, block.c0 + j)] = float(row[j])

    def dims(self, sheet: str) -> tuple[int, int]:
        cells = self.sheets[sheet]
        if not cells:
            return 1, 1
        return max(r for r, _ in cells), max(c for _, c in cells)

    def grid(self, sheet: str):
        """Dense row-major list of lists (None for empty), for writers."""
        nr, nc = self.dims(sheet)
        g = [[None] * nc for _ in range(nr)]
        for (r, c), v in self.sheets[sheet].items():
            g[r - 1][c - 1] = v
        return g

    def stats(self) -> dict:
        n = f = chars = 0
        per = {}
        for name, cells in self.sheets.items():
            k = sum(1 for v in cells.values() if is_formula(v))
            per[name] = {"cells": len(cells), "formulas": k}
            n += len(cells)
            f += k
            chars += sum(len(v) for v in cells.values() if is_formula(v))
        return {"cells": n, "formulas": f, "formula_chars": chars, "sheets": len(self.sheets), "per_sheet": per}


def is_formula(v) -> bool:
    return isinstance(v, str) and not isinstance(v, Text) and v.startswith("=")
