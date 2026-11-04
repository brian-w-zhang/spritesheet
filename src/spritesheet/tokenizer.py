"""Python twin of the in-sheet tokenizer. The FORMULAS are the source of truth; this mirrors them exactly.

Sheet pipeline (see pixcell/emit.py, Tokens tab; D-30):
  clean  = SUBSTITUTE(... SUBSTITUTE(LOWER(prompt), p1, " ") ..., pn, " ")   for p in PUNCT
  rest_1 = TRIM(clean)
  word_i = TRIM(MID(SUBSTITUTE(MID(rest_i,1,W)," ",REPT(" ",W)),1,W))      first token of rest_i (≤ W chars)
  rest_{i+1} = TRIM(SUBSTITUTE(rest_i, word_i, "", 1))                     drop that token (it is a prefix)
  id_i   = IF(word_i = "", 1, IFERROR(MATCH(word_i, Vocab!A1:A<V>, 0), 2))   i = 1..L
Ids are 1-based rows of the Vocab tab: row 1 = <pad>, row 2 = <unk>.
Effect: split on spaces (TRIM collapses runs), a token longer than W chars is cut into W-char pieces, keep
the first L pieces. (The older fixed-width split MID(…,(i-1)*W+1,W) broke when the first words summed to
≥ W characters; see docs/decisions.md D-30.)
"""
L = 12          # max words
W = 100         # max chars per word piece (MID/REPT field width)
PAD, UNK = 1, 2
# Everything that is not a-z, 0-9 or space gets replaced. MATCH wildcards * ? ~ are included on purpose.
PUNCT = list(".,;:!?\"'()[]{}<>/\\|-_+=*&^%$#@~`")


def clean(prompt: str) -> str:
    s = str(prompt).lower()
    for p in PUNCT:
        s = s.replace(p, " ")
    return s


def trim(s: str) -> str:
    """Spreadsheet TRIM: strip leading/trailing spaces (char 32 only) and collapse inner runs to one."""
    return " ".join(t for t in s.split(" ") if t != "")


def split_trace(prompt: str) -> tuple[str, list[str], list[str]]:
    """(clean, rest_1..rest_L, word_1..word_L), cell for cell as in the Tokens tab."""
    c = clean(prompt)
    rest = trim(c)
    rests, words = [], []
    for _ in range(L):
        word = rest[:W].split(" ")[0]
        rests.append(rest)
        words.append(word)
        rest = trim(rest[len(word):])
    return c, rests, words


def normalize(prompt: str) -> list[str]:
    return [w for w in split_trace(prompt)[2] if w != ""]


def load_vocab(path: str) -> dict[str, int]:
    words = [w.rstrip("\n") for w in open(path, encoding="utf-8")]
    return index_vocab(words)


def index_vocab(words: list[str]) -> dict[str, int]:
    assert words[0] == "<pad>" and words[1] == "<unk>", "vocab rows 1-2 must be <pad>, <unk>"
    idx = {}
    for i, w in enumerate(words, start=1):
        idx.setdefault(w.lower(), i)  # MATCH(…,0) is case-insensitive and returns the first match
    return idx


def encode(prompt: str, vocab: dict[str, int]) -> list[int]:
    ws = normalize(prompt)
    ids = [vocab.get(w, UNK) for w in ws]
    return ids + [PAD] * (L - len(ids))
