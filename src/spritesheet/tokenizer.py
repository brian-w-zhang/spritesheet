"""Python twin of the in-sheet tokenizer. The FORMULAS are the source of truth; this mirrors them exactly.

Sheet pipeline (see pixcell/emit.py):
  clean = SUBSTITUTE(... SUBSTITUTE(LOWER(prompt), p1, " ") ..., pn, " ")   for p in PUNCT
  word_i = TRIM(MID(SUBSTITUTE(TRIM(clean), " ", REPT(" ", W)), (i-1)*W+1, W))   i = 1..L
  id_i  = IF(word_i = "", 1, IFERROR(MATCH(word_i, Vocab!A1:A<V>, 0), 2))
Ids are 1-based rows of the Vocab tab: row 1 = <pad>, row 2 = <unk>.
"""
L = 12          # max words
W = 100         # fixed field width for the MID/REPT split; words are truncated to < W chars
PAD, UNK = 1, 2
# Everything that is not a-z, 0-9 or space gets replaced. MATCH wildcards * ? ~ are included on purpose.
PUNCT = list(".,;:!?\"'()[]{}<>/\\|-_+=*&^%$#@~`")


def clean(prompt: str) -> str:
    s = str(prompt).lower()
    for p in PUNCT:
        s = s.replace(p, " ")
    return s


def normalize(prompt: str) -> list[str]:
    # Excel/Sheets TRIM: strip leading/trailing spaces and collapse inner runs of spaces to one.
    words = [w[: W - 1] for w in clean(prompt).split(" ") if w != ""]
    return words[:L]


def load_vocab(path: str) -> dict[str, int]:
    words = [w.rstrip("\n") for w in open(path, encoding="utf-8")]
    assert words[0] == "<pad>" and words[1] == "<unk>", "vocab rows 1-2 must be <pad>, <unk>"
    idx = {}
    for i, w in enumerate(words, start=1):
        idx.setdefault(w, i)  # MATCH returns the first match
    return idx


def encode(prompt: str, vocab: dict[str, int]) -> list[int]:
    ws = normalize(prompt)
    ids = [vocab.get(w, UNK) for w in ws]
    return ids + [PAD] * (L - len(ids))
