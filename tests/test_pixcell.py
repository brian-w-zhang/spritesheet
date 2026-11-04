"""Pixcell gates A1 (tokenizer parity) and A2 (op parity, every intermediate tab) on HyperFormula and LibreOffice,
plus static checks (allowed formula subset) and a proof that the LibreOffice stale-value check bites."""
import json
import os

import pytest

from spritesheet import reference as R
from spritesheet import tokenizer as T
from spritesheet.oracles import libreoffice as LO
from spritesheet.oracles import scan_errors
from spritesheet.pixcell import emit, fixtures as F, gates as G, lint
from spritesheet.pixcell.workbook import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "artifacts", "metrics")
HAVE_LO = os.path.exists(LO.SOFFICE)
ENGINES = G.ENGINES if HAVE_LO else ("hyperformula",)


def _save(name, res):
    os.makedirs(METRICS, exist_ok=True)
    json.dump(res, open(os.path.join(METRICS, name), "w"), indent=1, default=G.np_default)


@pytest.fixture(scope="module")
def small():
    cfg = F.make_cfg()
    w, vocab, pal = F.random_model(cfg)
    return w, cfg, vocab, pal, emit.compile_model(w, cfg, vocab, pal)


def test_lint_allowed_subset(small):
    *_, wb = small
    res = lint.lint(wb)
    assert res["violations"] == [], res["violations"][:5]
    assert "SPARKLINE" not in res["functions"]
    w, cfg, vocab, pal, _ = small
    wr = emit.compile_model(w, cfg, vocab, pal, render=True)
    rr = lint.lint(wr)
    assert rr["violations"] == [] and rr["functions"].get("SPARKLINE", 0) == 0  # display now uses conditional formatting


def test_lint_catches_violations(small):
    wb2 = emit.compile_model(*small[:4])
    wb2.set("X", 1, 1, "=LET(a,1,a)")
    wb2.set("X", 1, 2, "=MATCH(1,A1:A3)")
    wb2.set("X", 1, 3, '=SPARKLINE(1,{"max",1})')
    v = lint.lint(wb2)["violations"]
    assert len(v) >= 4  # LET, MATCH w/o 0, SPARKLINE outside display, array literal


def test_tokenizer_twin_examples():
    vi = T.index_vocab(["<pad>", "<unk>", "a", "red", "apple", "Dog"])
    assert T.encode("A  RED, apple!!", vi)[:4] == [3, 4, 5, 1]
    assert T.encode("dog", vi)[0] == 6                    # MATCH is case-insensitive
    assert T.encode("a*pple ?", vi)[:3] == [3, 2, 1]      # wildcards stripped -> "a pple"
    long = " ".join(["abcdefghij"] * 12)                  # cumulative length > 100 (D-30)
    assert T.normalize(long) == ["abcdefghij"] * 12
    assert T.normalize("x" * 250) == ["x" * 100, "x" * 100, "x" * 50]


def test_a1_tokenizer_parity():
    res = G.gate_a1(n=320, engines=ENGINES)
    _save("parity_tokenizer.json", res)
    for eng, e in res["engines"].items():
        assert e["mismatches"] == 0, (eng, e["examples"])
        assert e["error_cells"] == 0, eng
    assert res["prompts"] >= 300


def test_a2_op_parity():
    res = G.gate_a2(engines=ENGINES)
    _save("parity_ops.json", res)
    print(G.summarize(res))
    for m in res["models"]:
        for eng, e in m["engines"].items():
            assert e["n_fail"] == 0, (eng, e["failures"][:2])
            assert e["max_rel"] <= 1e-12, (eng, e["max_rel"])
            assert e["error_cells"] == 0 and e["stale_cells"] == 0, eng
            assert e["style_mismatch"] == 0 and e["pixels_mismatch"] == 0, eng
        assert m["lint_violations"] == 0
    assert res["pass"]


@pytest.mark.skipif(not HAVE_LO, reason="LibreOffice not installed")
def test_libreoffice_stale_check_bites(small):
    """With recalc 'never', LibreOffice keeps the planted 999s: the check must see them."""
    *_, wb = small
    out = LO.run(wb, [{"prompt": "a red apple", "seed": 1, "read": ["Pixels", "Logits"]}], recalc=1)
    se = scan_errors(wb, out["runs"][0]["values"])
    assert se["stale"] > 0


def test_text_cells_stay_text():
    """Vocab entries that look like numbers/booleans/formulas must stay literal text in both engines."""
    vocab = ["<pad>", "<unk>", "1", "true", "1e5", "a", "=x"]
    wb = emit.tokenizer_rows(["1", "TRUE", "1e5", "a"], vocab)
    for eng in ENGINES:
        runs, _, _ = G._engine_run(eng, wb, [{"read": ["Tok", "Vocab"]}])
        v = runs[0]["values"]
        assert [r[0] for r in v["Vocab"]] == vocab, eng
        c0 = wb.meta["tok"]["id_c0"]
        # "1e5" is cleaned to "1e5" (no punctuation), "true" matches row 4
        assert [int(v["Tok"][i][c0 - 1]) for i in range(4)] == [3, 4, 5, 6], eng
    assert isinstance(wb.sheets["Vocab"][(7, 1)], Text)


def test_seed_sanitise_matches_reference():
    for s in (0, 1, -4.7, 2147483646, 2147483647, 10 ** 12 + 0.5):
        u = R.lcg_uniform(s)
        assert 0 < u < 1
