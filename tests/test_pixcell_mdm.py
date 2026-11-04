"""Pixcell v2 (masked discrete diffusion, spec mdm.ref_generate): lint, tokenizer/front reuse, layout, xlsx render,
and gate A2 (every block of every denoising step) on HyperFormula + LibreOffice, random-init MDMNet weights."""
import json
import os

import numpy as np
import openpyxl
import pytest

from spritesheet import mdm as D
from spritesheet import tokenizer as T
from spritesheet.oracles import libreoffice as LO
from spritesheet.pixcell import emit, emit_mdm, fixtures as F, gates as G, lint, trace as TR, xlsx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "artifacts", "metrics")
ENGINES = G.ENGINES if os.path.exists(LO.SOFFICE) else ("hyperformula",)


@pytest.fixture(scope="module")
def small():
    w, cfg = F.random_mdm(F.MDM_SMALL, seed=0, logit_scale=3.0)
    vocab, pal = F.vocab_words(cfg["V"], 0), F.palette(cfg["K"], 0)
    return w, cfg, vocab, pal


def test_lint_and_shape(small):
    w, cfg, vocab, pal = small
    for render in (False, True):
        wb = emit_mdm.compile_mdm(w, cfg, vocab, pal, render=render)
        res = lint.lint(wb)
        assert res["violations"] == [], res["violations"][:5]
        assert "SPARKLINE" not in res["functions"]
    # one shared set of weight tabs, T sets of step tabs
    for i in range(1, 7):
        assert f"W_c{i}" in wb.sheets and f"W_f{i}" in wb.sheets
    assert not any(s.startswith("S1_W") for s in wb.sheets)
    for t in range(1, cfg["T"] + 1):
        for part in emit_mdm.STEP_PARTS:
            assert emit_mdm.step_sheet(t, part) in wb.sheets
    assert f"S{cfg['T'] + 1}_Emb" not in wb.sheets
    # every step references the shared kernels
    assert "W_c1!" in wb.sheets["S3_A1"][(1, 1)] and "W_c1!" in wb.sheets["S1_A1"][(1, 1)]
    # commit rule: >= tau, then MASK; last step commits everything still masked
    M = cfg["K"] + 1
    assert wb.sheets["S1_Canvas"][(1, 1)] == f"=IF(S1_Conf!AH1>={cfg['tau'][0]!r},S1_Conf!BO1,{M})"
    assert wb.sheets["S2_Canvas"][(1, 1)].startswith(f"=IF(S1_Canvas!A1={M},IF(S2_Conf!AH1>=")
    assert wb.sheets["S3_Canvas"][(1, 1)] == f"=IF(S2_Canvas!A1={M},S3_Conf!BO1,S2_Canvas!A1)"
    assert wb.sheets["Pixels"][(1, 1)] == "=S3_Canvas!A1"


def test_front_reuses_v1_tokenizer_and_encoder(small):
    """Prompt/Tokens/encoder/Latent cells are produced by the same emitter as v1 (emit.emit_front)."""
    w, cfg, vocab, pal = small
    wb = emit_mdm.compile_mdm(w, cfg, vocab, pal)
    v1cfg = F.make_cfg(dict(F.SMALL, d=cfg["d"], h=cfg["h"], S=cfg["S"], e=cfg["e"], K=cfg["K"], V=cfg["V"]))
    w1 = F.random_model(v1cfg)[0]
    w1.update({k: w[k] for k in ("wordvec", "pos", "Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo", "W1", "b1",
                                 "W2", "b2", "Ws", "bs", "style_emb")})
    v1cfg["attn_scale"] = cfg["attn_scale"]
    wb1 = emit.compile_model(w1, v1cfg, vocab, pal)
    for sh in ("Tokens", "Vocab", "WordVec", "X", "Q", "K", "VT", "Scores", "Softmax", "Ctx", "Resid", "MLP", "H",
               "Latent", "W_s", "W_emb"):
        assert wb.sheets[sh] == wb1.sheets[sh], sh
    p1 = {k: v for k, v in wb1.sheets["Prompt"].items() if k[0] <= 4 and k != (1, 2)}
    assert all(wb.sheets["Prompt"][k] == v for k, v in p1.items())


def test_trace_matches_spec(small):
    w, cfg, vocab, pal = small
    vi = T.index_vocab(vocab)
    for p, s in (("a red apple", 1), ("", 3), ("dog " * 14, 123456789)):
        tr = TR.trace_mdm(p, s, w, cfg, vi)
        ref = D.ref_generate(T.encode(p, vi), s, w, cfg)
        assert np.array_equal(tr["expected"]["Pixels"], ref["classes"])
        assert tr["forward"]["style"] == ref["style"]
        assert (ref["classes"] <= cfg["K"]).all()


def test_filmstrip_and_xlsx_render(small, tmp_path):
    w, cfg, vocab, pal = small
    wb = emit_mdm.compile_mdm(w, cfg, vocab, pal, render=True)
    sp = emit.render_spec(wb)
    assert len(sp["grids"]) == 1 + cfg["T"]
    assert (cfg["K"] + 1, emit_mdm.MASK_COLOUR) in sp["classes"] and len(sp["classes"]) == cfg["K"] + 2
    for f in wb.meta["filmstrip"]:
        assert wb.sheets["Prompt"][(f["r0"], f["c0"])] == f"=S{f['step']}_Canvas!A1"
        assert wb.sheets["Prompt"][(f["label_row"], f["c0"])].startswith(f"step {f['step']}")
    p = xlsx.to_xlsx(wb, str(tmp_path / "v2.xlsx"))
    ws = openpyxl.load_workbook(p)["Prompt"]
    cf = list(ws.conditional_formatting)
    assert len(cf) == 1 and len(cf[0].rules) == cfg["K"] + 2
    assert ws["C6"].number_format == ";;;" and ws.column_dimensions["C"].width == xlsx.PIXEL_COL_WIDTH
    f0 = wb.meta["filmstrip"][0]
    assert ws.cell(f0["r0"], f0["c0"]).number_format == ";;;"
    # v1 xlsx gets the same colouring (no MASK rule)
    w1, v1cfg = F.random_model(F.make_cfg())[0], F.make_cfg()
    wb1 = emit.compile_model(w1, v1cfg, F.vocab_words(v1cfg["V"]), F.palette(v1cfg["K"]), render=True)
    ws1 = openpyxl.load_workbook(xlsx.to_xlsx(wb1, str(tmp_path / "v1.xlsx")))["Prompt"]
    assert len(list(ws1.conditional_formatting)[0].rules) == v1cfg["K"] + 1


def test_a2_mdm_op_parity():
    res = G.gate_a2_mdm(engines=ENGINES)
    os.makedirs(METRICS, exist_ok=True)
    json.dump(res, open(os.path.join(METRICS, "parity_ops_mdm.json"), "w"), indent=1, default=G.np_default)
    print(G.summarize(res))
    committed = np.zeros(3)
    for m in res["models"]:
        assert m["lint_violations"] == 0
        for eng, e in m["engines"].items():
            assert e["n_fail"] == 0, (eng, e["failures"][:2])
            assert e["max_rel"] <= 1e-12, (eng, e["max_rel"])
            assert e["error_cells"] == 0 and e["stale_cells"] == 0, eng
            assert e["pixels_mismatch"] == 0 and e["style_mismatch"] == 0
        committed += [s["committed_mean"] for s in m["steps"]]
        assert not any(s["near_tie"] for s in m["steps"]), m["steps"]
    assert (committed > 0).all(), committed   # the models commit pixels at every step, so every branch is exercised


def test_a2_mdm_temperature_sampling():
    """cfg temp > 0: seeded Gumbel-max (LN only in S{t}_Score), conf = p(chosen); same parity bar."""
    res = G.gate_a2_mdm(engines=ENGINES, cases=G.A2_CASES[:5], temp=0.7)
    json.dump(res, open(os.path.join(METRICS, "parity_ops_mdm_temp.json"), "w"), indent=1, default=G.np_default)
    print(G.summarize(res))
    for m in res["models"]:
        assert m["lint_violations"] == 0 and m["functions"].get("LN", 0) > 0
        for eng, e in m["engines"].items():
            assert e["n_fail"] == 0 and e["max_rel"] <= 1e-12, (eng, e["failures"][:2], e["max_rel"])
            assert e["error_cells"] == 0 and e["stale_cells"] == 0, eng
        assert not any(s["near_tie"] for s in m["steps"]), m["steps"]


def test_a2_mdm_global_map():
    """v2b: cfg global_map / w["Wg"]: one GMap tab (cond @ Wg + bg, 8x8xC3) added to every step's A3 after the ReLU."""
    res = G.gate_a2_mdm(engines=ENGINES, cases=G.A2_CASES[:5], global_map=True)
    json.dump(res, open(os.path.join(METRICS, "parity_ops_mdm_gmap.json"), "w"), indent=1, default=G.np_default)
    print(G.summarize(res))
    for m in res["models"]:
        assert m["lint_violations"] == 0
        for eng, e in m["engines"].items():
            assert e["n_fail"] == 0 and e["max_rel"] <= 1e-12, (eng, e["failures"][:2], e["max_rel"])
            assert e["error_cells"] == 0 and e["stale_cells"] == 0, eng
            assert e["per_block"]["GMap"]["bad"] == 0
        assert not any(s["near_tie"] for s in m["steps"]), m["steps"]
