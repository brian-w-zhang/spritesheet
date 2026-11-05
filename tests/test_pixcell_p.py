"""Gate A2 for v3-P: every block of the PatchGPT workbook (random small model) matches reference_p in HyperFormula and
LibreOffice: continuous blocks to 1e-12, codes and pixels exactly. With and without guidance.
Writes artifacts/metrics/parity_ops_v3p.json."""
import json
import os

import numpy as np
import pytest

from spritesheet import reference_p as RP
from spritesheet.pixcell import emit_p, fixtures as F, gates as G
from test_patchgpt import _net

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [("a red apple", 1), ("Big blue CAT with a hat!!", 7), ("", 3), ("dog " * 14, 123456789), ("zzzunknown", 2),
         ("   pixel   art   sword  ", 2147483646), ("apple", 0), ("a", -4.7)]


def _model(guidance, seed):
    net, unet, D, cfg = _net(seed, guidance > 0)
    w, rc = RP.export(net, D, unet=unet)
    rc["guidance"] = guidance
    V = w["wordvec"].shape[0]
    vocab = F.vocab_words(V, seed)
    pal = F.palette(rc["K"], seed)
    return w, rc, vocab, pal


@pytest.mark.parametrize("guidance", [0.0, 2.0])
def test_parity_v3p(guidance):
    if not os.path.exists(os.path.join(ROOT, "tools", "node_modules", "hyperformula")):
        pytest.skip("HyperFormula not installed (npm ci --prefix tools)")
    w, cfg, vocab, pal = _model(guidance, 3)
    wb = emit_p.compile_p(w, cfg, vocab, pal)
    r = G.parity(wb, w, cfg, vocab, CASES, G.ENGINES)
    out = os.path.join(ROOT, "artifacts", "metrics", "parity_ops_v3p.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev[f"guidance={guidance:g}"] = {k: v for k, v in r.items() if k != "functions"} | {"functions": r["functions"]}
    json.dump(prev, open(out, "w"), indent=1, default=G.np_default)
    for eng, e in r["engines"].items():
        assert e["n_fail"] == 0 and e["error_cells"] == 0, (eng, e["failures"][:2], e["per_block"])
    assert r["lint_violations"] == 0
    assert r["pass"], {eng: e["max_rel"] for eng, e in r["engines"].items()}


@pytest.mark.parametrize("guidance", [0.0, 2.0])
def test_parity_v3pg(guidance):
    if not os.path.exists(os.path.join(ROOT, "tools", "node_modules", "hyperformula")):
        pytest.skip("HyperFormula not installed (npm ci --prefix tools)")
    from test_patchgpt import _gnet
    net, unet, D, cfg0 = _gnet(3, guidance > 0)
    w, cfg = RP.export_g(net, D, unet=unet)
    cfg["guidance"] = guidance
    vocab, pal = F.vocab_words(w["wordvec"].shape[0], 3), F.palette(cfg["K"], 3)
    wb = emit_p.compile_pg(w, cfg, vocab, pal)
    r = G.parity(wb, w, cfg, vocab, CASES, G.ENGINES)
    out = os.path.join(ROOT, "artifacts", "metrics", "parity_ops_v3p.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev[f"grouped,guidance={guidance:g}"] = {k: v for k, v in r.items()}
    json.dump(prev, open(out, "w"), indent=1, default=G.np_default)
    for eng, e in r["engines"].items():
        assert e["n_fail"] == 0 and e["error_cells"] == 0, (eng, e["failures"][:2])
    assert r["lint_violations"] == 0 and r["pass"]
