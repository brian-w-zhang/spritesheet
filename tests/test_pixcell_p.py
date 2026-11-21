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


@pytest.mark.parametrize("guidance,sizes", [(0.0, None), (2.0, None), (2.0, "cosine")])
def test_parity_v3pg(guidance, sizes):
    if not os.path.exists(os.path.join(ROOT, "tools", "node_modules", "hyperformula")):
        pytest.skip("HyperFormula not installed (npm ci --prefix tools)")
    from test_patchgpt import _gnet
    net, unet, D, cfg0 = _gnet(3, guidance > 0, sizes)
    w, cfg = RP.export_g(net, D, unet=unet)
    cfg["guidance"] = guidance
    vocab, pal = F.vocab_words(w["wordvec"].shape[0], 3), F.palette(cfg["K"], 3)
    wb = emit_p.compile_pg(w, cfg, vocab, pal)
    r = G.parity(wb, w, cfg, vocab, CASES, G.ENGINES)
    out = os.path.join(ROOT, "artifacts", "metrics", "parity_ops_v3p.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev[f"grouped,guidance={guidance:g}" + (f",sizes={sizes}" if sizes else "")] = {k: v for k, v in r.items()}
    json.dump(prev, open(out, "w"), indent=1, default=G.np_default)
    for eng, e in r["engines"].items():
        assert e["n_fail"] == 0 and e["error_cells"] == 0, (eng, e["failures"][:2])
    assert r["lint_violations"] == 0 and r["pass"]


def _slot_model(guidance, seed=3):
    """small random v4 slot model (4 palette tokens + 16 2x2 shape tokens) with an optional U chain."""
    import torch
    from spritesheet.patchgpt import PatchGPT, UPatchGPT
    from spritesheet.slots import NS
    torch.manual_seed(seed)
    r = np.random.default_rng(seed)
    Ks, nc, Tsh = 40, 6, 16
    T, K = NS + Tsh, Ks + nc
    cfg = {"model": "v3p", "N": 8, "L": 12, "T": T, "Kp": K, "P": 2, "K": nc - 1, "d": 16, "dm": 32, "H": 4, "mlp": 48,
           "layers": 2, "slots": NS, "Ks": Ks}
    net = PatchGPT(cfg, r.normal(0, 1, (50, 16))).double()
    unet = UPatchGPT({"dm": 24, "H": 2, "mlp": 32, "layers": 1, "T": T, "Kp": K}).double() if guidance else None
    with torch.no_grad():
        for m in (net, unet) if unet is not None else (net,):
            for n_, p in m.named_parameters():
                p.copy_(torch.randn_like(p) * 0.4 + (1.0 if n_.endswith(".g") else 0.0))
    D = r.integers(0, NS + 1, (Ks, 4))
    w, rc = RP.export(net.eval(), D, unet=None if unet is None else unet.eval())
    rc["guidance"] = guidance
    return w, rc, F.vocab_words(w["wordvec"].shape[0], seed), F.palette(rc["K"], seed)


@pytest.mark.parametrize("guidance", [0.0, 1.0])
def test_parity_v4_slots(guidance):
    if not os.path.exists(os.path.join(ROOT, "tools", "node_modules", "hyperformula")):
        pytest.skip("HyperFormula not installed (npm ci --prefix tools)")
    w, cfg, vocab, pal = _slot_model(guidance)
    wb = emit_p.compile_p(w, cfg, vocab, pal)
    r = G.parity(wb, w, cfg, vocab, CASES, G.ENGINES)
    out = os.path.join(ROOT, "artifacts", "metrics", "parity_ops_v3p.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev[f"slots,guidance={guidance:g}"] = {k: v for k, v in r.items()}
    json.dump(prev, open(out, "w"), indent=1, default=G.np_default)
    for eng, e in r["engines"].items():
        assert e["n_fail"] == 0 and e["error_cells"] == 0, (eng, e["failures"][:2], e["per_block"])
        assert "Pal" in e["per_block"] and "Pixels" in e["per_block"], e["per_block"].keys()
    assert r["lint_violations"] == 0 and r["pass"]
