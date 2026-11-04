"""v3 shapes through Pixcell: 24x24 canvas (3x3 dense grid), 5x5 kernels, 16 variants, 48 colours.
Op parity on every intermediate tab (A2-style) in both local engines; random weights."""
import json
import os

from spritesheet import reference as R
from spritesheet.pixcell import emit, fixtures as F, gates as G

from test_pixcell import ENGINES, _save


def test_a2_v3_shapes():
    res = {"gate": "A2-v3", "models": []}
    for wseed, scale, ks, fz in ((0, 0.3, (5, 5, 3), False), (1, 0.6, (3, 5, 5), True)):
        cfg = F.make_cfg()
        cfg.update(N=24, S=16, K=48, k1=ks[0], k2=ks[1], k3=ks[2], film_z=fz)
        w = R.random_weights(cfg, seed=wseed, scale=scale)
        vocab, pal = F.vocab_words(cfg["V"], wseed), F.palette(cfg["K"], wseed)
        wb = emit.compile_model(w, cfg, vocab, pal, style_names=[f"v{i + 1}" for i in range(16)])
        assert wb.meta["N"] == 24
        r = G.parity(wb, w, cfg, vocab, G.A2_CASES[:6], ENGINES)
        r.update({"weights_seed": wseed, "scale": scale, "config": cfg})
        res["models"].append(r)
    res["pass"] = all(m["pass"] for m in res["models"])
    _save("parity_ops_v3.json", res)
    assert res["pass"], json.dumps({m["weights_seed"]: {e: v["max_rel"] for e, v in m["engines"].items()} for m in res["models"]})
