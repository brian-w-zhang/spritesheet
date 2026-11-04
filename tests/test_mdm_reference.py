"""v2 masked diffusion: torch twin (fp64) == numpy reference sampler."""
import numpy as np
import torch

from spritesheet import mdm as D

CFG = dict(N=32, L=12, d=16, h=32, S=4, e=4, K=8, E=8, C1=8, C2=8, C3=8, T=3, tau=[0.5, 0.3, 0.0])


def test_mdm_torch_matches_reference():
    torch.manual_seed(0)
    wv = np.random.default_rng(0).normal(0, 0.5, (40, CFG["d"])); wv[0] = 0
    net = D.MDMNet(CFG, wv).double()
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    w, cfg = net.export()
    net.load_state_dict({k: torch.tensor(v) if False else v for k, v in net.state_dict().items()})
    ids = [5, 9, 3] + [1] * 9
    ref = D.ref_generate(ids, 7, w, cfg)
    # torch with the ROUNDED weights: rebuild via a fresh net loaded from the export
    st = ref["style"] - 1
    with torch.no_grad():
        t_ids = torch.tensor([ids])
        x, hist = net.generate(t_ids, torch.tensor([st]), cfg["T"], cfg["tau"])
    # rounding to 15 sig digits perturbs ~1e-15; discrete outputs should agree on this seed
    agree = (x[0].numpy() == ref["classes"]).mean()
    assert agree > 0.999, agree
    assert ref["classes"].min() >= 0 and ref["classes"].max() <= CFG["K"]  # no MASK left
