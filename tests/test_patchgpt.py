"""v3-P: the torch training twin (patchgpt.py) and the numpy spec (reference_p.py) agree: same codes for the same
seed, logits to ~1e-12, with folded RMSNorm gains and with/without guidance."""
import numpy as np
import torch

from spritesheet import reference_p as RP
from spritesheet.patchgpt import PatchGPT, UPatchGPT, sample_guided


def _net(seed=0, guidance=False):
    torch.manual_seed(seed)
    r = np.random.default_rng(seed)
    V, d, K, T = 50, 16, 64, 16
    cfg = {"model": "v3p", "N": 8, "L": 12, "T": T, "Kp": K, "P": 2, "K": 5, "d": d, "dm": 32, "H": 4, "mlp": 48, "layers": 2}
    net = PatchGPT(cfg, r.normal(0, 1, (V, d))).double()
    with torch.no_grad():
        for n_, p in net.named_parameters():
            p.copy_(torch.randn_like(p) * (0.4 if not n_.endswith(".g") else 0.3) + (1.0 if n_.endswith(".g") else 0.0))
    unet = None
    if guidance:
        unet = UPatchGPT({"dm": 24, "H": 2, "mlp": 32, "layers": 1, "T": T, "Kp": K}).double()
        with torch.no_grad():
            for n_, p in unet.named_parameters():
                p.copy_(torch.randn_like(p) * 0.4 + (1.0 if n_.endswith(".g") else 0.0))
    D = r.integers(0, 6, (K, 4))
    return net.eval(), unet, D, cfg


def _check(guidance):
    net, unet, D, cfg = _net(1, guidance > 0)
    w, rc = RP.export(net, D, unet=unet)
    rc["guidance"] = guidance
    for ids, seed in (([3, 7, 9] + [1] * 9, 1), ([5] * 12, 42), ([2] + [1] * 11, 123456)):
        ref = RP.forward(ids, seed, w, rc)
        noise = torch.tensor(RP.gumbel(seed, cfg["T"], cfg["Kp"]))[None]
        it = torch.tensor([ids])
        codes = sample_guided(net, unet, it, noise, guidance)[0].numpy()
        assert (codes == ref["codes"]).all(), (codes, ref["codes"])
        lg = net.logits(it, torch.tensor(codes)[None])[0].detach().numpy()
        if guidance:
            lg = (1 + guidance) * lg - guidance * unet.logits(torch.tensor(codes)[None])[0].detach().numpy()
        assert np.abs(lg - ref["logits"]).max() / max(1.0, np.abs(ref["logits"]).max()) < 1e-9
        assert ref["pixels"].shape == (8, 8)


def test_twin_matches_spec():
    _check(0.0)


def test_twin_matches_spec_guided():
    _check(2.0)
