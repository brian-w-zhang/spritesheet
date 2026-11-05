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


# ---------------------------------------------------------------- v3-PG (grouped)
def _gnet(seed=0, guidance=False):
    from spritesheet.patchgpt import GroupGPT
    torch.manual_seed(seed)
    r = np.random.default_rng(seed)
    V, d, K, T = 50, 16, 64, 16
    cfg = {"model": "v3pg", "N": 8, "L": 12, "T": T, "Kp": K, "P": 2, "K": 5, "d": d, "dm": 32, "H": 4, "mlp": 48,
           "layers": 2, "G": 4, "order_seed": 3}
    net = GroupGPT(cfg, r.normal(0, 1, (V, d))).double()
    unet = GroupGPT(dict(cfg, dm=24, H=2, mlp=32, layers=1)).double() if guidance else None
    with torch.no_grad():
        for m in (net, unet) if unet is not None else (net,):
            for n_, p in m.named_parameters():
                p.copy_(torch.randn_like(p) * (0.3 if n_.endswith(".g") else 0.4) + (1.0 if n_.endswith(".g") else 0.0))
    D = r.integers(0, 6, (K, 4))
    return net.eval(), (unet.eval() if unet is not None else None), D, cfg


def test_grouped_only_sees_earlier_groups():
    net, _, _, cfg = _gnet(2)
    ids, c = torch.randint(2, 50, (2, 12)), torch.randint(0, 64, (2, 16))
    n = cfg["T"] // cfg["G"]
    for g in range(cfg["G"]):
        later = net.order[g * n:]
        c2 = c.clone()
        c2[:, later] = (c2[:, later] + 7) % 64
        here = net.order[g * n:(g + 1) * n]
        assert torch.allclose(net.logits(ids, c)[:, here], net.logits(ids, c2)[:, here])


def _check_g(guidance):
    from spritesheet.patchgpt import sample_grouped
    net, unet, D, cfg = _gnet(1, guidance > 0)
    w, rc = RP.export_g(net, D, unet=unet)
    rc["guidance"] = guidance
    for ids, seed in (([3, 7, 9] + [1] * 9, 1), ([5] * 12, 42), ([1] * 12, 123456)):
        ref = RP.forward_g(ids, seed, w, rc)
        noise = torch.tensor(RP.gumbel(seed, cfg["T"], cfg["Kp"]))[None]
        codes = sample_grouped(net, unet, torch.tensor([ids]), noise, guidance)[0].numpy()
        assert (codes == ref["codes"]).all(), (codes, ref["codes"])


def test_grouped_twin_matches_spec():
    _check_g(0.0)


def test_grouped_twin_matches_spec_guided():
    _check_g(2.0)
