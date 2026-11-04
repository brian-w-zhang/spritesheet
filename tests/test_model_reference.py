"""torch twin (fp64) == numpy reference (the spec) on random weights."""
import numpy as np
import torch

from spritesheet import reference as R
from spritesheet.model import SpriteNet

CFG = dict(N=32, L=12, d=16, h=32, S=4, e=4, C0=8, C1=8, C2=8, C3=8, K=8)


def test_torch_matches_reference():
    torch.manual_seed(0)
    wv = np.random.default_rng(0).normal(0, 0.5, (40, CFG["d"]))
    wv[0] = 0
    net = SpriteNet(CFG, wv).double()
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    w, cfg = net.export()
    net.load_state_dict({**net.state_dict()}, strict=True)
    # rebuild torch from the rounded export to compare like for like
    net2 = SpriteNet(CFG, w["wordvec"]).double()
    sd = net2.state_dict()
    for n in "qkvo":
        sd[f"{n}.weight"] = torch.tensor(w[f"W{n}"].T); sd[f"{n}.bias"] = torch.tensor(w[f"b{n}"])
    sd["pos"] = torch.tensor(w["pos"]); sd["f1.weight"] = torch.tensor(w["W1"].T); sd["f1.bias"] = torch.tensor(w["b1"])
    sd["f2.weight"] = torch.tensor(w["W2"].T); sd["f2.bias"] = torch.tensor(w["b2"])
    sd["style_head.weight"] = torch.tensor(w["Ws"].T); sd["style_head.bias"] = torch.tensor(w["bs"])
    sd["style_emb"] = torch.tensor(w["style_emb"]); sd["dense.weight"] = torch.tensor(w["Wd"].T); sd["dense.bias"] = torch.tensor(w["bd"])
    for i in range(3):
        sd[f"convs.{i}.weight"] = torch.tensor(w[f"Wc{i+1}"].transpose(3, 2, 0, 1)); sd[f"convs.{i}.bias"] = torch.tensor(w[f"bc{i+1}"])
        sd[f"films.{i}.weight"] = torch.tensor(w[f"F{i+1}"].T)
    sd["out.weight"] = torch.tensor(w["Wout"].T[:, :, None, None]); sd["out.bias"] = torch.tensor(w["bout"])
    net2.load_state_dict(sd)
    ids = [5, 9, 3, 2, 17] + [1] * 7
    for seed in (1, 7, 123456):
        ref = R.forward(ids, seed, w, cfg)
        with torch.no_grad():
            t = torch.tensor([ids])
            c = net2.encode(t)
            assert np.allclose(c[0].numpy(), ref["c"], rtol=1e-12, atol=1e-12)
            logits = net2.decode(c, torch.tensor([ref["style"] - 1]))[0].permute(1, 2, 0).numpy()
            st = net2.style_head(c)[0].numpy()
        assert np.allclose(st, ref["c"] @ w["Ws"] + w["bs"], rtol=1e-12, atol=1e-12)
        assert np.abs(logits - ref["logits"]).max() < 1e-9
        assert (logits.argmax(-1) == ref["classes"]).all()
