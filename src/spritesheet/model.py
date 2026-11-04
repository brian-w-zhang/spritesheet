"""PyTorch twin of reference.py (training only). Uses only ops Pixcell can emit."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference import NEG, round15

DEFAULT_CFG = dict(N=32, L=12, d=64, h=128, S=8, e=8, C0=64, C1=48, C2=32, C3=24, K=32)


class SpriteNet(nn.Module):
    def __init__(self, cfg, wordvec: np.ndarray, bow: bool = False, learn_wordvec: bool = False):
        super().__init__()
        self.cfg = dict(cfg)
        d, h, S, e, L = cfg["d"], cfg["h"], cfg["S"], cfg["e"], cfg["L"]
        C = [cfg["C0"], cfg["C1"], cfg["C2"], cfg["C3"]]
        self.cfg["attn_scale"] = 1.0 / math.sqrt(d)
        self.bow = bow  # ablation C2: skip attention + positions (order-free)
        wv = torch.tensor(wordvec, dtype=torch.float32)
        if learn_wordvec:  # ablation C3: no CLIP table, learned embeddings
            self.wordvec = nn.Parameter(torch.randn_like(wv) * 0.1)
        else:
            self.register_buffer("wordvec", wv)
        self.pos = nn.Parameter(torch.zeros(L, d))
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.f1, self.f2 = nn.Linear(d, h), nn.Linear(h, d)
        self.style_head = nn.Linear(d, S)
        self.style_emb = nn.Parameter(torch.randn(S, e) * 0.5)
        self.G0 = cfg.get("N", 32) // 8  # dense output grid (4 for 32x32, 3 for 24x24)
        self.dense = nn.Linear(d + e, self.G0 ** 2 * C[0])
        ks = [cfg.get(f"k{i + 1}", 3) for i in range(3)]  # odd kernel sizes per block (default 3)
        self.convs = nn.ModuleList(nn.Conv2d(C[i], C[i + 1], ks[i], padding=ks[i] // 2) for i in range(3))
        self.films = nn.ModuleList(nn.Linear(d, C[i + 1], bias=False) for i in range(3))
        self.out = nn.Conv2d(C[3], cfg["K"] + 1, 1)

    def encode(self, ids):  # ids: [B, L] 1-based
        wv = self.wordvec
        if not isinstance(wv, nn.Parameter):
            wv = wv
        x = F.embedding(ids - 1, wv)
        m = (ids != 1).float()
        if not self.bow:
            x = x + self.pos
            S = torch.einsum("btd,bud->btu", self.q(x), self.k(x)) * self.cfg["attn_scale"] + (1 - m)[:, None, :] * NEG
            a = x + self.o(torch.softmax(S, dim=-1) @ self.v(x))
        else:
            a = x
        hdn = a + self.f2(F.relu(self.f1(a)))
        c = (m[..., None] * hdn).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        return c

    def decode(self, c, style_idx):  # style_idx 0-based [B]
        C0 = self.cfg["C0"]
        z = torch.cat([c, self.style_emb[style_idx]], dim=-1)
        g = F.relu(self.dense(z)).view(-1, self.G0, self.G0, C0).permute(0, 3, 1, 2)
        for conv, film in zip(self.convs, self.films):
            g = F.interpolate(g, scale_factor=2, mode="nearest")
            g = F.relu(conv(g) + film(c)[..., None, None])
        return self.out(g)  # [B, K+1, N, N]

    def forward(self, ids, style_idx):
        c = self.encode(ids)
        return self.decode(c, style_idx), self.style_head(c)

    def export(self):
        """→ (weights dict float64 rounded to 15 sig digits, cfg) in reference.py layout."""
        sd = {k: v.detach().double().cpu().numpy() for k, v in self.state_dict().items()}
        C0 = self.cfg["C0"]
        w = {"wordvec": sd["wordvec"].copy(), "pos": sd["pos"] if not self.bow else np.zeros_like(sd["pos"])}
        for n in "qkvo":
            w[f"W{n}"] = sd[f"{n}.weight"].T
            w[f"b{n}"] = sd[f"{n}.bias"]
        w["W1"], w["b1"] = sd["f1.weight"].T, sd["f1.bias"]
        w["W2"], w["b2"] = sd["f2.weight"].T, sd["f2.bias"]
        w["Ws"], w["bs"] = sd["style_head.weight"].T, sd["style_head.bias"]
        w["style_emb"] = sd["style_emb"]
        # torch dense output index = c*16 + y*4 + x (view(-1,4,4,C0) then permute → wait: view is [4,4,C0])
        # We used view(-1, 4, 4, C0): output index = (y*4 + x)*C0 + c, i.e. (y, x, c) row-major. Matches reference.
        w["Wd"], w["bd"] = sd["dense.weight"].T, sd["dense.bias"]
        for i in range(3):
            W = sd[f"convs.{i}.weight"]  # [Cout, Cin, 3, 3]
            w[f"Wc{i + 1}"] = W.transpose(2, 3, 1, 0)  # → [3,3,Cin,Cout]
            w[f"bc{i + 1}"] = sd[f"convs.{i}.bias"]
            w[f"F{i + 1}"] = sd[f"films.{i}.weight"].T
        w["Wout"] = sd["out.weight"][:, :, 0, 0].T
        w["bout"] = sd["out.bias"]
        w["wordvec"][0] = 0.0
        cfg = dict(self.cfg)
        cfg["V"] = int(w["wordvec"].shape[0])
        if self.bow:
            cfg["bow"] = True
        return {k: round15(v) for k, v in w.items()}, cfg
