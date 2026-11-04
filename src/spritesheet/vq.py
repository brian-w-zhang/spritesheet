"""v3-VQ (torch prototype; D-52 pending): a grid of discrete codes gives the one-shot decoder enough bits to draw detail.

  encoder (training only): sprite → G×G code logits (Kc codes), straight-through Gumbel-softmax
  decoder (sheet):         G0[y,x] = relu(codebook[code_yx] + c·Wc + b) → v1-style upsample+conv(+FiLM(c)) blocks
  prior (sheet):           codes in raster order; step t: h = relu([c, onehot(t), emb(code_<t)] · W1 + b1),
                           logits = h · W2 + b2; pick by seeded inverse-CDF on softmax(logits / tau) (no LN needed)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SpriteNet


class Encoder(nn.Module):
    def __init__(self, K1, Kc, G, width=128):
        super().__init__()
        assert G == 3, "prototype: 3x3 code grid (decoder starts at C0 channels)"
        n_down = 3
        layers, cin = [nn.Conv2d(K1, width // 2, 3, padding=1), nn.ReLU()], width // 2
        for i in range(n_down):
            layers += [nn.Conv2d(cin, width, 4, stride=2, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
            cin = width
        layers += [nn.Conv2d(width, Kc, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, onehot):
        return self.net(onehot)  # [B, Kc, G, G]


class VQNet(nn.Module):
    """Decoder + prior reuse SpriteNet's text encoder and conv blocks (so the sheet path is v1's)."""

    def __init__(self, cfg, wordvec):
        super().__init__()
        self.cfg = dict(cfg)
        d, C, K, G, Kc, e, h = cfg["d"], [cfg["C0"], cfg["C1"], cfg["C2"], cfg["C3"]], cfg["K"], cfg["G"], cfg["Kc"], cfg["e"], cfg["hp"]
        base = dict(cfg, S=1)
        self.sn = SpriteNet(base, wordvec)          # text encoder + convs + films + out (dense/style unused)
        self.cfg["attn_scale"] = self.sn.cfg["attn_scale"]
        self.enc = Encoder(K + 1, Kc, G)
        self.codebook = nn.Parameter(torch.randn(Kc, C[0]) * 0.5)
        self.dense_c = nn.Linear(d, G * G * C[0])  # text → per-position bias at the code grid
        n_up = int(round(math.log2(cfg["N"] / G)))
        assert G * 2 ** n_up == cfg["N"] and n_up in (2, 3)
        self.n_up = n_up
        # prior
        T = G * G
        self.p_emb = nn.Parameter(torch.randn(Kc, e) * 0.3)
        self.p1 = nn.Linear(d + T + (T - 1) * e, h)
        self.p2 = nn.Linear(h, Kc)

    # ---------------- decoder
    def decode_soft(self, c, code_w):
        """code_w: [B, T, Kc] one-hot (or straight-through soft) → logits [B, K+1, N, N]."""
        B, G, C0 = c.shape[0], self.cfg["G"], self.cfg["C0"]
        emb = code_w @ self.codebook                                  # [B, T, C0]
        g = F.relu(emb + self.dense_c(c).view(B, G * G, C0))
        g = g.view(B, G, G, C0).permute(0, 3, 1, 2)
        convs, films = list(self.sn.convs)[-self.n_up:], list(self.sn.films)[-self.n_up:]
        for conv, film in zip(convs, films):
            g = F.interpolate(g, scale_factor=2, mode="nearest")
            g = F.relu(conv(g) + film(c)[..., None, None])
        return self.sn.out(g)

    def encode_codes(self, tgt, tau, hard=True):
        oh = F.one_hot(tgt.long(), self.cfg["K"] + 1).permute(0, 3, 1, 2).float()
        lg = self.enc(oh).flatten(2).transpose(1, 2)                  # [B, T, Kc]
        w = F.gumbel_softmax(lg, tau=tau, hard=hard, dim=-1)
        return w, lg

    # ---------------- prior (teacher-forced: all positions in parallel)
    def prior_inputs(self, c, codes):
        """codes [B, T] long → x [B, T, d + T + (T-1)e] where position t sees codes < t only."""
        B, T = codes.shape
        e = self.cfg["e"]
        E = self.p_emb[codes]                                          # [B, T, e]
        prev = torch.zeros(B, T, (T - 1) * e, device=c.device)
        for t in range(1, T):
            prev[:, t, : t * e] = E[:, :t].reshape(B, t * e)
        pos = torch.eye(T, device=c.device).expand(B, T, T)
        return torch.cat([c[:, None, :].expand(B, T, c.shape[1]), pos, prev], -1)

    def prior_logits(self, c, codes):
        return self.p2(F.relu(self.p1(self.prior_inputs(c, codes))))  # [B, T, Kc]

    @torch.no_grad()
    def sample_codes(self, c, uniforms, tau=1.0):
        """uniforms [B, T] in (0,1): inverse-CDF pick per position (sheet: SUMPRODUCT((cum<u)*1)+1)."""
        B, T = uniforms.shape
        codes = torch.zeros(B, T, dtype=torch.long, device=c.device)
        for t in range(T):
            lg = self.prior_logits(c, codes)[:, t] / tau
            cum = torch.softmax(lg, -1).cumsum(-1)
            codes[:, t] = (cum < uniforms[:, t:t + 1]).sum(-1).clamp(max=self.cfg["Kc"] - 1)
        return codes

    def generate(self, ids, uniforms, tau=1.0):
        c = self.sn.encode(ids)
        codes = self.sample_codes(c, uniforms, tau)
        w = F.one_hot(codes, self.cfg["Kc"]).float()
        return self.decode_soft(c, w), codes
