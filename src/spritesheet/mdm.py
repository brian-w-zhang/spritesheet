"""SpriteSheet v2: few-step MASKED DISCRETE DIFFUSION (MaskGIT-style) at 32×32.

Generation: start with every pixel = MASK. For t = 1..T: run the denoiser on the current canvas, take per-pixel
softmax confidence and argmax; commit masked pixels whose confidence ≥ tau_t (all of them at t = T). Committed
pixels never change. The text encoder and the seeded style choice are identical to v1 (reference.py) and run once.

numpy float64 `ref_*` functions are the SPEC for the compiler; `MDMNet` is the PyTorch twin for training.
Denoiser per step (cond = [c, style_emb[style]], d+e dims; every conv gets a FiLM shift Fi·cond + fi):
  h0 = Emb[x]                                   32×32×E   (x ∈ 0..K+1, K+1 = MASK)
  a1 = relu(conv3(h0) + F1)                     32×32×C1
  a2 = relu(conv3(avg2(a1)) + F2)               16×16×C2
  a3 = relu(conv3(avg2(a2)) + F3)                8×8×C3
  a4 = relu(conv3(a3) + F4)                      8×8×C3
  u2 = relu(conv3(up2(a4 @ P2)) + a2 + F5)      16×16×C2   (P2: 1×1 C3→C2 before upsampling)
  u1 = relu(conv3(up2(u2 @ P1)) + a1 + F6)      32×32×C1   (P1: 1×1 C2→C1)
  logits = u1 @ Wout + bout                     32×32×(K+1)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import reference as R

MDM_CFG = dict(N=32, L=12, d=64, h=128, S=7, e=8, K=32, E=24, C1=24, C2=48, C3=96, T=6)


# ------------------------------------------------------------------ numpy reference (spec)
HASH_B = 104729  # odd multiplier for the per-cell index


def seeded_uniform(s1, t, N, K1):
    """Exact per-(step, pixel, class) uniforms in (0,1) from the seed state s1 (Latent!B7, same as v1).
    Sheet formula per cell: v = MOD(48271*MOD(s1 + 104729*idx, 2147483647), 2147483647); u = (v + 0.5)/2147483647
    with idx = ((t-1)*N*N + y*N + x)*K1 + k + 1. Every intermediate < 2^53, so it is exact in every engine."""
    idx = ((t - 1) * N * N + np.arange(N * N)[:, None]) * K1 + np.arange(K1)[None, :] + 1
    a = (s1 + HASH_B * idx) % R.M31
    v = (R.LCG_A * a) % R.M31
    return ((v + 0.5) / R.M31).reshape(N, N, K1)


def seed_state(seed):
    s0 = int(math.floor(abs(seed))) % (R.M31 - 1) + 1
    return (R.LCG_A * s0) % R.M31

def avg2(x):
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).sum(axis=(1, 3)) / 4.0


def ref_denoise(x, cond, w, cfg):
    """x: int [N,N] classes incl. MASK = K+1; cond: [d+e]. Returns logits [N,N,K+1]."""
    f = lambda i: cond @ w[f"F{i}"] + w[f"f{i}"]
    h0 = w["Emb"][x]
    a1 = np.maximum(0.0, R.conv3x3(h0, w["Wc1"], 0.0) + f(1))
    a2 = np.maximum(0.0, R.conv3x3(avg2(a1), w["Wc2"], 0.0) + f(2))
    a3 = np.maximum(0.0, R.conv3x3(avg2(a2), w["Wc3"], 0.0) + f(3))
    if "Wg" in w:  # v2b: prompt → 8×8×C3 spatial prior (where things go), same for every step
        n8 = cfg["N"] // 4
        a3 = a3 + (cond @ w["Wg"] + w["bg"]).reshape(n8, n8, cfg["C3"])
    a4 = np.maximum(0.0, R.conv3x3(a3, w["Wc4"], 0.0) + f(4))
    u2 = np.maximum(0.0, R.conv3x3(R.upsample2(a4 @ w["P2"]), w["Wc5"], 0.0) + a2 + f(5))
    u1 = np.maximum(0.0, R.conv3x3(R.upsample2(u2 @ w["P1"]), w["Wc6"], 0.0) + a1 + f(6))
    return u1 @ w["Wout"] + w["bout"]


def ref_generate(ids, seed, w, cfg, keep_steps=True, temp=None):
    c, enc = R.encode_text(ids, w, cfg)
    style, p_style, u = R.pick_style(c, seed, w, cfg)
    cond = np.concatenate([c, w["style_emb"][style - 1]])
    N, K, T = cfg["N"], cfg["K"], cfg["T"]
    M = K + 1
    x = np.full((N, N), M, dtype=np.int64)
    steps, margins = [], []
    for t in range(1, T + 1):
        lg = ref_denoise(x, cond, w, cfg)
        mx = lg.max(-1)
        conf = 1.0 / np.exp(lg - mx[..., None]).sum(-1)          # max softmax prob
        tp = cfg.get("temp", 0.0) if temp is None else temp
        if tp > 0:  # Gumbel-max sampling from softmax(logits/tp), reproducible from the seed
            u_ = seeded_uniform(seed_state(seed), t, N, K + 1)
            score = lg + tp * (-np.log(-np.log(u_)))
            am = score.argmax(-1)
            conf = np.take_along_axis(np.exp(lg - mx[..., None]), am[..., None], -1)[..., 0] * conf  # p(chosen)
        else:
            am = lg.argmax(-1)
        top2 = np.sort(lg, -1)[..., -2:]
        margins.append(float((top2[..., 1] - top2[..., 0])[x == M].min()) if (x == M).any() else None)
        tau = cfg["tau"][t - 1] if t < T else -1.0
        commit = (x == M) & (conf >= tau)
        x = np.where(commit, am, x)
        if keep_steps:
            steps.append({"canvas": x.copy(), "conf": conf, "logits": lg})
    return {"classes": x, "style": style, "u": u, "c": c, "steps": steps, "min_margin": margins}


# ------------------------------------------------------------------ torch twin
class MDMNet(nn.Module):
    def __init__(self, cfg, wordvec):
        super().__init__()
        self.cfg = dict(cfg)
        d, h, S, e, L, K, E = cfg["d"], cfg["h"], cfg["S"], cfg["e"], cfg["L"], cfg["K"], cfg["E"]
        C1, C2, C3 = cfg["C1"], cfg["C2"], cfg["C3"]
        self.cfg["attn_scale"] = 1.0 / math.sqrt(d)
        self.register_buffer("wordvec", torch.tensor(wordvec, dtype=torch.float32))
        self.pos = nn.Parameter(torch.zeros(L, d))
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.f1, self.f2 = nn.Linear(d, h), nn.Linear(h, d)
        self.style_head = nn.Linear(d, S)
        self.style_emb = nn.Parameter(torch.randn(S, e) * 0.5)
        self.emb = nn.Embedding(K + 2, E)
        conv = lambda a, b: nn.Conv2d(a, b, 3, padding=1, bias=False)
        self.c1, self.c2, self.c3, self.c4 = conv(E, C1), conv(C1, C2), conv(C2, C3), conv(C3, C3)
        self.c5, self.c6 = conv(C2, C2), conv(C1, C1)
        self.p2, self.p1 = nn.Linear(C3, C2, bias=False), nn.Linear(C2, C1, bias=False)
        self.film = nn.ModuleList(nn.Linear(d + e, c) for c in (C1, C2, C3, C3, C2, C1))
        self.out = nn.Linear(C1, K + 1)
        self.use_global = bool(cfg.get("global_map", False))
        if self.use_global:
            n8 = cfg["N"] // 4
            self.gmap = nn.Linear(d + e, n8 * n8 * C3)

    def encode(self, ids):
        x = F.embedding(ids - 1, self.wordvec) + self.pos
        m = (ids != 1).float()
        S = torch.einsum("btd,bud->btu", self.q(x), self.k(x)) * self.cfg["attn_scale"] + (1 - m)[:, None, :] * R.NEG
        a = x + self.o(torch.softmax(S, -1) @ self.v(x))
        hdn = a + self.f2(F.relu(self.f1(a)))
        return (m[..., None] * hdn).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)

    def denoise(self, xcls, cond):  # xcls [B,N,N] long; cond [B,d+e]
        fl = [f(cond)[..., None, None] for f in self.film]
        h0 = self.emb(xcls).permute(0, 3, 1, 2)
        a1 = F.relu(self.c1(h0) + fl[0])
        a2 = F.relu(self.c2(F.avg_pool2d(a1, 2)) + fl[1])
        a3 = F.relu(self.c3(F.avg_pool2d(a2, 2)) + fl[2])
        if self.use_global:
            n8 = self.cfg["N"] // 4
            a3 = a3 + self.gmap(cond).view(-1, n8, n8, self.cfg["C3"]).permute(0, 3, 1, 2)
        a4 = F.relu(self.c4(a3) + fl[3])
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        u2 = F.relu(self.c5(up(self.p2(a4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) + a2 + fl[4])
        u1 = F.relu(self.c6(up(self.p1(u2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) + a1 + fl[5])
        return self.out(u1.permute(0, 2, 3, 1))  # [B,N,N,K+1]

    def cond(self, ids, style_idx):
        c = self.encode(ids)
        return c, torch.cat([c, self.style_emb[style_idx]], -1)

    def export(self):
        sd = {k: v.detach().double().cpu().numpy() for k, v in self.state_dict().items()}
        w = {"wordvec": sd["wordvec"].copy(), "pos": sd["pos"]}
        for n in "qkvo":
            w[f"W{n}"], w[f"b{n}"] = sd[f"{n}.weight"].T, sd[f"{n}.bias"]
        w["W1"], w["b1"], w["W2"], w["b2"] = sd["f1.weight"].T, sd["f1.bias"], sd["f2.weight"].T, sd["f2.bias"]
        w["Ws"], w["bs"], w["style_emb"] = sd["style_head.weight"].T, sd["style_head.bias"], sd["style_emb"]
        w["Emb"] = sd["emb.weight"]
        for i, n in enumerate(["c1", "c2", "c3", "c4", "c5", "c6"], start=1):
            w[f"Wc{i}"] = sd[f"{n}.weight"].transpose(2, 3, 1, 0)
            w[f"F{i}"], w[f"f{i}"] = sd[f"film.{i - 1}.weight"].T, sd[f"film.{i - 1}.bias"]
        w["P2"], w["P1"] = sd["p2.weight"].T, sd["p1.weight"].T
        w["Wout"], w["bout"] = sd["out.weight"].T, sd["out.bias"]
        if self.use_global:  # Wg columns in (y, x, c) row-major order, matching reshape(n8, n8, C3)
            n8, C3 = self.cfg["N"] // 4, self.cfg["C3"]
            Wg = sd["gmap.weight"].reshape(n8, n8, C3, -1)      # torch view(-1,n8,n8,C3): out index = (y*n8+x)*C3+c
            w["Wg"], w["bg"] = Wg.reshape(n8 * n8 * C3, -1).T, sd["gmap.bias"]
        w["wordvec"][0] = 0.0
        cfg = dict(self.cfg)
        cfg["V"] = int(w["wordvec"].shape[0])
        return {k: R.round15(v) for k, v in w.items()}, cfg

    @torch.no_grad()
    def generate(self, ids, style_idx, T, tau, temp=0.0, seeds=None):
        """Batched torch sampler with the same commit rule as ref_generate (temp > 0 needs per-row seeds)."""
        B, N, K = ids.shape[0], self.cfg["N"], self.cfg["K"]
        M = K + 1
        _, cond = self.cond(ids, style_idx)
        x = torch.full((B, N, N), M, dtype=torch.long, device=ids.device)
        hist = []
        for t in range(1, T + 1):
            lg = self.denoise(x, cond)
            p = torch.softmax(lg, -1)
            conf, am = p.max(-1)
            if temp > 0:
                g = torch.stack([torch.from_numpy(-np.log(-np.log(seeded_uniform(seed_state(s), t, N, K + 1))))
                                 for s in seeds]).to(lg.dtype).to(lg.device)
                am = (lg + temp * g).argmax(-1)
                conf = torch.gather(p, -1, am[..., None])[..., 0]
            th = tau[t - 1] if t < T else -1.0
            commit = (x == M) & (conf >= th)
            x = torch.where(commit, am, x)
            hist.append(x.clone())
        return x, hist
