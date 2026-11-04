"""v3-P ("PatchGPT"): text → 64 patch tokens (8x8 grid of 3x3 patches from a fixed dictionary) → 24x24 sprite.

Designed to compile to spreadsheet formulas as ONE causal grid (no per-step copies):
  rows 0..L-1   : prompt words   x = (wordvec[id] @ Win + b_in) + pos_text[i]
  rows L..L+T-2 : image tokens   x = E[code_{t}] + pos_img[t]        (code_t picked from row L-1+t)
  row L-1+t predicts code_t (t = 0..T-1).
Each block: pre-RMSNorm → H-head causal attention (no max-subtraction softmax, pad keys masked, a row always
sees itself) → residual → pre-RMSNorm → ReLU MLP → residual. Final RMSNorm → logits over K patches.
Sampling is Gumbel-max with sheet-exact integer noise (see `gumbel`), so a seed gives one fixed drawing.
Decoding = patch lookup: pixel = D[code][offset].
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

M31 = 2147483647
P_HASH = 999983


def seed_key(seed):
    """Non-linear seed scramble with float64-exact integer ops (each product < 2^53)."""
    h = int(seed) % 1000000
    for c in (12345, 67891, 23456):
        h = (h * h + c) % P_HASH
    return h + 1


def noise_tables(T, K, seed=20261104):
    r = np.random.default_rng(seed)
    return r.integers(1, 2 ** 31 - 1, size=(T, K)).astype(np.int64), r.integers(0, 2 ** 31 - 1, size=(T, K)).astype(np.int64)


def gumbel(seed, A, B):
    """[T, K] Gumbel noise for a seed: u = (MOD(key*A + B, M31) + 0.5) / M31, g = -LN(-LN(u))."""
    key = seed_key(seed)
    u = (np.mod(key * A + B, M31).astype(np.float64) + 0.5) / M31
    return -np.log(-np.log(u))


class RMS(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class Block(nn.Module):
    def __init__(self, dm, H, mlp):
        super().__init__()
        self.n1, self.n2 = RMS(dm), RMS(dm)
        self.qkv, self.o = nn.Linear(dm, 3 * dm), nn.Linear(dm, dm)
        self.f1, self.f2 = nn.Linear(dm, mlp), nn.Linear(mlp, dm)
        self.H = H

    def forward(self, x, mask):
        B, N, dm = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.H, dm // self.H).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(dm // self.H)
        s = s.masked_fill(~mask[:, None], float("-inf"))
        x = x + self.o((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, N, dm))
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class PatchGPT(nn.Module):
    def __init__(self, cfg, wordvec):
        super().__init__()
        self.cfg = dict(cfg)
        dm, L, T, K = cfg["dm"], cfg["L"], cfg["T"], cfg["Kp"]
        self.register_buffer("wordvec", torch.tensor(wordvec, dtype=torch.float32))
        self.win = nn.Linear(wordvec.shape[1], dm)
        self.pos_text = nn.Parameter(torch.randn(L, dm) * 0.02)
        self.E = nn.Parameter(torch.randn(K, dm) * 0.02)
        self.pos_img = nn.Parameter(torch.randn(T - 1, dm) * 0.02)
        self.blocks = nn.ModuleList(Block(dm, cfg["H"], cfg["mlp"]) for _ in range(cfg["layers"]))
        self.nf = RMS(dm)
        self.out = nn.Linear(dm, K)
        self.emb_noise = 0.0

    def text_rows(self, ids):
        x = F.embedding(ids - 1, self.wordvec)
        if self.training and self.emb_noise > 0:
            x = x * (1 + self.emb_noise * torch.randn_like(x))
        return self.win(x) + self.pos_text

    def mask(self, ids, n_img):
        B, L = ids.shape
        N = L + n_img
        keyok = torch.cat([ids != 1, torch.ones(B, n_img, dtype=torch.bool, device=ids.device)], 1)  # PAD id is 1
        causal = torch.ones(N, N, dtype=torch.bool, device=ids.device).tril()
        eye = torch.eye(N, dtype=torch.bool, device=ids.device)
        return (causal[None] & keyok[:, None, :]) | eye[None]

    def logits(self, ids, codes):
        """Teacher-forced: ids [B, L], codes [B, T] → logits [B, T, K] (code t predicted from row L-1+t)."""
        L, T = ids.shape[1], codes.shape[1]
        x = torch.cat([self.text_rows(ids), self.E[codes[:, :-1]] + self.pos_img[: T - 1]], 1)
        m = self.mask(ids, T - 1)
        for b in self.blocks:
            x = b(x, m)
        return self.out(self.nf(x[:, L - 1:]))

    @torch.no_grad()
    def sample(self, ids, noise, tau=1.0):
        """noise [B, T, K] Gumbel → codes [B, T] (greedy if noise is zero). Recomputes the prefix each step (fine at T=64)."""
        B, T = ids.shape[0], noise.shape[1]
        codes = torch.zeros(B, T, dtype=torch.long, device=ids.device)
        for t in range(T):
            lg = self.logits(ids, codes[:, : t + 1])[:, t]       # inputs: codes < t (codes[:, t] is unused)
            codes[:, t] = (lg / tau + noise[:, t]).argmax(-1)
        return codes
