"""v3-P ("PatchGPT"): text → 64 patch tokens (8x8 grid of 3x3 patches from a fixed dictionary) → 24x24 sprite.

Designed to compile to spreadsheet formulas as ONE causal grid (no per-step copies):
  rows 0..L-1   : prompt words   x = (wordvec[id] @ Win + b_in) + pos_text[i]
  rows L..L+T-2 : image tokens   x = E[code_{t}] + pos_img[t]        (code_t picked from row L-1+t)
  row L-1+t predicts code_t (t = 0..T-1).
Each block: pre-RMSNorm → H-head causal attention (no max-subtraction softmax, pad keys masked, a row always
sees itself) → residual → pre-RMSNorm → ReLU MLP → residual. Final RMSNorm → logits over K patches.
Sampling is Gumbel-max with sheet-exact integer noise (reference_p.gumbel), so a seed gives one fixed drawing.
Decoding = patch lookup: pixel = D[code][offset].
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference_p import gumbel  # noqa: F401  (sheet-exact seeded Gumbel noise [T, K]; the spec owns it)


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


class UPatchGPT(nn.Module):
    """Unconditional prior over patch codes (for classifier-free guidance in the sheet: a second, smaller chain).
    Row 0 = learned start row; row t >= 1 = E[code_{t-1}] + pos_img[t-1]; row t predicts code_t."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        dm, T, K = cfg["dm"], cfg["T"], cfg["Kp"]
        self.start = nn.Parameter(torch.randn(dm) * 0.02)
        self.E = nn.Parameter(torch.randn(K, dm) * 0.02)
        self.pos_img = nn.Parameter(torch.randn(T - 1, dm) * 0.02)
        self.blocks = nn.ModuleList(Block(dm, cfg["H"], cfg["mlp"]) for _ in range(cfg["layers"]))
        self.nf = RMS(dm)
        self.out = nn.Linear(dm, K)

    def logits(self, codes):
        B, T = codes.shape
        x = torch.cat([self.start.expand(B, 1, -1), self.E[codes[:, :-1]] + self.pos_img[: T - 1]], 1)
        m = torch.ones(T, T, dtype=torch.bool, device=codes.device).tril()[None].expand(B, T, T)
        for b in self.blocks:
            x = b(x, m)
        return self.out(self.nf(x))


def pick(lg, noise, tau=1.0, trunc=None, bg=0):
    """Gumbel-max pick. trunc = c > 0: min-p truncation among the foreground codes only: a code other than the
    empty patch `bg` can win only if its logit >= (max foreground logit) - c. The empty patch is never truncated, so
    the background-vs-object decision keeps its full probability (truncating it empties the raster-order image)."""
    s = lg / tau + noise
    if trunc:
        fgl = lg.clone()
        fgl[..., bg] = -1e30
        keep = lg >= fgl.max(-1, keepdim=True).values - trunc
        keep[..., bg] = True
        s = torch.where(keep, s, torch.full_like(s, -1e30))
    return s.argmax(-1)


@torch.no_grad()
def sample_guided(net, unet, ids, noise, guidance=0.0, tau=1.0, trunc=None):
    """Gumbel-max sampling with classifier-free guidance: l = (1+g) l_c - g l_u (as reference_p.forward)."""
    B, T = ids.shape[0], noise.shape[1]
    codes = torch.zeros(B, T, dtype=torch.long, device=ids.device)
    for t in range(T):
        lg = net.logits(ids, codes[:, : t + 1])[:, t]
        if guidance and unet is not None:
            lg = (1 + guidance) * lg - guidance * unet.logits(codes[:, : t + 1])[:, t]
        codes[:, t] = pick(lg, noise[:, t], tau, trunc)
    return codes


# ---------------------------------------------------------------- v3-PG: grouped (parallel-per-group) decoding
def group_layout(T, G, seed=0):
    """Fixed spread-out order of the T patch positions, split into G equal groups.
    → order (list of positions), rows [(kind, pos, group)] for the image part: Q rows of group g, then C rows of
    group g (the last group has no C rows: nothing reads them)."""
    order = [int(p) for p in np.random.default_rng(seed).permutation(T)]
    n = T // G
    rows = []
    for g in range(G):
        grp = order[g * n:(g + 1) * n]
        rows += [("q", p, g) for p in grp]
        if g < G - 1:
            rows += [("c", p, g) for p in grp]
    return order, rows


class GroupGPT(nn.Module):
    """Text rows, then per group g: query rows (input = qpos[p]; give the logits for patch p) and context rows
    (input = E[code_p] + cpos[p]). Causal in row order; query rows are never keys (not even for themselves), and
    text row 0 is always a key (a "start" row, so no row is ever without keys). Group g is predicted from the prompt
    and the patches of groups < g only; in the sheet the dependency depth is G rounds. Without a word table it is the
    unconditional prior (guidance): its first row is a learned start vector."""

    def __init__(self, cfg, wordvec=None):
        super().__init__()
        self.cfg = dict(cfg)
        dm, T, K, L = cfg["dm"], cfg["T"], cfg["Kp"], cfg.get("L", 12)
        self.L = L if wordvec is not None else 1
        self.cond = wordvec is not None
        self.order, self.rows = group_layout(T, cfg["G"], cfg.get("order_seed", 0))
        if not self.cond:
            self.start = nn.Parameter(torch.randn(dm) * 0.02)
        if self.cond:
            self.register_buffer("wordvec", torch.tensor(wordvec, dtype=torch.float32))
            self.win = nn.Linear(wordvec.shape[1], dm)
            self.pos_text = nn.Parameter(torch.randn(self.L, dm) * 0.02)
        self.E = nn.Parameter(torch.randn(K, dm) * 0.02)
        self.qpos = nn.Parameter(torch.randn(T, dm) * 0.02)
        self.cpos = nn.Parameter(torch.randn(T, dm) * 0.02)
        self.blocks = nn.ModuleList(Block(dm, cfg["H"], cfg["mlp"]) for _ in range(cfg["layers"]))
        self.nf = RMS(dm)
        self.out = nn.Linear(dm, K)
        self.emb_noise = 0.0
        kinds = [k for k, _, _ in self.rows]
        self.register_buffer("qrow", torch.tensor([i for i, k in enumerate(kinds) if k == "q"]))
        self.register_buffer("qpos_of", torch.tensor([p for k, p, _ in self.rows if k == "q"]))
        self.register_buffer("cpos_of", torch.tensor([p for k, p, _ in self.rows if k == "c"]))
        self.register_buffer("is_c", torch.tensor([k == "c" for k in kinds]))
        self.register_buffer("is_q", torch.tensor([k == "q" for k in kinds]))

    def logits(self, ids, codes):
        """ids [B, L] (ignored when L = 0), codes [B, T] by position → logits [B, T, K] by position."""
        B, dev = codes.shape[0], codes.device
        R = len(self.rows)
        x = torch.zeros(B, R, self.cfg["dm"], device=dev, dtype=self.E.dtype)
        x[:, self.is_q] = self.qpos[self.qpos_of].expand(B, -1, -1)
        x[:, self.is_c] = self.E[codes[:, self.cpos_of]] + self.cpos[self.cpos_of]
        keyok = (~self.is_q)[None].expand(B, R)
        if self.cond:
            t = F.embedding(ids - 1, self.wordvec)
            if self.training and self.emb_noise > 0:
                t = t * (1 + self.emb_noise * torch.randn_like(t))
            x = torch.cat([self.win(t) + self.pos_text, x], 1)
            tk = ids != 1
            tk[:, 0] = True                                       # text row 0 is always a key
            keyok = torch.cat([tk, keyok], 1)
        else:
            x = torch.cat([self.start.expand(B, 1, -1), x], 1)
            keyok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), keyok], 1)
        N = x.shape[1]
        causal = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
        selfok = torch.cat([torch.ones(self.L, dtype=torch.bool, device=dev), ~self.is_q])
        m = (causal[None] & keyok[:, None, :]) | (torch.eye(N, dtype=torch.bool, device=dev) & selfok[None, :])[None]
        for b in self.blocks:
            x = b(x, m)
        lq = self.out(self.nf(x[:, self.L + self.qrow]))          # [B, T, K] in query-row order
        out = torch.empty_like(lq)
        out[:, self.qpos_of] = lq
        return out


@torch.no_grad()
def sample_grouped(net, unet, ids, noise, guidance=0.0, tau=1.0, trunc=None):
    """G rounds; round g fixes the patches of group g (noise [B, T, K] indexed by position, as reference_p)."""
    B, T = noise.shape[0], noise.shape[1]
    codes = torch.zeros(B, T, dtype=torch.long, device=noise.device)
    n = T // net.cfg["G"]
    for g in range(net.cfg["G"]):
        pos = torch.tensor(net.order[g * n:(g + 1) * n], device=noise.device)
        lg = net.logits(ids, codes)[:, pos]
        if guidance and unet is not None:
            lg = (1 + guidance) * lg - guidance * unet.logits(None, codes)[:, pos]
        codes[:, pos] = pick(lg, noise[:, pos], tau, trunc)
    return codes
