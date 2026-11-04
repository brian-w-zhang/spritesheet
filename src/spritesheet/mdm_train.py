"""Train the v2 masked-diffusion model, then calibrate per-step confidence thresholds.

uv run python -m spritesheet.mdm_train --steps 40000 --out artifacts/model/v2_mdm
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

from . import tokenizer as T
from .mdm import MDM_CFG, MDMNet
from .train import BUILD, encode_batch, load_text


def mask_batch(tgt, M, gen):
    """MaskGIT cosine schedule: ratio r = cos(pi/2 * u); each pixel masked with prob r (at least one)."""
    B = tgt.shape[0]
    u = torch.rand(B, 1, 1, generator=gen, device="cpu").to(tgt.device)
    r = torch.cos(math.pi / 2 * u)
    m = torch.rand(tgt.shape, generator=gen, device="cpu").to(tgt.device) < r
    m[:, 0, 0] |= ~m.view(B, -1).any(1)
    return torch.where(m, torch.full_like(tgt, M), tgt), m


@torch.no_grad()
def calibrate_tau(net, sampler, vocab, T, dev, n=512):
    """tau_t = the confidence quantile among still-masked pixels so that the masked fraction after step t
    follows cos(pi/2 * t/T) on a held calibration batch (deterministic given the batch)."""
    caps, tgt, sty, _ = sampler.batch(n)
    ids = encode_batch(caps, vocab).to(dev)
    sty = torch.as_tensor(np.asarray(sty), device=dev)
    _, cond = net.cond(ids, sty)
    N, K = net.cfg["N"], net.cfg["K"]
    M = K + 1
    x = torch.full((n, N, N), M, dtype=torch.long, device=dev)
    taus = []
    for t in range(1, T):
        lg = net.denoise(x, cond)
        conf, am = torch.softmax(lg, -1).max(-1)
        keep_frac = math.cos(math.pi / 2 * t / T)          # fraction of ALL pixels still masked after step t
        masked = x == M
        cm = conf[masked]
        n_masked, n_total = masked.sum().item(), masked.numel()
        n_commit = max(0, int(n_masked - keep_frac * n_total))
        tau = float(torch.sort(cm, descending=True).values[min(n_commit, len(cm)) - 1]) if n_commit > 0 else 1.01
        taus.append(round(tau, 6))
        x = torch.where(masked & (conf >= tau), am, x)
    return taus + [0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--mix", default='{"emoji_var":0.6,"emoji_plain":0.4}')
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--bg-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="warm-start from a model.pt")
    a = ap.parse_args()
    from .data.synth import Sampler
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    vocab, wv = load_text()
    styles = json.load(open(os.path.join(BUILD, "styles.json")))
    pal = json.load(open(os.path.join(BUILD, "palette.json")))
    cfg = dict(MDM_CFG)
    cfg.update(d=int(wv.shape[1]), S=len(styles), K=len(pal["colors"]))
    cfg.update(json.loads(a.cfg))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = MDMNet(cfg, wv).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev))
    sampler = Sampler(split="train", seed=a.seed, tier=a.tier, mix=json.loads(a.mix) if a.mix else None)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    K = cfg["K"]
    M = K + 1
    cw = torch.ones(K + 1, device=dev)
    cw[0] = a.bg_weight
    os.makedirs(a.out, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        caps, tgt, sty, _ = sampler.batch(a.batch)
        ids = encode_batch(caps, vocab).to(dev)
        tgt = torch.as_tensor(np.asarray(tgt), dtype=torch.long, device=dev)
        sty = torch.as_tensor(np.asarray(sty), dtype=torch.long, device=dev)
        c, cond = net.cond(ids, sty)
        xin, m = mask_batch(tgt, M, gen)
        lg = net.denoise(xin, cond)
        loss_px = (F.cross_entropy(lg.permute(0, 3, 1, 2), tgt, weight=cw, reduction="none") * m).sum() / m.sum()
        loss_st = F.cross_entropy(net.style_head(c), sty)
        loss = loss_px + 0.1 * loss_st
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                acc = ((lg.argmax(-1) == tgt) & m).sum().item() / m.sum().item()
            rec = {"step": step, "loss": round(loss.item(), 4), "masked_acc": round(acc, 4), "s": round(time.time() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    net.eval()
    cal = Sampler(split="train", seed=999, tier=a.tier, mix=json.loads(a.mix) if a.mix else None)
    cfg_tau = calibrate_tau(net, cal, vocab, cfg["T"], dev)
    net.cfg["tau"] = cfg_tau
    net = net.cpu()
    w, cfg_out = net.export()
    cfg_out.update(model="mdm", tau=cfg_tau, tier=a.tier, steps=a.steps, mix=a.mix, train_seconds=round(time.time() - t0, 1),
                   params=int(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    np.savez(os.path.join(a.out, "weights.npz"), **w)
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    print("saved", a.out, cfg_out["params"], "params; tau", cfg_tau)


if __name__ == "__main__":
    main()
