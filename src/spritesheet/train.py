"""Train SpriteNet on the synthetic stream.

uv run python -m spritesheet.train --tier 0 --steps 6000 --out artifacts/model/v32_t0
Ablations: --bow (C2), --learn-wordvec (C3), --pixelgpt-n 5000 (C5), --mix '{"emoji_var":1.0}' (C5)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import tokenizer as T
from .model import DEFAULT_CFG, SpriteNet

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "data", "build")


def load_text():
    vocab = T.load_vocab(os.path.join(BUILD, "vocab.txt"))
    wv = np.load(os.path.join(BUILD, "wordvec.npy"))
    return vocab, wv


def encode_batch(captions, vocab):
    return torch.tensor([T.encode(c, vocab) for c in captions], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="{}", help="json overrides for model cfg")
    ap.add_argument("--bow", action="store_true")
    ap.add_argument("--learn-wordvec", action="store_true")
    ap.add_argument("--mix", default=None, help="json mixture weights for Sampler")
    ap.add_argument("--pixelgpt-n", type=int, default=None)
    ap.add_argument("--bg-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from .data.synth import Sampler
    import inspect
    torch.manual_seed(a.seed)
    vocab, wv = load_text()
    styles = json.load(open(os.path.join(BUILD, "styles.json")))
    palette = json.load(open(os.path.join(BUILD, "palette.json")))
    cfg = dict(DEFAULT_CFG)
    cfg.update(d=int(wv.shape[1]), S=len(styles), K=len(palette["colors"]))
    cfg.update(json.loads(a.cfg))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = SpriteNet(cfg, wv, bow=a.bow, learn_wordvec=a.learn_wordvec).to(dev)
    kw = {"split": "train", "seed": a.seed, "tier": a.tier}
    sig = inspect.signature(Sampler).parameters
    if a.mix and "mix" in sig:
        kw["mix"] = json.loads(a.mix)
    if a.pixelgpt_n is not None and "pixelgpt_n" in sig:
        kw["pixelgpt_n"] = a.pixelgpt_n
    sampler = Sampler(**kw)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    cw = torch.ones(cfg["K"] + 1, device=dev)
    cw[0] = a.bg_weight
    os.makedirs(a.out, exist_ok=True)
    log = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        caps, tgt, sty, _ = sampler.batch(a.batch)
        ids = encode_batch(caps, vocab).to(dev)
        tgt = torch.as_tensor(np.asarray(tgt), dtype=torch.long, device=dev)
        sty = torch.as_tensor(np.asarray(sty), dtype=torch.long, device=dev)
        logits, slog = net(ids, sty)
        loss_px = F.cross_entropy(logits, tgt, weight=cw)
        loss_st = F.cross_entropy(slog, sty)
        loss = loss_px + 0.1 * loss_st
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                acc = (logits.argmax(1) == tgt).float().mean().item()
                fg = tgt != 0
                fga = ((logits.argmax(1) == tgt) & fg).sum().item() / max(1, fg.sum().item())
            rec = {"step": step, "loss": round(loss.item(), 4), "px_acc": round(acc, 4), "fg_acc": round(fga, 4),
                   "style_loss": round(loss_st.item(), 4), "s": round(time.time() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    net = net.cpu()
    w, cfg_out = net.export()
    np.savez(os.path.join(a.out, "weights.npz"), **w)
    cfg_out.update(tier=a.tier, steps=a.steps, bow=a.bow, learn_wordvec=a.learn_wordvec, mix=a.mix,
                   pixelgpt_n=a.pixelgpt_n, train_seconds=round(time.time() - t0, 1),
                   params=int(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    print("saved", a.out, cfg_out["params"], "params")


if __name__ == "__main__":
    main()
