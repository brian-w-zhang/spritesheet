"""Train v3: 24x24 one-shot SpriteNet on PixelGPT 216K with learned VARIANTS instead of styles (D-51).

Anti-averaging: S variant embeddings; each training sprite updates only the variant that already draws it best
(winner-takes-all, multiple choice learning), with a share eps of random assignments so no variant starves.
The variant head (reference.pick_style) learns p(variant | prompt) from the winners; in the sheet the seed
samples a variant exactly like v1's style. S=1 gives the plain one-shot baseline (pure averaging).
Text augmentation from "The Five-Dollar Model" (Merino et al., 2023): multiplicative N(1, sigma) noise on word vectors.

uv run python -m spritesheet.train_v3 --steps 60000 --S 16 --out artifacts/model/v3_var16
"""
import argparse, json, os, shutil, time

import numpy as np
import torch
import torch.nn.functional as F

from . import tokenizer as T
from .model import DEFAULT_CFG, SpriteNet
from .reference import NEG
from .data.v3 import OUT as V3, Sampler3, eval_set3


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# fixed prompts for training snapshots (row 1: top variant per prompt; row 2: every variant of the first prompt)
SNAP_PROMPTS = ["a frog", "a red apple", "a fox", "a sword", "a wizard with a staff", "a treasure chest", "a mushroom",
                "a ghost", "a blue potion", "a castle", "a cat", "a skeleton warrior", "a golden key", "a tree",
                "a dragon", "a slime"]


class Net3(SpriteNet):
    emb_noise = 0.0

    def encode(self, ids):
        x = F.embedding(ids - 1, self.wordvec)
        if self.training and self.emb_noise > 0:
            x = x * (1 + self.emb_noise * torch.randn_like(x))
        m = (ids != 1).float()
        x = x + self.pos
        S = torch.einsum("btd,bud->btu", self.q(x), self.k(x)) * self.cfg["attn_scale"] + (1 - m)[:, None, :] * NEG
        a = x + self.o(torch.softmax(S, dim=-1) @ self.v(x))
        hdn = a + self.f2(F.relu(self.f1(a)))
        return (m[..., None] * hdn).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)


def per_sample(lg, tgt, cw):
    return F.cross_entropy(lg, tgt, weight=cw, reduction="none").mean((1, 2))


@torch.no_grad()
def all_variant_losses(net, c, tgt, cw, S, chunk=4096):
    """[B, S] per-sample loss for every variant, decoded as one (B*S) batch (chunked)."""
    B = len(c)
    cc = c.repeat_interleave(S, 0)
    kk = torch.arange(S, device=c.device).repeat(B)
    tt = tgt.repeat_interleave(S, 0)
    out = torch.cat([per_sample(net.decode(cc[i:i + chunk], kk[i:i + chunk]), tt[i:i + chunk], cw)
                     for i in range(0, B * S, chunk)])
    return out.view(B, S)


def eps_at(step, total, warm=0.03, ramp=0.15, floor=0.05):
    f = step / total
    if f < warm:
        return 1.0
    if f < ramp:
        return 1.0 - (1.0 - floor) * (f - warm) / (ramp - warm)
    return floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--S", type=int, default=16, help="number of variants (1 = plain one-shot baseline)")
    ap.add_argument("--cfg", default='{"C0":256,"C1":128,"C2":64,"C3":32}')
    ap.add_argument("--emb-noise", type=float, default=0.15)
    ap.add_argument("--bg-weight", type=float, default=0.5)
    ap.add_argument("--prior-weight", type=float, default=0.2)
    ap.add_argument("--eps-floor", type=float, default=0.05)
    ap.add_argument("--refresh", type=float, default=0.15, help="share of cached variant assignments re-searched per step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="", help="one-line strategy note, saved with the run (for the slides)")
    ap.add_argument("--snap-every", type=int, default=5000)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    vocab = T.load_vocab(os.path.join(V3, "vocab.txt"))
    wv = np.load(os.path.join(V3, "wordvec.npy"))
    pal = json.load(open(os.path.join(V3, "palette.json")))
    cfg = dict(DEFAULT_CFG)
    cfg.update(N=24, d=int(wv.shape[1]), S=a.S, K=len(pal["colors"]))
    cfg.update(json.loads(a.cfg))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net3(cfg, wv).to(dev)
    net.emb_noise = a.emb_noise
    sampler = Sampler3("train", seed=a.seed)
    vc, vt = eval_set3("validation", 512)
    v_ids = torch.tensor([T.encode(c, vocab) for c in vc], device=dev)
    v_tgt = torch.as_tensor(vt, dtype=torch.long, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    cw = torch.ones(cfg["K"] + 1, device=dev)
    cw[0] = a.bg_weight
    S = a.S
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.out.rstrip("/"))
    run_dir = os.path.join(ROOT, "artifacts", "runs", name)
    os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
    json.dump({"name": name, "family": "v3", "args": vars(a), "cfg": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "strategy": a.note, "data": "PixelGPT-24x24-216K train split (195,154 sprites), Sampler3 augmentation"},
              open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    live = open(os.path.join(run_dir, "train_log.jsonl"), "w")
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal["colors"]], np.uint8)
    rgb = np.concatenate([np.array([[216, 212, 221]], np.uint8), rgb])
    snap_ids = torch.tensor([T.encode(p, vocab) for p in SNAP_PROMPTS], device=dev)

    @torch.no_grad()
    def snapshot(step):
        from PIL import Image
        net.eval()
        c = net.encode(snap_ids)
        k = net.style_head(c).argmax(1)
        g = net.decode(c, k).argmax(1).cpu().numpy()
        cf = net.encode(snap_ids[:1]).repeat(S, 1)
        gv = net.decode(cf, torch.arange(S, device=dev)).argmax(1).cpu().numpy()
        net.train()
        np.savez_compressed(os.path.join(run_dir, "snapshots", f"step_{step:06d}.npz"), prompts=np.array(SNAP_PROMPTS),
                            grids=g.astype(np.uint8), variants_first_prompt=gv.astype(np.uint8))
        tiles = [rgb[x] for x in g]
        row1 = np.concatenate([np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=255) for t in tiles], 1)
        vt = [rgb[x] for x in gv] + [np.full((24, 24, 3), 255, np.uint8)] * (len(tiles) - len(gv)) if len(gv) < len(tiles) else [rgb[x] for x in gv[:len(tiles)]]
        row2 = np.concatenate([np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=255) for t in vt], 1)
        Image.fromarray(np.concatenate([row1, row2], 0)).resize((row1.shape[1] * 4, row1.shape[0] * 8), Image.NEAREST).save(
            os.path.join(run_dir, "snapshots", f"step_{step:06d}.png"))

    @torch.no_grad()
    def validate():
        net.eval()
        c = net.encode(v_ids)
        L = all_variant_losses(net, c, v_tgt, cw, S)                       # [n, S]
        p = torch.softmax(net.style_head(c), -1)
        rec = {"val_oracle": round(L.min(1).values.mean().item(), 4),     # best variant (how well it CAN draw it)
               "val_expected": round((p * L).sum(1).mean().item(), 4),    # what a random seed gets on average
               "val_top": round(L.gather(1, p.argmax(1, keepdim=True)).mean().item(), 4)}
        net.train()
        return rec

    log, t0 = [], time.time()
    use_hist = torch.zeros(S)
    assign = torch.full((len(sampler.imgs),), -1, dtype=torch.long)  # cached winner per training sprite
    for step in range(1, a.steps + 1):
        caps, tgt, bidx = sampler.batch(a.batch, with_idx=True)
        bidx = torch.as_tensor(bidx)
        ids = torch.tensor([T.encode(c, vocab) for c in caps], device=dev)
        tgt = torch.as_tensor(tgt, dtype=torch.long, device=dev)
        c = net.encode(ids)
        if S > 1:
            # amortised winner-takes-all: re-search unassigned sprites and a random share of the rest
            win = assign[bidx].clone()
            todo = (win < 0) | (torch.rand(len(win)) < a.refresh)
            if todo.any():
                ti = todo.nonzero()[:, 0].to(dev)
                win[todo] = all_variant_losses(net, c.detach()[ti], tgt[ti], cw, S).argmin(1).cpu()
                assign[bidx[todo]] = win[todo]
            win = win.to(dev)
            eps = eps_at(step, a.steps, floor=a.eps_floor)
            rnd = torch.randint(0, S, win.shape, device=dev)
            use = torch.where(torch.rand(win.shape, device=dev) < eps, rnd, win)
            loss_pr = F.cross_entropy(net.style_head(c), win)
            use_hist += torch.bincount(win.cpu(), minlength=S).float()
        else:
            use = torch.zeros(len(caps), dtype=torch.long, device=dev)
            loss_pr = torch.zeros((), device=dev)
            eps = 0.0
        lg = net.decode(c, use)
        loss_px = F.cross_entropy(lg, tgt, weight=cw)
        loss = loss_px + a.prior_weight * loss_pr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                fg = tgt != 0
                fg_acc = ((lg.argmax(1) == tgt) & fg).sum().item() / max(1, fg.sum().item())
            u = use_hist / use_hist.sum().clamp(min=1)
            rec = {"step": step, "loss_px": round(loss_px.item(), 4), "loss_prior": round(loss_pr.item(), 4),
                   "fg_acc": round(fg_acc, 4), "eps": round(eps, 3), "s": round(time.time() - t0, 1),
                   "variants_used_eff": round(float(torch.exp(-(u * torch.log(u + 1e-12)).sum())), 2)}
            use_hist.zero_()
            if step % 2500 == 0 or step == 1:
                rec.update(validate())
            log.append(rec)
            print(json.dumps(rec), flush=True)
            live.write(json.dumps(rec) + "\n")
            live.flush()
        if step % a.snap_every == 0 or step == 1:
            snapshot(step)
        if step in {a.steps // 4, a.steps // 2, 3 * a.steps // 4}:
            torch.save(net.state_dict(), os.path.join(run_dir, f"ckpt_{step:06d}.pt"))
    net.eval()
    net = net.cpu()
    w, cfg_out = net.export()
    cfg_out.update(model="v3", N=24, steps=a.steps, S=S, variants=True, emb_noise=a.emb_noise,
                   train_seconds=round(time.time() - t0, 1), data="PixelGPT-24x24-216K train split",
                   params=int(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    np.savez(os.path.join(a.out, "weights.npz"), **w)
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    for f in ("vocab.txt", "palette.json"):
        shutil.copy(os.path.join(V3, f), os.path.join(a.out, f))
    for f in ("config.json", "train_log.json"):
        shutil.copy(os.path.join(a.out, f), os.path.join(run_dir, f))
    print("saved", a.out, cfg_out["params"], "params")


if __name__ == "__main__":
    main()
