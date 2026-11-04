"""Critic (adversarial) fine-tune of a v3-VQ model's sheet decoder, to punish averaged / mushy outputs.

The encoder and prior are frozen. The decoder (codebook, dense_c, conv blocks, output layer) is trained with:
  * reconstruction CE on encoder codes (keeps it faithful), plus
  * hinge GAN loss from a small conditional patch critic. The critic sees real sprites (one-hot) vs decoder outputs,
    fed as straight-through hard samples so it judges committed pixels, not soft probabilities. Fakes come from the
    REAL generation path: prompt → prior-sampled codes → decoder, plus encoder-code reconstructions.
The sheet formulas are unchanged (same architecture, new weights).

uv run python -m spritesheet.train_vq_gan --src artifacts/model/v3_vq6 --steps 15000 --out artifacts/model/v3_vq6_gan
"""
import argparse, json, os, shutil, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer as T
from .vq import VQNet
from .train_v3 import SNAP_PROMPTS, ROOT
from .data.v3 import OUT as V3, Sampler3


class Critic(nn.Module):
    def __init__(self, K1, d, w=96):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(K1, w, 3, padding=1), nn.LeakyReLU(0.2),
                               nn.Conv2d(w, w, 4, stride=2, padding=1), nn.LeakyReLU(0.2),     # 12
                               nn.Conv2d(w, 2 * w, 4, stride=2, padding=1), nn.LeakyReLU(0.2),  # 6
                               nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.LeakyReLU(0.2))
        self.head = nn.Conv2d(2 * w, 1, 1)
        self.proj = nn.Linear(d, 2 * w)  # projection conditioning on the prompt

    def forward(self, x, c):
        h = self.f(x)
        return self.head(h) + (h * self.proj(c)[..., None, None]).sum(1, keepdim=True) / h.shape[1] ** 0.5


def st_hard(logits, tau=1.0):
    """Straight-through one-hot sample of the decoder's per-pixel distribution."""
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adv-weight", type=float, default=0.1)
    ap.add_argument("--recon-weight", type=float, default=1.0)
    ap.add_argument("--bg-weight", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--snap-every", type=int, default=2500)
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    vocab = T.load_vocab(os.path.join(V3, "vocab.txt"))
    wv = np.load(os.path.join(V3, "wordvec.npy"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = VQNet(cfg, wv).to(dev)
    net.load_state_dict(torch.load(os.path.join(a.src, "model.pt"), map_location=dev))
    K1, Kc, Tn = cfg["K"] + 1, cfg["Kc"], cfg["G"] ** 2
    for n_, p in net.named_parameters():
        p.requires_grad = n_.startswith(("codebook", "dense_c", "sn.convs", "sn.films", "sn.out"))
    critic = Critic(K1, cfg["d"]).to(dev)
    opt_g = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=a.lr * 2, betas=(0.5, 0.9))
    cw = torch.ones(K1, device=dev)
    cw[0] = a.bg_weight
    sampler = Sampler3("train", seed=7)
    name = os.path.basename(a.out.rstrip("/"))
    run_dir = os.path.join(ROOT, "artifacts", "runs", name)
    os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"name": name, "family": "v3", "args": vars(a), "cfg": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "strategy": a.note or "critic fine-tune of the VQ decoder (hinge GAN + recon CE), encoder/prior frozen"},
              open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    live = open(os.path.join(run_dir, "train_log.jsonl"), "w")
    pal = json.load(open(os.path.join(V3, "palette.json")))["colors"]
    rgb = np.concatenate([np.array([[216, 212, 221]], np.uint8), np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])
    snap_ids = torch.tensor([T.encode(p, vocab) for p in SNAP_PROMPTS], device=dev)
    g_ = torch.Generator().manual_seed(123)
    snap_u = torch.rand(len(SNAP_PROMPTS), Tn, generator=g_).to(dev)
    frog_u = torch.rand(16, Tn, generator=g_).to(dev)
    tau = cfg.get("sample_tau", 0.8)

    @torch.no_grad()
    def snapshot(step):
        from PIL import Image
        net.eval()
        g = net.generate(snap_ids, snap_u, tau)[0].argmax(1).cpu().numpy()
        gf = net.generate(snap_ids[:1].repeat(16, 1), frog_u, tau)[0].argmax(1).cpu().numpy()
        net.train()
        np.savez_compressed(os.path.join(run_dir, "snapshots", f"step_{step:06d}.npz"), prompts=np.array(SNAP_PROMPTS),
                            grids=g.astype(np.uint8), frog=gf.astype(np.uint8))
        img = np.concatenate([np.concatenate([np.pad(rgb[x], ((1, 1), (1, 1), (0, 0)), constant_values=255) for x in r], 1) for r in (g, gf)], 0)
        Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(
            os.path.join(run_dir, "snapshots", f"step_{step:06d}.png"))

    t0 = time.time()
    snapshot(0)
    for step in range(1, a.steps + 1):
        caps, tgt = sampler.batch(a.batch)
        ids = torch.tensor([T.encode(c, vocab) for c in caps], device=dev)
        tgt = torch.as_tensor(tgt, dtype=torch.long, device=dev)
        with torch.no_grad():
            c = net.sn.encode(ids)
            _, elg = net.encode_codes(tgt, 0.3)
            ecodes = F.one_hot(elg.argmax(-1), Kc).float()
            pcodes = F.one_hot(net.sample_codes(c, torch.rand(len(c), Tn, device=dev), tau), Kc).float()
        real = F.one_hot(tgt, K1).permute(0, 3, 1, 2).float()
        rec = net.decode_soft(c, ecodes)
        gen = net.decode_soft(c, pcodes)
        fake = torch.cat([st_hard(rec), st_hard(gen)])
        cc = torch.cat([c, c])
        # critic step (hinge)
        d_loss = F.relu(1 - critic(real, c)).mean() + F.relu(1 + critic(fake.detach(), cc)).mean()
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()
        # generator step
        g_adv = -critic(fake, cc).mean()
        g_rec = F.cross_entropy(rec, tgt, weight=cw)
        g_loss = a.recon_weight * g_rec + a.adv_weight * g_adv
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt_g.step()
        if step % 250 == 0 or step == 1:
            rec_ = {"step": step, "d_loss": round(d_loss.item(), 4), "g_adv": round(g_adv.item(), 4), "g_rec": round(g_rec.item(), 4),
                    "s": round(time.time() - t0, 1)}
            print(json.dumps(rec_), flush=True)
            live.write(json.dumps(rec_) + "\n")
            live.flush()
        if step % a.snap_every == 0:
            snapshot(step)
    net.eval()
    net = net.cpu()
    cfg_out = dict(cfg, gan_finetune={"src": a.src, "steps": a.steps, "adv_weight": a.adv_weight})
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    for f in ("vocab.txt", "palette.json"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    shutil.copy(os.path.join(a.out, "config.json"), os.path.join(run_dir, "config.json"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
