"""Train the v3-VQ prototype (src/spritesheet/vq.py): autoencoder (encoder → G×G codes → sheet decoder) and the
text-conditioned prior over codes, jointly (prior on stop-grad encoder codes).

uv run python -m spritesheet.train_vq --steps 60000 --out artifacts/model/v3_vq --note "..."
Writes artifacts/runs/<name>/ (run.json, train_log.jsonl, snapshots: reconstructions + prior samples, ckpts).
"""
import argparse, json, os, shutil, time

import numpy as np
import torch
import torch.nn.functional as F

from . import tokenizer as T
from .model import DEFAULT_CFG
from .vq import VQNet
from .train_v3 import Net3, SNAP_PROMPTS, ROOT
from .data.v3 import OUT as V3, Sampler3, eval_set3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cfg", default='{"C0":256,"C1":128,"C2":64,"C3":32,"G":3,"Kc":64,"e":16,"hp":256}')
    ap.add_argument("--emb-noise", type=float, default=0.15)
    ap.add_argument("--bg-weight", type=float, default=0.5)
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--usage-weight", type=float, default=0.1, help="batch code-usage entropy bonus (avoid dead codes)")
    ap.add_argument("--tau0", type=float, default=1.0)
    ap.add_argument("--tau1", type=float, default=0.3)
    ap.add_argument("--sample-tau", type=float, default=0.8)
    ap.add_argument("--cond-drop", type=float, default=0.0, help="share of prior training rows with the prompt blanked (enables guidance)")
    ap.add_argument("--init", default=None, help="warm-start from a v3vq model.pt (e.g. to continue training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--snap-every", type=int, default=5000)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    vocab = T.load_vocab(os.path.join(V3, "vocab.txt"))
    wv = np.load(os.path.join(V3, "wordvec.npy"))
    pal = json.load(open(os.path.join(V3, "palette.json")))
    cfg = dict(DEFAULT_CFG)
    cfg.update(N=24, d=int(wv.shape[1]), K=len(pal["colors"]))
    cfg.update(json.loads(a.cfg))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = VQNet(cfg, wv).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev))
    net.sn.__class__ = Net3   # word-vector noise in the text encoder (training only)
    net.sn.emb_noise = a.emb_noise
    sampler = Sampler3("train", seed=a.seed)
    vc, vt = eval_set3("validation", 512)
    v_ids = torch.tensor([T.encode(c, vocab) for c in vc], device=dev)
    v_tgt = torch.as_tensor(vt, dtype=torch.long, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    cw = torch.ones(cfg["K"] + 1, device=dev)
    cw[0] = a.bg_weight
    Tn = cfg["G"] ** 2
    name = os.path.basename(a.out.rstrip("/"))
    run_dir = os.path.join(ROOT, "artifacts", "runs", name)
    os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"name": name, "family": "v3", "args": vars(a), "cfg": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "strategy": a.note, "data": "PixelGPT-24x24-216K train split, Sampler3 augmentation"},
              open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    live = open(os.path.join(run_dir, "train_log.jsonl"), "w")
    rgb = np.concatenate([np.array([[216, 212, 221]], np.uint8),
                          np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal["colors"]], np.uint8)])
    snap_ids = torch.tensor([T.encode(p, vocab) for p in SNAP_PROMPTS], device=dev)
    gen = torch.Generator().manual_seed(123)
    snap_u = torch.rand(len(SNAP_PROMPTS), Tn, generator=gen).to(dev)
    frog_u = torch.rand(16, Tn, generator=gen).to(dev)

    def tau_at(step):
        return a.tau0 * (a.tau1 / a.tau0) ** min(1.0, step / (0.6 * a.steps))

    @torch.no_grad()
    def validate():
        net.eval()
        c = net.sn.encode(v_ids)
        w, lg = net.encode_codes(v_tgt, 0.3)
        codes = lg.argmax(-1)
        rec = net.decode_soft(c, F.one_hot(codes, cfg["Kc"]).float())
        fg = v_tgt != 0
        r = {"val_recon_loss": round(F.cross_entropy(rec, v_tgt, weight=cw).item(), 4),
             "val_recon_fg_acc": round((((rec.argmax(1) == v_tgt) & fg).sum() / fg.sum()).item(), 4),
             "val_prior_nll": round(F.cross_entropy(net.prior_logits(c, codes).flatten(0, 1), codes.flatten()).item(), 4),
             "codes_used": int(codes.unique().numel())}
        net.train()
        return r

    @torch.no_grad()
    def snapshot(step):
        from PIL import Image
        net.eval()
        lg, _ = net.generate(snap_ids, snap_u, a.sample_tau)
        g = lg.argmax(1).cpu().numpy()
        lf, _ = net.generate(snap_ids[:1].repeat(16, 1), frog_u, a.sample_tau)
        gf = lf.argmax(1).cpu().numpy()
        # reconstructions of 16 validation sprites (top: target, bottom: through the codes)
        c = net.sn.encode(v_ids[:16])
        _, clg = net.encode_codes(v_tgt[:16], 0.3)
        gr = net.decode_soft(c, F.one_hot(clg.argmax(-1), cfg["Kc"]).float()).argmax(1).cpu().numpy()
        net.train()
        np.savez_compressed(os.path.join(run_dir, "snapshots", f"step_{step:06d}.npz"), prompts=np.array(SNAP_PROMPTS),
                            grids=g.astype(np.uint8), frog=gf.astype(np.uint8), recon=gr.astype(np.uint8))
        rows = [g, gf, v_tgt[:16].cpu().numpy(), gr]
        img = np.concatenate([np.concatenate([np.pad(rgb[x], ((1, 1), (1, 1), (0, 0)), constant_values=255) for x in r], 1)
                              for r in rows], 0)
        Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(
            os.path.join(run_dir, "snapshots", f"step_{step:06d}.png"))

    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        caps, tgt = sampler.batch(a.batch)
        ids = torch.tensor([T.encode(c, vocab) for c in caps], device=dev)
        tgt = torch.as_tensor(tgt, dtype=torch.long, device=dev)
        c = net.sn.encode(ids)
        w, lg = net.encode_codes(tgt, tau_at(step))
        rec = net.decode_soft(c, w)
        loss_rec = F.cross_entropy(rec, tgt, weight=cw)
        codes = w.argmax(-1).detach()
        cp = c
        if a.cond_drop > 0:
            keep = (torch.rand(len(c), 1, device=dev) >= a.cond_drop).to(c.dtype)
            cp = c * keep
        loss_pr = F.cross_entropy(net.prior_logits(cp, codes).flatten(0, 1), codes.flatten())
        pm = torch.softmax(lg, -1).mean(0)                        # [T, Kc] batch usage per position
        usage_ent = -(pm * torch.log(pm + 1e-9)).sum(-1).mean()
        loss = loss_rec + a.prior_weight * loss_pr - a.usage_weight * usage_ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            rec_ = {"step": step, "loss_px": round(loss_rec.item(), 4), "loss_prior": round(loss_pr.item(), 4),
                    "usage_entropy": round(usage_ent.item(), 3), "tau": round(tau_at(step), 3), "s": round(time.time() - t0, 1)}
            if step % 2500 == 0 or step == 1:
                rec_.update(validate())
            log.append(rec_)
            print(json.dumps(rec_), flush=True)
            live.write(json.dumps(rec_) + "\n")
            live.flush()
        if step % a.snap_every == 0 or step == 1:
            snapshot(step)
        if step in {a.steps // 4, a.steps // 2, 3 * a.steps // 4}:
            torch.save(net.state_dict(), os.path.join(run_dir, f"ckpt_{step:06d}.pt"))
    net.eval()
    net = net.cpu()
    cfg_out = dict(net.cfg, model="v3vq", steps=a.steps, sample_tau=a.sample_tau, cond_drop=a.cond_drop, train_seconds=round(time.time() - t0, 1),
                   params_sheet=int(sum(p.numel() for n_, p in net.named_parameters() if not n_.startswith("enc.") and "dense" not in n_.split(".")[1:2])),
                   params_total=int(sum(p.numel() for p in net.parameters())))
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    for f in ("vocab.txt", "palette.json"):
        shutil.copy(os.path.join(V3, f), os.path.join(a.out, f))
    for f in ("config.json", "train_log.json"):
        shutil.copy(os.path.join(a.out, f), os.path.join(run_dir, f))
    print("saved", a.out, cfg_out["params_total"], "params (total)")


if __name__ == "__main__":
    main()
