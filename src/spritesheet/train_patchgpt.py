"""Train v3-P (src/spritesheet/patchgpt.py): a causal transformer over patch tokens from a fixed dictionary.

uv run python -m spritesheet.train_patchgpt --dict artifacts/model/patchdict_k1024_p3 --steps 60000 --out artifacts/model/v3p_a
Writes artifacts/runs/<name>/ (run.json, train_log.jsonl, snapshots of model samples, ckpts).
"""
import argparse, json, os, shutil, time

import numpy as np
import torch
import torch.nn.functional as F

from . import tokenizer as T
from .patchgpt import PatchGPT, gumbel, noise_tables
from .patches import to_patches, from_patches
from .train_v3 import SNAP_PROMPTS, ROOT
from .data.v3 import OUT as V3, Sampler3, eval_set3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--cfg", default='{"dm":128,"H":4,"mlp":256,"layers":4}')
    ap.add_argument("--emb-noise", type=float, default=0.15)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--p-colour", type=float, default=0.25)
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--snap-every", type=int, default=5000)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    vocab = T.load_vocab(os.path.join(V3, "vocab.txt"))
    wv = np.load(os.path.join(V3, "wordvec.npy"))
    pal = json.load(open(os.path.join(V3, "palette.json")))["colors"]
    pd = json.load(open(os.path.join(a.dict, "patchdict.json")))
    D = np.load(os.path.join(a.dict, "patches.npy"))
    P, K1, Kp = pd["P"], pd["K1"], pd["K"]
    Tn = (24 // P) ** 2
    cfg = {"model": "v3p", "N": 24, "L": T.L, "T": Tn, "Kp": Kp, "P": P, "K": len(pal), "d": int(wv.shape[1]),
           **json.loads(a.cfg)}
    net = PatchGPT(cfg, wv).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev))
    net.emb_noise = a.emb_noise
    Dh = F.one_hot(torch.as_tensor(D, dtype=torch.long), K1).flatten(1).float().to(dev)   # [Kp, P*P*K1]

    def codes_of(grids):
        pt = torch.as_tensor(to_patches(np.asarray(grids), P), dtype=torch.long, device=dev)  # [B, T, P*P]
        x = F.one_hot(pt, K1).flatten(2).float()
        return (x @ Dh.T).argmax(-1)                                                        # [B, T]

    sampler = Sampler3("train", seed=a.seed, p_colour=a.p_colour)
    vc, vt = eval_set3("validation", 1024)
    v_ids = torch.tensor([T.encode(c, vocab) for c in vc], device=dev)
    v_codes = codes_of(vt)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.03)
    name = os.path.basename(a.out.rstrip("/"))
    run_dir = os.path.join(ROOT, "artifacts", "runs", name)
    os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"name": name, "family": "v3", "args": vars(a), "cfg": cfg, "dict": pd | {"fit_log": None},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "strategy": a.note,
               "data": "PixelGPT-24x24-216K train split, Sampler3 augmentation; targets = nearest dictionary patch per 3x3 cell"},
              open(os.path.join(run_dir, "run.json"), "w"), indent=1)
    live = open(os.path.join(run_dir, "train_log.jsonl"), "w")
    rgb = np.concatenate([np.array([[216, 212, 221]], np.uint8), np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])
    A, B = noise_tables(Tn, Kp)
    snap_ids = torch.tensor([T.encode(p, vocab) for p in SNAP_PROMPTS], device=dev)
    snap_noise = torch.tensor(np.stack([gumbel(1, A, B)] * len(SNAP_PROMPTS)), dtype=torch.float32, device=dev)
    frog_noise = torch.tensor(np.stack([gumbel(s, A, B) for s in range(1, 17)]), dtype=torch.float32, device=dev)

    @torch.no_grad()
    def validate():
        net.eval()
        nll, acc = [], []
        for i in range(0, len(v_ids), 256):
            lg = net.logits(v_ids[i:i + 256], v_codes[i:i + 256])
            nll.append(F.cross_entropy(lg.flatten(0, 1), v_codes[i:i + 256].flatten(), reduction="sum").item())
            acc.append((lg.argmax(-1) == v_codes[i:i + 256]).float().sum().item())
        net.train()
        n = v_codes.numel()
        return {"val_nll": round(sum(nll) / n, 4), "val_tok_acc": round(sum(acc) / n, 4)}

    @torch.no_grad()
    def snapshot(step):
        from PIL import Image
        net.eval()
        g = from_patches(D[net.sample(snap_ids, snap_noise).cpu().numpy()], P)
        gf = from_patches(D[net.sample(snap_ids[:1].repeat(16, 1), frog_noise).cpu().numpy()], P)
        net.train()
        np.savez_compressed(os.path.join(run_dir, "snapshots", f"step_{step:06d}.npz"), prompts=np.array(SNAP_PROMPTS),
                            grids=g.astype(np.uint8), frog=gf.astype(np.uint8))
        img = np.concatenate([np.concatenate([np.pad(rgb[x], ((1, 1), (1, 1), (0, 0)), constant_values=255) for x in r], 1) for r in (g, gf)], 0)
        Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(
            os.path.join(run_dir, "snapshots", f"step_{step:06d}.png"))

    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        caps, tgt = sampler.batch(a.batch)
        ids = torch.tensor([T.encode(c, vocab) for c in caps], device=dev)
        codes = codes_of(tgt)
        lg = net.logits(ids, codes)
        loss = F.cross_entropy(lg.flatten(0, 1), codes.flatten(), label_smoothing=a.label_smooth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": round(sched.get_last_lr()[0], 6), "s": round(time.time() - t0, 1)}
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
    cfg_out = dict(cfg, steps=a.steps, train_seconds=round(time.time() - t0, 1), dict_dir=a.dict,
                   params=int(sum(p.numel() for p in net.parameters())), final_log=log[-1])
    json.dump(cfg_out, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    np.save(os.path.join(a.out, "patches.npy"), D)
    for f in ("vocab.txt", "palette.json"):
        shutil.copy(os.path.join(V3, f), os.path.join(a.out, f))
    for f in ("config.json", "train_log.json"):
        shutil.copy(os.path.join(a.out, f), os.path.join(run_dir, f))
    print("saved", a.out, cfg_out["params"], "params")


if __name__ == "__main__":
    main()
