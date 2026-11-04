"""Re-calibrate a trained v2 model's per-step thresholds for a new step count T and background penalty beta
(no retraining; T and the commit rule are sampling settings). Writes a new model dir sharing the weights.
uv run python scripts/recalibrate_v2.py --src artifacts/model/v2_mdm_c_pgmajor --T 4 --beta 0.6 --out artifacts/model/v2c_T4_b06
"""
import argparse, json, math, os, shutil, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import tokenizer as T  # noqa
from spritesheet.train import encode_batch  # noqa
from compare_models import load_v2  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--T", type=int, default=4); ap.add_argument("--beta", type=float, default=0.6)
ap.add_argument("--n", type=int, default=512)
a = ap.parse_args()
from spritesheet.data.synth import Sampler
net, cfg = load_v2(a.src)
mix = json.loads(cfg["mix"]) if isinstance(cfg.get("mix"), str) else cfg.get("mix")
vocab = T.load_vocab(os.path.join(ROOT, "data/build/vocab.txt"))
caps, tgt, sty, _ = Sampler(split="train", seed=999, tier=1, mix=mix).batch(a.n)
ids = encode_batch(caps, vocab); sty = torch.as_tensor(np.asarray(sty))
N, K = cfg["N"], cfg["K"]; M = K + 1
with torch.no_grad():
    _, cond = net.cond(ids, sty)
    x = torch.full((a.n, N, N), M, dtype=torch.long); taus = []
    for t in range(1, a.T):
        lg = net.denoise(x, cond); conf, am = torch.softmax(lg, -1).max(-1)
        score = conf * torch.where(am == 0, torch.full_like(conf, a.beta), torch.ones_like(conf))
        masked = x == M
        keep = math.cos(math.pi / 2 * t / a.T)
        n_commit = max(0, int(masked.sum().item() - keep * masked.numel()))
        sm = torch.sort(score[masked], descending=True).values
        tau = float(sm[min(n_commit, len(sm)) - 1]) if n_commit > 0 else 0.9999
        tau = round(min(tau, 0.9999), 6); taus.append(tau)
        x = torch.where(masked & (score >= tau), am, x)
taus.append(0.0)
os.makedirs(a.out, exist_ok=True)
for f in ("weights.npz", "model.pt"):
    shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
cfg2 = dict(cfg); cfg2.update(T=a.T, tau=taus, bg_beta=a.beta, recalibrated_from=os.path.basename(a.src))
json.dump(cfg2, open(os.path.join(a.out, "config.json"), "w"), indent=1)
print("tau", taus, "→", a.out)
