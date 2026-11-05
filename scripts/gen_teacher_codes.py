"""Sequence-level distillation data for the grouped (parallel) PatchGPT: sample patch codes from the causal teacher.

uv run python scripts/gen_teacher_codes.py --teacher artifacts/model/v3p_a --n 200000 --tau 0.8 --out data/build/v3/kd_v3p_a.npz
Captions come from the training stream (Sampler3, same augmentation as training). The teacher's sample is a single
coherent drawing per (caption, noise), which removes most of the multimodality that parallel decoding cannot model
(the standard fix for non-autoregressive models). The npz holds captions and code grids only (no images); it stays in
data/build/ (gitignored) anyway, because the codes encode PixelGPT-derived patches.
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spritesheet import tokenizer as T  # noqa: E402
from spritesheet.data.v3 import OUT as V3, Sampler3  # noqa: E402
from spritesheet.patchgpt import PatchGPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--p-short", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.teacher, "config.json")))
    net = PatchGPT(cfg, np.load(os.path.join(V3, "wordvec.npy"))).to(dev)
    net.load_state_dict(torch.load(os.path.join(a.teacher, "model.pt"), map_location=dev))
    net.eval()
    vocab = T.load_vocab(os.path.join(V3, "vocab.txt"))
    s = Sampler3("train", seed=a.seed, p_short=a.p_short)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    caps, codes, t0 = [], [], time.time()
    while len(caps) < a.n:
        c, _ = s.batch(a.batch)
        ids = torch.tensor([T.encode(x, vocab) for x in c], device=dev)
        u = torch.rand(len(c), cfg["T"], cfg["Kp"], generator=g).clamp(1e-9, 1 - 1e-9)
        noise = (-torch.log(-torch.log(u))).to(dev)
        codes.append(net.sample(ids, noise, a.tau).cpu().numpy().astype(np.int16))
        caps += c
        if len(caps) % (a.batch * 20) == 0:
            print(len(caps), f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, caption=np.array(caps[: a.n]), codes=np.concatenate(codes)[: a.n],
                        meta=json.dumps({"teacher": a.teacher, "tau": a.tau, "n": a.n, "p_short": a.p_short}))
    print("wrote", a.out, a.n, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
