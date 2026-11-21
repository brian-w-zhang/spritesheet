"""Training step time of PatchGPT sizes on this machine (MPS), fp32 vs bf16 autocast. Sizing aid for teachers.
uv run python scripts/bench_train_speed.py"""
import sys, os, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from spritesheet.patchgpt import PatchGPT
wv = np.load("data/build/v3/wordvec.npy")
dev = "mps"
for cfg, B, T in [({"dm": 128, "H": 4, "mlp": 256, "layers": 4}, 256, 64), ({"dm": 256, "H": 8, "mlp": 1024, "layers": 6}, 256, 64),
                  ({"dm": 384, "H": 6, "mlp": 1536, "layers": 8}, 256, 64), ({"dm": 256, "H": 8, "mlp": 1024, "layers": 6}, 256, 144)]:
    net = PatchGPT({"L": 12, "T": T, "Kp": 1024, **cfg}, wv).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-4)
    ids = torch.randint(2, 100, (B, 12), device=dev); codes = torch.randint(0, 1024, (B, T), device=dev)
    for dt in [torch.float32, torch.bfloat16]:
        ts = []
        for i in range(6):
            t0 = time.time()
            with torch.autocast("mps", dtype=dt, enabled=dt != torch.float32):
                loss = F.cross_entropy(net.logits(ids, codes).flatten(0, 1).float(), codes.flatten())
            opt.zero_grad(); loss.backward(); opt.step(); torch.mps.synchronize(); ts.append(time.time() - t0)
        print(cfg, "T", T, "B", B, str(dt), "params", sum(p.numel() for p in net.parameters()), "s/step", round(float(np.median(ts[2:])), 3), flush=True)
