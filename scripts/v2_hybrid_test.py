"""Hypothesis: v1 one-shot gives global shape; v2 masked diffusion refines low-confidence pixels.
Compare like-for-like on the same eval data: v1 alone, v2 alone (best sweep variant), hybrid(keep frac q)."""
import json, math, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spritesheet import evaluate as E, tokenizer as T  # noqa
from compare_models import PROMPTS, load_v2  # noqa
from v2_sampler_sweep import sample  # noqa
from make_figures import font, INK, PAPER  # noqa
from PIL import Image, ImageDraw
from spritesheet.data.synth import eval_set
net1, _ = E.load_net(os.path.join(ROOT, "artifacts/model/v32"))
net2, cfg2 = load_v2(os.path.join(ROOT, "artifacts/model/v2_mdm_a"))
vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt")); rgb = E.palette_rgb()
K = cfg2["K"]; M = K + 1; N = 32

@torch.no_grad()
def hybrid(caps, sty, q, steps=3):
    ids = torch.tensor([T.encode(c, vocab) for c in caps])
    c1 = net1.encode(ids); lg1 = net1.decode(c1, sty).permute(0, 2, 3, 1)
    p1 = torch.softmax(lg1, -1); conf, am = p1.max(-1)
    x = torch.full(am.shape, M, dtype=torch.long)
    for b in range(len(caps)):
        k = int(q * N * N); idx = torch.topk(conf[b].flatten(), k).indices
        flat = x[b].flatten(); flat[idx] = am[b].flatten()[idx]; x[b] = flat.view(N, N)
    _, cond = net2.cond(ids, sty)
    for t in range(1, steps + 1):
        lg = net2.denoise(x, cond); p = torch.softmax(lg, -1); cf, a2 = p.max(-1)
        masked = x == M; remain = masked.sum((1, 2))
        for b in range(len(caps)):
            n_commit = int(math.ceil(remain[b].item() / (steps - t + 1)))
            if n_commit <= 0: continue
            sc = torch.where(masked[b], cf[b], torch.full_like(cf[b], -1)).flatten()
            idx = torch.topk(sc, n_commit).indices
            flat = x[b].flatten(); flat[idx] = a2[b].flatten()[idx]; x[b] = flat.view(N, N)
    return x.numpy().astype(np.uint8)

data = eval_set("heldout_combo", 150, tier=1) + eval_set("train_sample", 150, tier=1)
caps = [d[0] for d in data]; tg = np.stack([d[1] for d in data]); sid = torch.tensor([d[2] for d in data])
fg = tg != 0
def score(o): return {"fg_acc": round(float(((o == tg) & fg).sum() / fg.sum()), 3), "px_acc": round(float((o == tg).mean()), 3),
                      "fg_area_ratio": round(float((o != 0).sum() / fg.sum()), 3)}
res = {}
with torch.no_grad():
    o1 = net1.decode(net1.encode(torch.tensor([T.encode(c, vocab) for c in caps])), sid).argmax(1).numpy()
res["v1_one_shot"] = score(o1)
for q in (0.5, 0.7, 0.85):
    res[f"hybrid_q{q}"] = score(np.concatenate([hybrid(caps[i:i+64], sid[i:i+64], q) for i in range(0, len(caps), 64)]))
for k, v in res.items(): print(k, v)
json.dump(res, open(os.path.join(ROOT, "artifacts/metrics/v2_hybrid_test.json"), "w"), indent=1)
g1 = net1.decode(net1.encode(torch.tensor([T.encode(p, vocab) for p in PROMPTS])), torch.zeros(len(PROMPTS), dtype=torch.long)).argmax(1).numpy()
gh = hybrid(PROMPTS, torch.zeros(len(PROMPTS), dtype=torch.long), 0.7)
cs = 100; im = Image.new("RGB", (12 * (cs + 8) + 16, 4 * (cs + 26) + 50), PAPER); d = ImageDraw.Draw(im)
d.text((16, 12), "rows 1-2: v1 one-shot   rows 3-4: hybrid (v1 keeps 70% most-confident px, v2 refines 30% in 3 steps)", fill=INK, font=font(20))
for j in range(24):
    r, c = divmod(j, 12)
    for blk, g in ((0, g1), (2, gh)):
        y = 44 + (r + blk) * (cs + 26)
        im.paste(Image.fromarray(E.to_rgb(g[j], rgb)).resize((cs, cs), Image.NEAREST), (16 + c * (cs + 8), y))
        d.text((16 + c * (cs + 8), y + cs + 3), PROMPTS[j][:16], fill=INK, font=font(12))
im.save("/private/tmp/claude-501/-Users-junzhang-Projects-spritesheet/9fcd60fd-03ae-433d-9143-938fc3dec60b/scratchpad/hybrid.png")
