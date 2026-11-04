"""C4: SpriteX32 vs SpriteSheet on the same prompts, scored with the same CLIP metrics.

uv run python scripts/bench_spritex32.py --model artifacts/model/v32 --n 120
SpriteX32's pipeline.py was read before use (plain PyTorch; its only download is the CLIP text encoder,
redirected here to the local copy in data/models/).
"""
import argparse, importlib.util, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from spritesheet import evaluate as E, tokenizer as T  # noqa: E402

SX = os.path.join(ROOT, "data", "models", "SpriteX32")


def load_spritex32(device):
    spec = importlib.util.spec_from_file_location("sx_pipeline", os.path.join(SX, "pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from safetensors.torch import load_file
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    cfg = json.load(open(os.path.join(SX, "config.json")))
    unet = mod.UNet(ch=cfg["base_channels"], mults=tuple(cfg["channel_mults"]), text_dim=cfg["text_dim"],
                    emb=cfg["embed_dim"], text_scale=cfg["text_scale"])
    unet.load_state_dict(load_file(os.path.join(SX, "model.safetensors")))
    tok = CLIPTokenizer.from_pretrained(E.CLIP_DIR)
    te = CLIPTextModelWithProjection.from_pretrained(E.CLIP_DIR)
    pipe = mod.SpriteFlowPipeline(unet, tok, te, cfg, device)
    n_unet = sum(p.numel() for p in unet.parameters())
    n_te = sum(p.numel() for p in te.parameters())
    return pipe, cfg, n_unet, n_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--sets", default="heldout_combo,unseen_words,pixelgpt_test")
    a = ap.parse_args()
    from spritesheet.data.synth import eval_set
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe, sxcfg, n_unet, n_te = load_spritex32(dev)
    net, cfg = E.load_net(a.model)
    vocab = T.load_vocab(os.path.join(E.BUILD, "vocab.txt"))
    clip = E.Clip()
    concepts = json.load(open(os.path.join(E.BUILD, "concepts.json")))
    names = [c["id"] for c in concepts]
    ctxt = clip.txt([f"an emoji of {c['name']}" for c in concepts])
    rgb = E.palette_rgb()
    out = {"spritex32": {"unet_params": n_unet, "text_encoder_params": n_te, "steps": sxcfg["default_steps"],
                         "guidance": sxcfg["default_guidance_scale"], "device": dev},
           "spritesheet": {"params": cfg.get("params"), "steps": 1}, "sets": {}}
    examples = []
    for s in a.sets.split(","):
        data = eval_set(s, a.n, tier=int(cfg.get("tier", 0)))
        caps = [d[0] for d in data]
        metas = [d[3] for d in data]
        t0 = time.time()
        sx_imgs = []
        for i in range(0, len(caps), 16):
            ims = pipe(caps[i:i + 16], num_images=1, seed=i)
            sx_imgs.extend(np.array(im.convert("RGB")) for im in ims)
        sx_s = (time.time() - t0) / len(caps)
        t0 = time.time()
        ours, _ = E.generate(net, caps, vocab, [1 + i for i in range(len(caps))])
        ours_s = (time.time() - t0) / len(caps)
        pt = clip.txt(caps)
        res = {}
        for name, imgs in (("spritex32", sx_imgs), ("spritesheet", [E.to_rgb(g, rgb) for g in ours])):
            ie = clip.img(imgs)
            r = {"clip_score": float((ie * pt).sum(1).mean())}
            cids = [m.get("concept") for m in metas]
            if cids and all(c in names for c in cids):
                gold = torch.tensor([names.index(c) for c in cids])
                top = (ie @ ctxt.T).topk(5, dim=1).indices
                r["clip_concept_top1"] = float((top[:, 0] == gold).float().mean())
                r["clip_concept_top5"] = float((top == gold[:, None]).any(1).float().mean())
            res[name] = r
        res["seconds_per_image"] = {"spritex32_torch": sx_s, "spritesheet_torch": ours_s}
        out["sets"][s] = res
        examples.extend((caps[i], sx_imgs[i], E.to_rgb(ours[i], rgb)) for i in range(min(4, len(caps))))
        print(s, json.dumps(res), flush=True)
    os.makedirs(os.path.join(ROOT, "artifacts", "metrics"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "metrics", "spritex32.json"), "w"), indent=1)
    np.save(os.path.join(ROOT, "artifacts", "tmp_sx_examples.npy"), np.array(examples, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
