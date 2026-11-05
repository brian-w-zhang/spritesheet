"""Export a trained v3-P (PatchGPT) [+ unconditional prior] into the spec layout (reference_p): weights.npz +
config.json in --out, ready for gates.load_model / push_sheets.py / verify_*.py.

uv run python scripts/export_v3p.py --model artifacts/model/v3p_a --out artifacts/model/v3p_a_sheet [--unet artifacts/model/v3p_u --guidance 2]
uv run python scripts/export_v3p.py --random --cfg '{"dm":128,"H":4,"mlp":256,"layers":4}' --out artifacts/model/v3p_random
"""
import argparse, json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spritesheet import reference_p as RP  # noqa: E402
from spritesheet.patchgpt import PatchGPT, UPatchGPT, GroupGPT  # noqa: E402

V3 = os.path.join(ROOT, "data", "build", "v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--unet", default=None)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--cfg", default='{"dm":128,"H":4,"mlp":256,"layers":4}')
    ap.add_argument("--dict", default=os.path.join(ROOT, "artifacts", "model", "patchdict_k1024_p3"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wv = np.load(os.path.join(V3, "wordvec.npy"))
    if a.random:
        pal = json.load(open(os.path.join(V3, "palette.json")))["colors"]
        pd = json.load(open(os.path.join(a.dict, "patchdict.json")))
        cfg = {"model": "v3p", "N": 24, "L": 12, "T": (24 // pd["P"]) ** 2, "Kp": pd["K"], "P": pd["P"], "K": len(pal),
               "d": int(wv.shape[1]), **json.loads(a.cfg)}
        if "G" in cfg:
            cfg["model"] = "v3pg"
        torch.manual_seed(0)
        net = GroupGPT(cfg, wv) if "G" in cfg else PatchGPT(cfg, wv)
        D = np.load(os.path.join(a.dict, "patches.npy"))
        src = a.dict
    else:
        cfg = json.load(open(os.path.join(a.model, "config.json")))
        net = GroupGPT(cfg, wv) if cfg.get("model") == "v3pg" else PatchGPT(cfg, wv)
        net.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location="cpu"))
        D = np.load(os.path.join(a.model, "patches.npy"))
        src = a.model
    unet = None
    if a.unet:
        ucfg = json.load(open(os.path.join(a.unet, "config.json")))
        unet = GroupGPT(ucfg) if "G" in ucfg else UPatchGPT(ucfg)
        unet.load_state_dict(torch.load(os.path.join(a.unet, "model.pt"), map_location="cpu"))
    w, rc = (RP.export_g if "G" in cfg else RP.export)(net.eval(), D, unet=unet.eval() if unet else None,
                      extra_cfg={"guidance": a.guidance if unet else 0.0, "tau": a.tau,
                                 "source": "random" if a.random else a.model, "unet": a.unet})
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "weights.npz"), **w)
    if not a.random:
        rc.update({k: cfg[k] for k in ("steps", "params", "dict_dir") if k in cfg})
    json.dump(rc, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in ("vocab.txt", "palette.json"):
        s = os.path.join(src, f) if os.path.exists(os.path.join(src, f)) else os.path.join(V3, f)
        shutil.copy(s, os.path.join(a.out, f))
    print("exported", a.out, {k: rc[k] for k in ("dm", "layers", "guidance", "tau")})


if __name__ == "__main__":
    main()
