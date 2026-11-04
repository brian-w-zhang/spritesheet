"""Combine a VQ decoder (e.g. the critic fine-tune) with a prior trained separately on the same frozen codes.

uv run python scripts/graft_prior.py --decoder artifacts/model/v3_vq6_gan --prior artifacts/model/v3_vq6_pg --out artifacts/model/v3_vq6_gan_pg
Valid only when both came from the same encoder + text encoder (train_vq --prior-only and train_vq_gan both freeze them);
the script checks that those tensors are identical.
"""
import argparse, json, os, shutil

import torch

PRIOR = ("p_emb", "p1.", "p2.")
SHARED = ("enc.", "sn.wordvec", "sn.pos", "sn.q.", "sn.k.", "sn.v.", "sn.o.", "sn.f1.", "sn.f2.")  # encoder + text encoder: must match exactly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dd = torch.load(os.path.join(a.decoder, "model.pt"), map_location="cpu")
    pd = torch.load(os.path.join(a.prior, "model.pt"), map_location="cpu")
    for k, v in dd.items():
        if k.startswith(SHARED) and not torch.equal(v, pd[k]):
            raise SystemExit(f"{k} differs between the sources: they do not share an encoder")
    sd = {k: v for k, v in dd.items() if not k.startswith(PRIOR)}
    sd.update({k: v for k, v in pd.items() if k.startswith(PRIOR)})
    dc = json.load(open(os.path.join(a.decoder, "config.json")))
    pc = json.load(open(os.path.join(a.prior, "config.json")))
    cfg = dict(dc, hp=pc["hp"], cond_drop=pc.get("cond_drop", 0.0), graft={"decoder": a.decoder, "prior": a.prior})
    os.makedirs(a.out, exist_ok=True)
    torch.save(sd, os.path.join(a.out, "model.pt"))
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in ("vocab.txt", "palette.json"):
        shutil.copy(os.path.join(a.decoder, f), os.path.join(a.out, f))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
