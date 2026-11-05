"""Turn a mid-run PatchGPT checkpoint into a model dir (for runs stopped early), so eval/export can load it.

uv run python scripts/ckpt_to_model.py --run artifacts/runs/v3pg_a --ckpt ckpt_030000.pt --out artifacts/model/v3pg_a
config.json gets steps = the checkpoint step and stopped_early = true (the LR schedule was not annealed).
"""
import argparse, json, os, shutil, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, "data", "build", "v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = json.load(open(os.path.join(a.run, "run.json")))
    cfg = dict(run["cfg"])
    step = int(a.ckpt.split("_")[1].split(".")[0])
    sd = torch.load(os.path.join(a.run, a.ckpt), map_location="cpu")
    log = [json.loads(l) for l in open(os.path.join(a.run, "train_log.jsonl"))]
    log = [r for r in log if r["step"] <= step]
    dict_dir = run["args"]["dict"]
    os.makedirs(a.out, exist_ok=True)
    cfg.update(steps=step, stopped_early=True, planned_steps=run["args"]["steps"], dict_dir=dict_dir,
               params=int(sum(v.numel() for v in sd.values())), final_log=log[-1])
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))
    torch.save(sd, os.path.join(a.out, "model.pt"))
    np.save(os.path.join(a.out, "patches.npy"), np.load(os.path.join(ROOT, dict_dir, "patches.npy")))
    for f in ("vocab.txt", "palette.json"):
        shutil.copy(os.path.join(V3, f), os.path.join(a.out, f))
    json.dump({"stopped_at": step, "reason": "stopped early to free the GPU for distillation; baseline only"},
              open(os.path.join(a.run, "STOPPED.json"), "w"))
    print("wrote", a.out, "step", step)


if __name__ == "__main__":
    main()
