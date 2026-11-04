"""Collect every training run (v1, ablations, v2, v3) into artifacts/runs/<name>/ for the slides:
run.json (family, strategy, architecture, data, outcome), train_log.jsonl (loss curve), loss.png;
plus artifacts/runs/index.json and comparison curves artifacts/figures/curves_<family>.png.

v3 runs write their own run dir while training (train_v3.py); this script only adds loss.png and the index.
uv run python scripts/collect_runs.py
"""
import glob, json, os, shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "artifacts")
RUNS = os.path.join(A, "runs")

# name: (family, log file under artifacts/, model dir name or None, strategy, outcome)
REGISTRY = {
    "v32_t0": ("v1", "train_t0.log", "v32_t0", "Tier-0 bring-up: small v1 (C 64/48/32/24), Noto-only concepts, 4k steps", "pipeline check; blobs"),
    "v32_t0b": ("v1", "train_t0b.log", "v32_t0b", "Tier-0 bring-up, wider (C 96/64/48/32), 15k steps", "pipeline check"),
    "v1_small": ("v1", "train_main.log", "v1_small", "First full v1: tier 1 (emoji x7 styles + PixelGPT 20K), C 128/64/48/32, 40k steps",
                 "verified in Sheets (A3/A4); blurry, averaged shapes"),
    "v32_long": ("v1", "train_long.log", None, "v1 at 100k steps (same arch as v1_small)", "stopped at 45.8k steps (superseded by v1-XL)"),
    "v32_emoji": ("v1", "train_emoji.log", None, "v1 trained on emoji only", "stopped at 15.2k steps"),
    "abl_base": ("ablation", "model_abl_base.log", "abl_base", "Ablation baseline: v1 arch, 8k steps, default mix", "reference point for C1-C5"),
    "abl_bow": ("ablation", "model_abl_bow.log", "abl_bow", "Ablation C2: bag-of-words text encoder (no attention/positions)", "worse on word order"),
    "abl_nocliptable": ("ablation", "model_abl_nocliptable.log", "abl_nocliptable", "Ablation C3: learned word embeddings instead of the CLIP table",
                        "worse generalisation to unseen words"),
    "abl_pg5k": ("ablation", "model_abl_pg5k.log", "abl_pg5k", "Ablation C5: only 5k PixelGPT sprites", "data-scaling point"),
    "abl_pg10k": ("ablation", "model_abl_pg10k.log", "abl_pg10k", "Ablation C5: only 10k PixelGPT sprites", "data-scaling point"),
    "abl_emojionly": ("ablation", "model_abl_emojionly.log", "abl_emojionly", "Ablation C5: emoji only (no PixelGPT)", "worse on novel prompts (CLIP R@5)"),
    "abl_pgmajor": ("ablation", "model_abl_pgmajor.log", "abl_pgmajor", "Ablation C5: 70% PixelGPT mixture", "mixture point"),
    "v1_xl": ("v1", "train_v1xl.log", "v1_xl", "v1-XL: C 256/128/64/32 (736k params), 150k steps, 50% PixelGPT, random sprite placement",
              "best v1; deployed as artifacts/model/v32 (D-50)"),
    "v2_mdm_a": ("v2", "train_v2a.log", "v2_mdm_a", "v2a masked diffusion (MaskGIT-style), T=6, emoji only", "blank outputs (background committed first)"),
    "v2_mdm_b": ("v2", "train_v2b.log", "v2_mdm_b", "v2b: + global prompt layout map, 30% PixelGPT, 60k steps", "no blanks; objects shrunken"),
    "v2_mdm_c_pgmajor": ("v2", "train_v2c.log", "v2_mdm_c_pgmajor", "v2c: layout map, 70% PixelGPT, 60k steps",
                         "most sprite-like (outlines); deployed re-calibrated T=4, beta=0.6 (v2c_T4_b06)"),
    "v2_mdm_d_big": ("v2", "train_v2d.log", None, "v2d: wider (C3=192), T=6, 120k steps planned", "stopped at 30k: undeployable size (D-47)"),
}


def read_jsonl(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def loss_key(r):
    for k in ("loss_px", "loss"):
        if k in r:
            return k


def plot(name, rows, path):
    k = loss_key(rows[0]) if rows else None
    if not k:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=140)
    ax.plot([r["step"] for r in rows if k in r], [r[k] for r in rows if k in r], lw=1, label=f"train {k}")
    for vk in ("val_loss", "val_oracle", "val_expected"):
        v = [r for r in rows if vk in r]
        if v:
            ax.plot([r["step"] for r in v], [r[vk] for r in v], "o-", ms=3, lw=1, label=vk)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(name)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(RUNS, exist_ok=True)
    index = {}
    for name, (fam, logf, mdir, strat, outcome) in REGISTRY.items():
        d = os.path.join(RUNS, name)
        os.makedirs(d, exist_ok=True)
        lp = os.path.join(A, logf)
        rows = read_jsonl(lp) if os.path.exists(lp) else []
        if rows:
            with open(os.path.join(d, "train_log.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
        cfg = {}
        if mdir and os.path.exists(os.path.join(A, "model", mdir, "config.json")):
            cfg = json.load(open(os.path.join(A, "model", mdir, "config.json")))
            shutil.copy(os.path.join(A, "model", mdir, "config.json"), os.path.join(d, "config.json"))
        meta = {"name": name, "family": fam, "strategy": strat, "outcome": outcome, "model_dir": mdir and f"artifacts/model/{mdir}",
                "steps_logged": rows[-1]["step"] if rows else None, "params": cfg.get("params"), "train_seconds": cfg.get("train_seconds"),
                "mix": cfg.get("mix"), "arch": {k: cfg[k] for k in ("N", "C0", "C1", "C2", "C3", "E", "T", "S", "K", "d", "h") if k in cfg}}
        json.dump(meta, open(os.path.join(d, "run.json"), "w"), indent=1)
        plot(name, rows, os.path.join(d, "loss.png"))
        index[name] = meta
    for d in sorted(glob.glob(os.path.join(RUNS, "v3*"))):  # v3 runs: own run.json + train_log.jsonl
        name = os.path.basename(d)
        rj = json.load(open(os.path.join(d, "run.json")))
        rows = read_jsonl(os.path.join(d, "train_log.jsonl")) if os.path.exists(os.path.join(d, "train_log.jsonl")) else []
        cfg = json.load(open(os.path.join(d, "config.json"))) if os.path.exists(os.path.join(d, "config.json")) else rj.get("cfg", {})
        rj.update(steps_logged=rows[-1]["step"] if rows else None, params=cfg.get("params"), finished=os.path.exists(os.path.join(d, "config.json")))
        plot(name, rows, os.path.join(d, "loss.png"))
        index[name] = {k: rj.get(k) for k in ("name", "family", "strategy", "steps_logged", "params", "finished", "started")}
    json.dump(index, open(os.path.join(RUNS, "index.json"), "w"), indent=1)
    # comparison curves per family
    for fam in sorted({m["family"] for m in index.values()}):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        for name, m in index.items():
            if m["family"] != fam:
                continue
            p = os.path.join(RUNS, name, "train_log.jsonl")
            if not os.path.exists(p):
                continue
            rows = read_jsonl(p)
            k = loss_key(rows[0]) if rows else None
            if k:
                xs = [r["step"] for r in rows if k in r]
                ys = [r[k] for r in rows if k in r]
                ax.plot(xs, ys, lw=1, label=name)
        ax.set_xlabel("step")
        ax.set_ylabel("training loss")
        ax.set_title(f"{fam} runs: training loss")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(A, "figures", f"curves_{fam}.png"))
        plt.close(fig)
    print(f"{len(index)} runs → {RUNS}/index.json")


if __name__ == "__main__":
    main()
