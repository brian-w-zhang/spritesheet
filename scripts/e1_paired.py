"""Paired bootstrap over prompts for VLM judge metrics: is model B better than A?

uv run python scripts/e1_paired.py v3 v4s   →  artifacts/metrics/e1_paired_v3__v4s.json
Each prompt's score is the mean over judged seeds; prompts are resampled (10k draws), so the CI accounts for
prompts differing in difficulty and for seeds of one prompt being correlated."""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "artifacts", "eval", "e1")


def per_prompt(name, task):
    v = json.load(open(os.path.join(EV, name, "vlm.json")))
    return np.array([np.mean(r) for r in v[task]])


def main(a, b, n=10000):
    rng = np.random.default_rng(0)
    out = {"a": a, "b": b, "draws": n}
    for task in ("mc", "blob", "eyes", "facing"):
        x, y = per_prompt(a, task), per_prompt(b, task)
        d = y - x
        idx = rng.integers(0, len(d), (n, len(d)))
        bs = d[idx].mean(1)
        out[task] = {"prompts": len(d), "a": round(float(x.mean()), 4), "b": round(float(y.mean()), 4),
                     "diff": round(float(d.mean()), 4), "ci95": [round(float(q), 4) for q in np.percentile(bs, [2.5, 97.5])],
                     "p_b_not_better": round(float((bs <= 0).mean()), 4),
                     "prompts_b_better": int((d > 0).sum()), "prompts_a_better": int((d < 0).sum())}
        print(task, out[task])
    json.dump(out, open(os.path.join(ROOT, "artifacts", "metrics", f"e1_paired_{a}__{b}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
