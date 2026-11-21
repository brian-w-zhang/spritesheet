"""Blind VLM judge for the E1 eval (run in tools/vlm, its own env):

  cd tools/vlm && uv run python vlm_judge.py --name v3 [--model mlx-community/Qwen3-VL-8B-Instruct-4bit] [--limit N]

For every sprite in artifacts/eval/e1/<name>/grids.npz (prompt x seed), the judge sees ONLY the image (never the
prompt) and answers:
  mc      multiple choice among 5 nouns (the prompt's head noun + 4 fixed distractors, order fixed per prompt/seed)
  blob    "clear drawing" vs "unrecognizable blob"
  eyes    creature prompts (creatures, pose, facing): are eyes visible
  facing  facing prompts: left or right
Writes artifacts/eval/e1/<name>/vlm.json (+ raw answers). Deterministic (greedy decoding).
"""
import argparse, json, os, sys, time

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EV = os.path.join(ROOT, "artifacts", "eval", "e1")
BG = np.array([0xD8, 0xD4, 0xDD], np.uint8)
LET = "ABCDE"


def palette_rgb():
    pal = json.load(open(os.path.join(ROOT, "data", "build", "v3", "palette.json")))["colors"]
    return np.concatenate([BG[None], np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in pal], np.uint8)])


class Judge:
    def __init__(self, repo):
        from mlx_vlm import load
        self.model, self.proc = load(repo)
        self.cfg = self.model.config

    def ask(self, img_path, q, max_tokens=3):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        p = apply_chat_template(self.proc, self.cfg, q, num_images=1)
        r = generate(self.model, self.proc, p, image=[img_path], max_tokens=max_tokens, temperature=0.0, verbose=False)
        return (r.text if hasattr(r, "text") else str(r)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--tasks", default="mc,blob,eyes,facing")
    ap.add_argument("--seeds", type=int, default=4, help="judge the first N of the 8 seeds (memory/time; CLIP uses all 8)")
    ap.add_argument("--mem-gb", type=float, default=7.0, help="hard MLX memory cap (the laptop has 16 GB shared)")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, "eval", "prompts_e1.json")))
    import mlx.core as mx
    mx.set_memory_limit(int(a.mem_gb * 2**30))
    mx.set_cache_limit(int(1 * 2**30))
    P, S = d["prompts"], min(a.seeds, len(d["meta"]["seeds"]))
    if a.limit:
        P = P[: a.limit]
    grids = np.load(os.path.join(EV, a.name, "grids.npz"))["grids"]
    rgb = palette_rgb()
    J = Judge(a.model)
    tmp = os.path.join(EV, a.name, "_tmp.png")
    tasks = a.tasks.split(",")
    out = {"seeds_judged": S, "model": a.model, "mc": [], "blob": [], "eyes": [], "facing": [], "raw": [], "done": 0}
    part = os.path.join(EV, a.name, "vlm_partial.json")
    if os.path.exists(part):
        out = json.load(open(part))
        assert out["model"] == a.model
        print("resuming at prompt", out["done"], flush=True)
    t0 = time.time()
    for i, p in enumerate(P):
        if i < out["done"]:
            continue
        rows = {k: [] for k in ("mc", "blob", "eyes", "facing")}
        for s in range(S):
            Image.fromarray(rgb[grids[i, s]]).resize((240, 240), Image.NEAREST).save(tmp)
            raw = {}
            if "mc" in tasks:
                opts = [p["noun"]] + p["distractors"]
                order = np.random.default_rng([i, s]).permutation(5)
                opts = [opts[k] for k in order]
                q = ("This is a small pixel-art sprite. Which one of these does it most likely show?\n"
                     + "\n".join(f"{LET[k]}) {o}" for k, o in enumerate(opts)) + "\nAnswer with the letter only.")
                ans = J.ask(tmp, q)
                raw["mc"] = ans
                rows["mc"].append(ans[:1].upper() == LET[int(np.where(order == 0)[0][0])])
            if "blob" in tasks:
                ans = J.ask(tmp, "Is this pixel-art sprite a clear, recognizable drawing of a specific thing, or an "
                                 "unrecognizable blob? Answer with one word: clear or blob.")
                raw["blob"] = ans
                rows["blob"].append(ans.lower().startswith("clear"))
            if "eyes" in tasks and p["group"] in ("creatures", "pose", "facing"):
                ans = J.ask(tmp, "Look closely at this pixel-art sprite. Does it show a creature with visible eyes? "
                                 "Answer yes or no.")
                raw["eyes"] = ans
                rows["eyes"].append(ans.lower().startswith("yes"))
            if "facing" in tasks and p["group"] == "facing":
                ans = J.ask(tmp, "This pixel-art sprite shows an animal. Is it facing left or right? Answer with one "
                                 "word: left or right.")
                raw["facing"] = ans
                rows["facing"].append(ans.lower().startswith(p["prompt"].split()[-1]))
            out["raw"].append({"i": i, "s": s, **raw})
        for k in rows:
            if rows[k]:
                out[k].append(rows[k])
        out["done"] = i + 1
        if i % 5 == 0:
            json.dump(out, open(part, "w"))
        if i % 10 == 0:
            mc = np.mean([x for r in out["mc"] for x in r]) if out["mc"] else float("nan")
            print(f"{i + 1}/{len(P)} {time.time() - t0:.0f}s mc={mc:.3f}", flush=True)
    os.remove(tmp)
    if os.path.exists(part):
        os.remove(part)
    out["seconds"] = round(time.time() - t0, 1)
    out["eyes_groups"] = ["creatures", "pose", "facing"]
    name = "vlm.json" if not a.limit else "vlm_smoke.json"
    json.dump(out, open(os.path.join(EV, a.name, name), "w"))
    print("done", out["seconds"], "s")


if __name__ == "__main__":
    main()
