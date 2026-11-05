"""Merge the visual review (artifacts/bank/review/agent_*.json) into the bank viewer pages and a metrics summary.

  uv run python scripts/bank_merge_review.py
Rebuilds artifacts/bank/index.html and index_local.html from scripts/bank_viewer.html with DATA.review filled in,
and writes artifacts/metrics/bank_review.json (verdict counts per model, good picks, prompts missing a review).
"""
import collections, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "artifacts", "bank")


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(BANK, "review", "agent_*.json"))):
        rows += json.load(open(f))
    tpl = open(os.path.join(ROOT, "scripts", "bank_viewer.html")).read()
    summary = {}
    for name in ("index.html", "index_local.html"):
        path = os.path.join(BANK, name)
        if not os.path.exists(path):
            continue
        data = json.loads(re.search(r"const DATA = (\{.*?\});\n", open(path).read(), re.S).group(1))
        P = len(data["prompts"])
        rev = {k: {} for k in data["models"]}
        for r in rows:
            i = int(r["i"])
            if not 0 <= i < P or r.get("prompt", data["prompts"][i]["prompt"]) != data["prompts"][i]["prompt"]:
                print("skipping mismatched row", r.get("i"), r.get("prompt"))
                continue
            for k in rev:
                if isinstance(r.get(k), dict):
                    rev[k][i] = r[k]
        sl = os.path.join(BANK, "review", "shortlist.json")
        if os.path.exists(sl):
            for k, ids in json.load(open(sl)).items():
                for i in ids if k in rev else []:
                    if i in rev[k]:
                        rev[k][i]["shortlist"] = True
        data["review"] = rev
        open(path, "w").write(tpl.replace("/*DATA*/null", json.dumps(data)))
        summary = {"prompts": P, "rows_reviewed": len(rows), "models": {}}
        for k, d in rev.items():
            c = collections.Counter(v.get("verdict") for v in d.values())
            good = sorted(((data["prompts"][i]["prompt"], v.get("good_seeds", [])) for i, v in d.items()
                           if v.get("verdict") == "good"))
            summary["models"][k] = {"reviewed": len(d), "verdicts": dict(c),
                                    "good_sprites": sum(len(s) for _, s in good),
                                    "shortlist": sum(bool(v.get("shortlist")) for v in d.values()),
                                    "missing": [data["prompts"][i]["prompt"] for i in range(P) if i not in d][:50],
                                    "good_prompts": [p for p, _ in good]}
    json.dump(summary, open(os.path.join(ROOT, "artifacts", "metrics", "bank_review.json"), "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("reviewed", "verdicts", "good_sprites", "shortlist")} for k, v in summary["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
