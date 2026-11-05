"""Merge the visual review (artifacts/bank/review/agent_*.json) into the bank viewer pages and a metrics summary.

  uv run python scripts/bank_merge_review.py
Rebuilds artifacts/bank/index.html and index_local.html from scripts/bank_viewer.html with DATA.review filled in,
and writes artifacts/metrics/bank_review.json (verdict counts per model, good picks, prompts missing a review).
"""
import collections, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(ROOT, "artifacts", "bank")
REVIEW = os.path.join(BANK, "review")


def main():
    global BANK
    if len(sys.argv) > 1:   # another bank folder (reviews still come from artifacts/bank/review)
        BANK = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(REVIEW, "agent_*.json"))):
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
        pidx = {p["prompt"]: i for i, p in enumerate(data["prompts"])}
        for r in rows:   # matched by prompt text (indices can shift when the bank is regenerated)
            i = pidx.get(r.get("prompt"))
            j = r.get("i")
            if i is None and isinstance(j, int) and 0 <= j < P and (data["prompts"][j]["prompt"].startswith(str(r.get("prompt", ""))[:12])
                    or data["prompts"][j]["prompt"].split()[-1:] == str(r.get("prompt", "")).split()[-1:]):
                i = j   # truncated prompt text on the sheet: trust the row number
            if i is None:
                print("skipping row with unknown prompt", r.get("i"), r.get("prompt"))
                continue
            for k in rev:
                if isinstance(r.get(k), dict):
                    rev[k][i] = r[k]
        sl = os.path.join(REVIEW, "shortlist.json")
        if os.path.exists(sl):
            for k, names in json.load(open(sl)).items():
                for nm in names if k in rev else []:
                    i = pidx.get(nm)
                    if i is not None and i in rev[k]:
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
    if len(sys.argv) == 1:
        json.dump(summary, open(os.path.join(ROOT, "artifacts", "metrics", "bank_review.json"), "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("reviewed", "verdicts", "good_sprites", "shortlist")} for k, v in summary["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
