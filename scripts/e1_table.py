"""Print the E1 comparison table from artifacts/metrics/e1_<name>.json.  uv run python scripts/e1_table.py real v3 ..."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [("naming (VLM, all)", lambda d: d.get("vlm", {}).get("naming_acc", {}).get("all")),
        ("naming creatures", lambda d: d.get("vlm", {}).get("naming_acc", {}).get("creatures")),
        ("naming objects", lambda d: d.get("vlm", {}).get("naming_acc", {}).get("objects")),
        ("naming test", lambda d: d.get("vlm", {}).get("naming_acc", {}).get("test")),
        ("clear (not blob)", lambda d: d.get("vlm", {}).get("clear_share")),
        ("eyes visible", lambda d: d.get("vlm", {}).get("eyes_visible")),
        ("facing acc", lambda d: d.get("vlm", {}).get("facing_acc")),
        ("CLIP R@5 all", lambda d: d["clip_R@5"]["all"]),
        ("colour acc", lambda d: d["colour_acc"]),
        ("giant/tiny area", lambda d: d["size_giant_over_tiny_area"]),
        ("colours/sprite", lambda d: d["colours_per_sprite"]),
        ("fg frac", lambda d: d["fg_frac"]),
        ("distinct /8", lambda d: d["distinct_per_8"]),
        ("near-dup <5%", lambda d: d.get("neardup_share_lt5pct"))]
names = sys.argv[1:]
D = {n: json.load(open(os.path.join(ROOT, "artifacts", "metrics", f"e1_{n}.json"))) for n in names}
print("| metric | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for lab, f in rows:
    vals = []
    for n in names:
        try:
            v = f(D[n])
        except (KeyError, TypeError):
            v = None
        vals.append("—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
    print(f"| {lab} | " + " | ".join(vals) + " |")
