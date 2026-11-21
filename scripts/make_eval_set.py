"""Build the fixed evaluation set (E1): eval/prompts_e1.json. Run once; the file is committed and never edited in
place (a changed set gets a new name), so every model is scored on exactly the same prompts and seeds.

Groups (each prompt has a head noun for the blind multiple-choice naming test, and 4 fixed distractor nouns):
  creatures   common animals/monsters, judged also on "are eyes visible"
  objects     common items
  colour      colour + noun, mostly combinations rare or absent in the captions
  size        tiny / giant + noun (pairs, so area can be compared within a noun)
  facing      noun + facing left / right (pairs)
  pose        creature + pose
  test        verbatim held-out PixelGPT test captions (have a real sprite: the ceiling)
For every prompt: n_train = training captions containing the phrase (word-boundary match), n_noun = containing the noun.

uv run python scripts/make_eval_set.py
"""
import json, os, re, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CREATURES = "cat dog frog fox owl bird fish snake turtle rabbit bear wolf mouse pig duck bat spider crab snail bee " \
            "dragon slime ghost skeleton wizard knight robot penguin octopus mushroom".split()
CREATURES.remove("mushroom")
OBJECTS = "sword shield axe bow potion key crown coin gem candle lantern book scroll chest barrel helmet boot hat " \
          "bell clock wand anchor rocket boat house castle tree apple cake pizza".split()
COLOURS = ["red", "blue", "green", "yellow", "purple", "pink", "black", "white", "orange", "golden"]
COLOUR_NOUNS = ["cat", "frog", "sword", "potion", "crown", "mushroom", "fish", "car", "bird", "key", "tree", "ghost",
                "dragon", "boot", "apple", "slime", "hat", "flower", "bat", "house"]
SIZE_NOUNS = ["cat", "dog", "tree", "sword", "potion", "mushroom", "house", "fish"]
FACE_NOUNS = ["cat", "dog", "fox", "bird", "fish", "horse", "duck", "rabbit"]
POSES = [("sleeping", "cat"), ("flying", "bird"), ("running", "dog"), ("sitting", "frog"), ("jumping", "rabbit"),
         ("swimming", "fish"), ("flying", "dragon"), ("sleeping", "fox")]


def main():
    from spritesheet.data.v3 import OUT as V3, PREP
    z = np.load(os.path.join(V3, "imgs_24.npz"))
    caps, split = z["caption"], z["split"]
    train = [" " + re.sub(r"[^a-z ]", " ", str(c).lower()) + " " for c in caps[split == 0]]

    def count(phrase):
        p = f" {phrase} "
        return sum(p in c for c in train)

    rng = np.random.default_rng(1)
    items = []
    for n in CREATURES:
        items.append(("creatures", f"a {n}", n))
    for n in OBJECTS:
        items.append(("objects", f"a {n}", n))
    for i, n in enumerate(COLOUR_NOUNS):
        for c in (COLOURS[i % 10], COLOURS[(i + 3) % 10]):
            items.append(("colour", f"a {c} {n}", n))
    for n in SIZE_NOUNS:
        for s in ("tiny", "giant"):
            items.append(("size", f"a {s} {n}", n))
    for n in FACE_NOUNS:
        for s in ("left", "right"):
            items.append(("facing", f"a {n} facing {s}", n))
    for p, n in POSES:
        items.append(("pose", f"a {p} {n}", n))
    # held-out PixelGPT test captions: 48 fixed, short-ish ones (<= 5 words) so the head noun is clear
    tz = np.nonzero(split == 2)[0]
    pick = [int(i) for i in rng.permutation(tz) if 2 <= len(str(caps[i]).split()) <= 5][:48]
    nouns_all = sorted(set(CREATURES + OBJECTS + COLOUR_NOUNS + SIZE_NOUNS + FACE_NOUNS + [n for _, n in POSES]))
    out = []
    for g, p, n in items:
        phrase = p.split(" ", 1)[1]
        out.append({"group": g, "prompt": p, "noun": n, "n_train": count(phrase), "n_noun": count(n)})
    for i in pick:
        c = str(caps[i])
        w = c.lower().split()
        cut = next((k for k, x in enumerate(w) if k > 0 and x in PREP), len(w))   # head noun: last word before a preposition
        out.append({"group": "test", "prompt": c, "noun": w[cut - 1], "n_train": count(c.lower()), "test_index": i})
    # fixed distractors: 2 from the same group's nouns, 2 from all nouns (never the noun itself)
    for it in out:
        same = sorted({o["noun"] for o in out if o["group"] == it["group"]} - {it["noun"]})
        pool = [x for x in nouns_all if x != it["noun"]]
        d = list(rng.choice(same, 2, replace=False)) if len(same) >= 2 else []
        while len(d) < 4:
            x = str(rng.choice(pool))
            if x not in d:
                d.append(x)
        it["distractors"] = [str(x) for x in d]
    meta = {"name": "e1", "seeds": list(range(1, 9)), "n_prompts": len(out),
            "groups": {g: sum(o["group"] == g for o in out) for g in dict.fromkeys(o["group"] for o in out)},
            "note": "fixed set; never edit in place. test prompts are verbatim PixelGPT test-split captions."}
    os.makedirs(os.path.join(ROOT, "eval"), exist_ok=True)
    json.dump({"meta": meta, "prompts": out}, open(os.path.join(ROOT, "eval", "prompts_e1.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    for g in meta["groups"]:
        v = [o["n_train"] for o in out if o["group"] == g]
        print(g, "phrase in 0 train captions:", sum(x == 0 for x in v), "/", len(v))


if __name__ == "__main__":
    main()
