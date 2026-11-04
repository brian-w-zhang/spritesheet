"""A5 gate: data pipeline (docs/03, docs/08). Needs data/raw/ and a tier build
(`uv run python -m spritesheet.data.build --tier 0` / `--tier 1`); tests skip when those are missing."""
import json
import os

import numpy as np
import pytest

from spritesheet.data import BUILD, RAW, build_dir

HAVE_RAW = os.path.isdir(os.path.join(RAW, "noto-emoji"))
PAL_PATH = os.path.join(BUILD, "palette.json")


def built(tier):
    return os.path.exists(os.path.join(build_dir(tier), "bases_32.npz"))


def any_tier():
    for t in (1, 0):
        if built(t):
            return t
    pytest.skip("no tier build present")


@pytest.mark.skipif(not (HAVE_RAW and os.path.exists(PAL_PATH)), reason="raw data or palette missing")
def test_build_deterministic(tmp_path):
    from spritesheet.data.build import build
    from spritesheet.data.normalize import Palette
    pal = Palette.load(PAL_PATH)
    reps = []
    for k in range(2):
        out = tmp_path / f"b{k}"
        reps.append(build(0, str(out), limit=12, palette=pal, workers=2 if k else 1, write_root=False, quiet=True))
    assert reps[0]["arrays_sha256"] == reps[1]["arrays_sha256"]
    a, b = np.load(tmp_path / "b0" / "bases_32.npz"), np.load(tmp_path / "b1" / "bases_32.npz")
    for key in a.files:
        assert np.array_equal(a[key], b[key]), key
    assert (tmp_path / "b0" / "splits.json").read_text() == (tmp_path / "b1" / "splits.json").read_text()
    assert (tmp_path / "b0" / "concepts.json").read_text() == (tmp_path / "b1" / "concepts.json").read_text()


@pytest.mark.skipif(not built(1), reason="tier 1 build missing")
def test_pixelgpt_padded_1to1():
    from spritesheet.data import sources as S
    from spritesheet.data.synth import load_build
    D = load_build(1)
    pal = D["palette"]
    rows = S.pixelgpt_rows("test")[:200]
    keys = {str(k): i for i, k in enumerate(D["key"])}
    for r in rows:
        g = D["imgs"][keys["pixelgpt/" + r["key"]]]
        assert np.array_equal(g[4:28, 4:28], pal.snap(r["rgba"])), "24x24 core must equal the masked+snapped source"
        border = g.copy()
        border[4:28, 4:28] = 0
        assert not border.any(), "padding must be transparent (class 0)"
        # tile_bg pixels are transparent
        assert (g[4:28, 4:28][r["rgba"][..., 3] == 0] == 0).all()


def test_pad_pixel_art_is_exact():
    from spritesheet.data.normalize import pad_pixel_art
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (24, 24, 4), dtype=np.uint8)
    src[..., 3] = np.where(rng.random((24, 24)) < 0.5, 0, 255)
    out = pad_pixel_art(src, 32)
    on = src[..., 3] == 255
    assert np.array_equal(out[4:28, 4:28][on], src[on])
    assert (out[4:28, 4:28][~on] == 0).all()
    assert out[:4].sum() == out[28:].sum() == out[:, :4].sum() == out[:, 28:].sum() == 0


@pytest.mark.skipif(not os.path.exists(PAL_PATH), reason="palette missing")
def test_palette_snap_idempotent():
    from spritesheet.data.normalize import Palette
    pal = Palette.load(PAL_PATH)
    assert pal.K == len(json.load(open(PAL_PATH))["colors"])
    rng = np.random.default_rng(1)
    rgba = rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)
    ids = pal.snap(rgba)
    assert ids.max() <= pal.K and ((ids == 0) == (rgba[..., 3] < 128)).all()
    again = pal.snap(pal.to_rgba(ids))
    assert np.array_equal(ids, again)
    # every palette colour snaps to itself
    assert np.array_equal(pal.nearest(pal.rgb), np.arange(pal.K))


def test_sampler_deterministic():
    from spritesheet.data.synth import Sampler
    t = any_tier()
    a = Sampler("train", seed=123, tier=t).batch(300)
    b = Sampler("train", seed=123, tier=t).batch(300)
    c = Sampler("train", seed=124, tier=t).batch(300)
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and a[3] == b[3]
    assert a[0] != c[0]
    assert a[1].dtype == np.uint8 and a[1].shape == (300, 32, 32)


@pytest.mark.parametrize("tier", [0, 1])
def test_heldout_combos_never_in_train(tier):
    from spritesheet.data.synth import COLORS, TRANSFORMS, Sampler
    if not built(tier):
        pytest.skip(f"tier {tier} build missing")
    s = Sampler("train", seed=0, tier=tier)
    hc, ht = s.held_color, s.held_tf
    assert hc and ht
    seen_concepts, seen_col, seen_tf = set(), set(), set()
    for _ in range(40):  # 20k samples
        _, _, _, meta = s.batch(500)
        for m in meta:
            if m["concept"] is None:
                continue
            seen_concepts.add(m["concept"])
            assert (m["concept"], m["colour"]) not in hc
            assert (m["concept"], m["transform"]) not in ht
            seen_col.add(m["colour"])
            seen_tf.add(m["transform"])
    assert set(COLORS) <= seen_col and set(TRANSFORMS) <= seen_tf
    # every held-out pair's concept and value is seen in training
    assert {c for c, _ in hc | ht} <= {s.concepts[i]["id"] for i in s.cidx}
    if tier == 0:
        assert {c for c, _ in hc | ht} <= seen_concepts
    # and the heldout_combo eval split only produces held-out pairs
    h = Sampler("heldout_combo", seed=1, tier=tier)
    for m in h.batch(300)[3]:
        assert (m["concept"], m["colour"]) in hc or (m["concept"], m["transform"]) in ht


def test_test_keywords_unused_in_train_captions():
    from spritesheet.data.synth import Sampler, words
    t = any_tier()
    s = Sampler("train", seed=5, tier=t)
    test_words = {k for c in s.concepts for k in c["keywords_test"]}
    assert test_words
    for _ in range(10):
        caps = s.batch(500)[0]
        for cap in caps:
            assert not (set(words(cap)) & test_words), cap


def test_targets_match_meta():
    """Exact-target sanity: mirrored = flipped base, background fills only class-0 pixels."""
    from spritesheet.data.synth import Sampler
    t = any_tier()
    s = Sampler("train", seed=9, tier=t)
    caps, tg, st, meta = s.batch(2000)
    for g, m in zip(tg, meta):
        b = m["base_idx"]
        if m["kind"] == "emoji_plain":
            assert np.array_equal(g, s.D["imgs"][b])
        if m["kind"] != "pixelgpt" and m["transform"] == "mirror" and not m["colour"] and not m["size"] and not m["bg"]:
            assert np.array_equal(g, s.D["imgs"][b][:, ::-1])
        if m["bg"] is not None:
            assert (g > 0).all()
        assert st.dtype == np.int64
