"""SpriteSheet data pipeline: raw sources -> normalised N x N class-id grids -> synthetic captions.

Modules: `sources` (raw loaders + concept table), `normalize` (rendering, palette, snapping),
`build` (CLI that writes data/build/...), `synth` (Sampler / eval_set). Spec: docs/03, docs/08.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(ROOT, "data", "raw")
BUILD = os.path.join(ROOT, "data", "build")


def build_dir(tier: int) -> str:
    """Directory holding one tier's build (palette.json, styles.json, concepts.json, bases_N.npz, splits.json)."""
    return os.path.join(BUILD, f"tier{tier}")
