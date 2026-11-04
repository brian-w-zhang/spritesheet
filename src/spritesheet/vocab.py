"""Vocabulary + frozen CLIP word table (offline distillation of CLIP's text encoder into a lookup table).

uv run python -m spritesheet.vocab --d 32 --common 4000
→ data/build/vocab.txt (row 1 <pad>, row 2 <unk>), data/build/wordvec.npy [V, d] float64, data/build/vocab_report.json
"""
import argparse
import json
import os
import re

import numpy as np
import torch

from . import tokenizer as T
from .reference import round15

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "data", "build")
CLIP_DIR = os.path.join(ROOT, "data", "models", "clip-vit-base-patch32")


def clip_common_words(n):
    """Whole-word tokens from CLIP's BPE vocab, in token-id order (≈ merge order ≈ frequency)."""
    vocab = json.load(open(os.path.join(CLIP_DIR, "vocab.json")))
    words = []
    for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
        if tok.endswith("</w>"):
            w = tok[:-4]
            if re.fullmatch(r"[a-z]{2,20}", w):
                words.append(w)
    return words[:n]


def caption_words(captions):
    s = set()
    for c in captions:
        s.update(T.normalize(c))
    return s


@torch.no_grad()
def clip_text_embed(words, batch=512, template="{}"):
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained(CLIP_DIR).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_DIR)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = m.to(dev)
    out = []
    for i in range(0, len(words), batch):
        enc = tok([template.format(w) for w in words[i:i + batch]], return_tensors="pt", padding=True).to(dev)
        e = m.get_text_features(**enc)
        e = getattr(e, "pooler_output", e)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out).double().numpy()


def pca(X, d):
    mu = X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Z = (X - mu) @ Vt[:d].T
    Z = Z / Z[:, :1].std()  # unit-ish scale on the top component
    explained = float((S[:d] ** 2).sum() / (S ** 2).sum())
    return Z, explained


def build(extra_words, d=32, n_common=4000, template="{}"):
    words = sorted(set(extra_words) | set(clip_common_words(n_common)))
    words = [w for w in words if w and len(w) < T.W - 1 and w not in ("<pad>", "<unk>")]
    E = clip_text_embed(words, template=template)
    Z, explained = pca(E, d)
    vocab = ["<pad>", "<unk>"] + words
    wv = np.zeros((len(vocab), d))
    wv[2:] = Z
    wv[1] = Z.mean(0)  # <unk> = centroid
    wv = round15(wv)
    os.makedirs(BUILD, exist_ok=True)
    open(os.path.join(BUILD, "vocab.txt"), "w").write("\n".join(vocab) + "\n")
    np.save(os.path.join(BUILD, "wordvec.npy"), wv)
    rep = {"V": len(vocab), "d": d, "n_common": n_common, "n_caption_words": len(extra_words),
           "pca_explained": explained, "template": template}
    json.dump(rep, open(os.path.join(BUILD, "vocab_report.json"), "w"), indent=1)
    return vocab, wv, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--common", type=int, default=4000)
    ap.add_argument("--words", default=os.path.join(BUILD, "caption_words.txt"))
    a = ap.parse_args()
    extra = [w.strip() for w in open(a.words)] if os.path.exists(a.words) else []
    print(build(extra, a.d, a.common)[2])
