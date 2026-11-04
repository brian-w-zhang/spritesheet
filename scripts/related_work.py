"""E: related-work table. Facts ONLY from sources listed as opened in docs/02-prior-art.md. Unknown → "not stated".
SpriteSheet rows are filled from artifacts/metrics/{systems,parity_local}.json when present."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "artifacts", "metrics")
NS = "not stated"
rows = [
    {"project": "SpriteX32", "source": "https://huggingface.co/viknesh100/SpriteX32",
     "medium": "PyTorch (CPU/GPU)", "task": "text → 32×32 pixel-art sprite",
     "runs_inside_medium": "full model", "model_size": "11.4M-param U-Net + frozen CLIP ViT-B/32 text encoder",
     "steps": "50 Euler steps (default), CFG 3.0", "correctness_shown": "qualitative only (card: no FID computed)",
     "training_data": "tkarr/sprite_caption_dataset, 12,830 pairs (Pokémon sprites; no licence on card)"},
    {"project": "Spreadsheets Are All You Need (GPT-2 in Excel)", "source": "https://github.com/ianand/spreadsheets-are-all-you-need",
     "medium": "Microsoft Excel", "task": "GPT-2 Small text generation",
     "runs_inside_medium": "forward pass in standard formulas (no VBA)", "model_size": "124M params; v0.7.0 .xlsb 1,340,222,411 bytes",
     "steps": "1 forward pass per token", "correctness_shown": NS, "training_data": "pretrained GPT-2 weights"},
    {"project": "Freak in the Sheets", "source": "https://devpost.com/software/freak-in-the-sheets-7jl542",
     "medium": "Google Sheets", "task": "run C/Rust via an LLVM backend",
     "runs_inside_medium": "bytecode interpreter in formulas (iterative calculation)", "model_size": "n/a (not a neural network)",
     "steps": "VM steps", "correctness_shown": NS, "training_data": "n/a"},
]
ours = {"project": "SpriteSheet v32 (ours)", "source": "this repo", "medium": "Google Sheets (also HyperFormula, LibreOffice)",
        "task": "text → 32×32 pixel-art sprite", "runs_inside_medium": "full model: tokenizer, CLIP word table, attention encoder, seeded style sampler, conv decoder, argmax",
        "steps": "1 forward pass (single recalculation)", "training_data": "Noto/Twemoji/Fluent/OpenMoji emoji + PixelGPT 20K (licensed; see docs/03)"}
for f in ("systems.json", "parity_local.json", "parity_sheets.json"):
    p = os.path.join(M, f)
    if os.path.exists(p):
        ours[f.replace(".json", "")] = json.load(open(p))
rows.insert(0, ours)
os.makedirs(M, exist_ok=True)
json.dump(rows, open(os.path.join(M, "related_work.json"), "w"), indent=1, ensure_ascii=False)
print("wrote", len(rows), "rows")
