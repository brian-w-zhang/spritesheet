# Draft for the course `abstract.md` (update in place in GitLab; add the private YouTube link)

**SpriteSheet: a text-to-pixel-art neural network that runs as native Google Sheets formulas.** Team Whiplash · SE 390 MP2 · Advanced Technology

Type a prompt into a Google Sheet, and a sprite is drawn by roughly 220,000 ordinary formulas. There is no Apps Script, add-on or external call. We trained a small text-to-image network: word vectors, a tiny attention layer, and a convolutional decoder that outputs 32×32 palette-indexed pixel art. Our compiler, Pixcell, lowers it into a workbook. A float64 numpy reference is the specification. The live sheet reproduces it pixel for pixel: 20 test prompts, 0 mismatches, relative error about 1e-13. The same check also runs locally in HyperFormula and LibreOffice.

The deployed model (736,224 parameters, 1.77M cells) handles prompts outside its training data, with CLIP R@5 0.41 on novel prompts. An earlier one-pass version scored a higher CLIP score than SpriteX32, an 11.4M-parameter, 50-step diffusion model, on held-out prompts (0.267 vs 0.249).

Our iteration attacked mode averaging, the blur a one-shot per-pixel model produces:
- **Masked discrete diffusion:** runs in Sheets at 836k formulas. It was no better by eye, and slower.
- **Learned variants:** on PixelGPT 216K at 24×24. Colour-adjective accuracy rose from 0.66 to 0.90, but structure stayed flat.
- **A VQ code grid with an autoregressive prior:** more varied drawings, with colour accuracy 0.97. Still blurry.

Every number is produced by a script in the repo.

Video: <private YouTube link>
Repo: https://github.com/brian-w-zhang/spritesheet
