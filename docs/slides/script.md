# Video script draft (about 5 minutes)

A draft for the team to rewrite in their own voices. The course grades narration flow, and every member should speak: assign the `[Speaker]` slots yourselves. Numbers match `docs/slides/README.md`, which names each source file.

---

**1. Hook (0:00–0:35) · [Speaker A]**
*(Screen recording: type "a red frog" into the sheet; the sprite appears.)*
This is a Google Sheet. There's no script, no add-on and no API call behind it. Every pixel you see comes from about two hundred and twenty thousand ordinary formulas: SUMPRODUCTs, EXPs, MATCHes. Together they form a neural network that turns text into pixel art. We call the compiler that builds it **Pixcell**.

**2. How it works (0:35–1:30) · [Speaker B]**
*(pipeline.svg, then network_tabs.png)*
We train a small text-to-image network in PyTorch. A word-vector table and a tiny attention layer read the prompt. A convolutional decoder grows a 4×4 grid into a 32×32 sprite. A palette picks each pixel's colour.
Pixcell then lowers every layer into a sheet tab: matrix multiplies become SUMPRODUCTs over weight blocks, and attention becomes softmax formulas. The deployed model has 736 thousand trained parameters in 1.77 million cells, under nine percent of Google's cell limit.
The hard requirement was exactness. A float64 numpy reference is our spec. In real Google Sheets, twenty test prompts matched it pixel for pixel, with relative error around ten to the minus thirteen.

**3. v1 results (1:30–2:20) · [Speaker C]**
*(lineup_xl_novel.png, seeds.png)*
We trained on emoji in four art styles plus twenty thousand PixelGPT sprites. The seed cell picks a style. The model handles prompts that are in neither dataset, with a CLIP top-5 retrieval rate of 0.41.
The biggest lever was simply scale: training four times longer and 2.4 times wider raised held-out pixel accuracy from 0.625 to 0.676.
And against SpriteX32, a diffusion model with eleven million parameters running fifty steps on a GPU, our first one-pass spreadsheet model scored a higher CLIP score on held-out prompts: 0.267 versus 0.249.

**4. Iteration: the averaging problem (2:20–3:30) · [Speaker D]**
*(v3_lineup_r4_creatures.png)*
Look closely and v1's sprites are blurry. That's not a bug: a network trained pixel by pixel learns the *average* of every frog it has seen.
We tried three fixes.
- **Diffusion (v2):** drawing in four steps. It runs in Sheets, at 836 thousand formulas, but it wasn't better by eye and it was four times slower.
- **Learned variants (v3):** sixteen variants, each owning the sprites it draws best, with the seed choosing between them. Colour words got much better, 0.66 to 0.90, but the variants split by colour, not by shape.
- **A code grid:** a 6×6 grid of 256 learned codes, sampled one at a time by a small prior. That gives the first interiors and real seed variety, with diversity doubled. But it's still mushy.
Honestly: on our retrieval metric, every model ties within its confidence interval, so v1 stays the live demo.
*(Add the critic / guidance results if they land.)*

**5. How we worked (3:30–4:05) · [Speaker A]**
*(kanban screenshot)* TODO, team: meeting cadence, the board, who owned what.

**6. Use of AI (4:05–4:40) · [Speaker B]**
We used Claude Code heavily: for research, the overnight implementation, training and figures. The rule was *checks, not trust*. Every claim is a script that passes or fails, and the parity gates caught three real spec bugs that review missed. What didn't work: the agent initially trusted loss numbers, so we made it look at real Sheets output. We kept the project choice, novelty claims and this narration for ourselves.

**7. Lessons and next (4:40–5:00) · [Speaker C]**
Exactness turned out to be the easy part, once there was a spec and gates. Quality is the hard part. Next: putting the code-grid model into the sheet. TODO: the MP3 pivot.
