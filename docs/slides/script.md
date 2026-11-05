# MP2 narration (5 min, ~650 words) for `docs/slides/deck/slides.pdf`

Roughly 35–40 s per slide. Every number is on the slide and comes from the metrics file named in `docs/slides/README.md`. Who speaks which slide is for the team to decide; mark it here.

**1 · SpriteSheet (title)**
Type a prompt into a Google Sheet, and a neural network made of nothing but formulas draws the sprite. No scripts, no add-ons, no API calls: 493,532 ordinary formulas. On the right is a real cell from our sheet. Pixel L15 of this potion is an `INDEX` into a patch dictionary, picked by the model a few tabs upstream.

**2 · Problem**
Neural networks feel like magic, mostly because you can't see inside them. A spreadsheet can't hide anything. Our idea: put a whole text-to-image model inside the most familiar tool there is, so every multiply-add is a cell you can click. It's for students, teachers and spreadsheet users. Spreadsheet ML so far means tiny classifiers. Text-to-pixel-art models like SpriteX32 run on GPUs. As far as our search found, nothing else runs a text-to-image model as native spreadsheet formulas.

**3 · How it works**
We built a compiler, Pixcell. We train in PyTorch on 195,154 licensed PixelGPT sprites. A float64 numpy reference is the definition of correct. Pixcell turns every multiply-add into a cell formula, like this ReLU. We upload the formulas as inert text and switch them on tab by tab. Then three engines, HyperFormula, LibreOffice and Google Sheets itself, must match the reference pixel for pixel. The model draws 64 patches one at a time, and a single recalculation draws the whole sprite.

**4 · Results**
Our first models drew blobs. A one-pass network trained per pixel draws the average of every frog it has seen. PatchGPT fixed that by choosing from hard 3×3 patches, so blur is impossible by construction. Then we added classifier-free guidance, inside the sheet, as a second small chain. Novel-prompt CLIP R@5 went from 0.41 to 0.61. We also tried and dropped diffusion, learned variants, a VQ code grid, faster parallel drawing and truncated sampling. Each one was blobbier or blank.

**5 · Demo**
This is the live sheet. These eight sprites were read back from Google Sheets, and all eight match our reference pixel for pixel. A new prompt takes two to three minutes to settle. Guidance is a live cell: type zero and it switches off. Same prompt, new seed, and you get a different frog.

**6 · Process**
Humans decide, agents build, scripts check. We set the problem and the quality bar. Notion holds the course context and our board, and agents read it over MCP. Claude Code runs long sessions from a written goal and logs every experiment. GitHub holds 44 logged decisions and 17 evaluation rounds. Every card on the board says whether a human, an agent or both did it. *(Team: add the meeting cadence.)*

**7 · Use of AI**
An AI agent wrote the code; we kept it honest. What worked: a spec first, then pass/fail gates. They caught three spec bugs review missed. What didn't: trusting the loss, because lower loss still meant blobs, so we made the agent look at every image. It once reported 41 seconds per prompt; it had re-rendered the old prompt, and the real number is about two minutes. We vetoed its plan to bake guidance into the weights. We didn't delegate the project, the dataset purchase, the quality bar, or this narration.

**8 · Lessons and next**
Exactness was cheap; quality was hard. Look at the pictures, not the loss. And speed follows formula count. A cheaper guidance chain is what got guidance onto the sheet. Next time: under a minute per prompt, and better creatures. For MP3 we're pivoting to the Waterloo Goose Simulator, a 3D game of campus where you play the goose. Thank you.
