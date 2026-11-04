# SpriteSheet: agent instructions

You are working on **SpriteSheet** (public repo, so mind what you commit): a text-to-pixel-art neural network that runs entirely as native Google Sheets formulas, compiled by our compiler **Pixcell**. It is an SE 390 (UWaterloo) MP2 Advanced Technology project for Team Whiplash.

## Read first, in order
1. `GOAL.md`: the mission, definition of done, phases.
2. `docs/00-README-START-HERE.md`: the doc index.
3. `docs/04-architecture.md`, `docs/05-sheets-engine-and-verification.md`, `docs/06-evaluation-and-benchmarks.md`, `docs/07-risks-and-fallbacks.md`.
4. `PROGRESS.md` (if it exists): where the last session stopped.

## Hard rules
- **The spec is `src/spritesheet/reference.py` (numpy float64).** The spreadsheet must match it; PyTorch is only used for training.
- **The core math is pure formulas.** No Apps Script, add-ons, `IMPORT*`, `GOOGLEFINANCE`, `IMAGE(url)` or external calls in the workbook, ever.
- **Never fabricate results.** A check that did not run is reported **NOT RUN**, with the reason. Every number in README, PROGRESS or slides must come from `artifacts/metrics/*.json` produced by a script.
- **Docs win.** If you change the design, add an entry to `docs/decisions.md` and update the affected doc in the same commit.
- **Commits:** small and frequent, under the repo's configured git identity. **No `Co-Authored-By`, `Assisted-by`, "Generated with" or any AI attribution in commit messages** (owner's rule). Never change git `user.name`/`user.email`. Push to `origin` (public GitHub repo) after every passing gate. **Never commit `data/raw/`, `data/models/`, `docs/course/`, PixelGPT images, or anything containing credentials**; check `git status` before each commit.
- **Google Sheets:** use only the spreadsheet ids in `~/.config/spritesheet/config.json`, via the service-account key in the same folder. The service account can't create spreadsheets. Never touch other Drive files, and never change sharing. Respect API limits: ≤60 write requests/min, ≤2 MB per request.
- **Never print, log or commit access tokens or credentials.**
- Use `uv` for Python (`uv run ...`); keep node deps local to `tools/`.
- Update `PROGRESS.md` after every milestone (what's done, what's verified, what's next, any blockers). A fresh agent must be able to resume from it.
