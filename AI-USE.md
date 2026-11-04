# AI use in SpriteSheet

Required by SE 390 guide §11. Update this when practice changes.

## Tools
- **Claude Code** (Anthropic; model Claude Opus 5.5), with research subagents. Used for prior-art and dataset research, the architecture docs, the overnight implementation run (`GOAL.md`), tests, benchmarks and figure generation.
- No generative image models are used. Every sprite in this project is either Nouns CC0 art rendered by our deterministic compositor, or output from *our own* trained model.

## How generated code is reviewed before it lands
- **Docs first.** `docs/` and `GOAL.md` are written and reviewed by us before any code. When code and docs conflict, the docs win, and changes go through `docs/decisions.md`.
- **Checks, not trust.** Every claim is a script with pass/fail output: tokenizer parity, op parity, end-to-end parity of the spreadsheet against a float64 numpy reference, and a real-Google-Sheets check. Numbers on slides come from `artifacts/metrics/*.json`, and `scripts/check_claims.py` verifies them.
- **Human review the next morning.** A team member reruns `scripts/verify_all.sh`, reads the diff for their assigned component, and signs off in `PROGRESS.md`.
- Commits are made under team members' own identities.

## What we did not delegate
- Choosing the project and judging whether it is worth doing.
- Expert outreach and feedback (asked and recorded by us).
- The final wording of claims about novelty and comparisons. These are checked against sources we opened ourselves.
- The presentation narration.

## What worked / what didn't (overnight run, Nov 3–4 2026)

**Worked**
- **Docs-first, spec-first.** `reference.py` (numpy float64) was written before the compiler, and `docs/08-interfaces.md` before the data code. That let two coding agents build the data pipeline and the Pixcell compiler in parallel without stepping on each other; they met at the interface contract.
- **Checks caught real bugs that review alone would have missed.** The compiler agent's parity gates found three spec bugs: (1) the fixed-width word split broke when the first words summed to ≥100 characters (D-30); (2) a single LCG step mapped every small seed to style 1, because u ≈ seed·2.2e-5 (D-31); (3) HyperFormula reads `\` in a string literal as an escape, which broke every cell (D-32). Each one is now covered by a test.
- **De-risking with spikes.** A random-weight model was pushed to real Google Sheets *before* training. That fixed the model-size budget (D-21) and showed that Sheets matches numpy exactly.
- **Research with sources.** Prior-art and dataset agents had to list the URLs they actually opened. That is how we found SpriteX32's training data was unlicensed Pokémon sprites, and the Kaggle "pixel art" set was 16×16 JPEGs.

**Didn't work / cost us time**
- **Google auth.** `gcloud`'s own OAuth client is blocked for Sheets/Drive scopes ("This app is blocked"), so we switched to a service account plus one sheet shared by a human. A service account cannot create spreadsheets (no Drive quota).
- **Agents left junk in the repo:** a LibreOffice profile and scratch xlsx files, removed in a follow-up commit. Lesson: put `.gitignore` entries in place before letting agents run tests.
- **First model sizes were too small** (blurry icons). Capacity is limited by what the sheet can recalculate, so we moved channels to the cheap low-resolution layers.
- **The safety classifier correctly blocked** publishing a work-account project name to the public repo, and repointing the git remote. Both needed a human decision.

**Not delegated:** project choice, dataset licence decisions (e.g. not using Pokémon or the paid PixelGPT set in the public repo), the final wording of novelty claims, expert outreach, narration.
