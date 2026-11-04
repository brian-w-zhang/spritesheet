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

## What worked / what didn't
*(fill in after the overnight run: be specific, including failures)*
