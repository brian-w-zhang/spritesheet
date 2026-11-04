# TODO: SpriteSheet (SE 390 MP2)

Deadline: **slides.pdf + private YouTube link due end of Wed Nov 4** (GitLab `mini-project2/TEAMNAME/`, sign-up sheet). Review is Thu Nov 5.

## Tonight, before starting `/goal` (Brian, about 10 min)
- [ ] Read `GOAL.md` once; edit anything you disagree with.
- [ ] Google Sheets access for the agent: run `python3 -I scripts/sheets_probe.py`. If it fails, follow the printed fix (re-login with Sheets+Drive scopes, set the quota project, enable the Sheets/Drive APIs), then rerun until PASS. Without a PASS the agent can only verify locally, and gate A4 will be reported NOT RUN.
- [ ] `uv` works offline? (The agent needs PyPI for numpy/torch/openpyxl the first time.)
- [ ] Keep the laptop plugged in; disable sleep (`caffeinate -dimsu &`).
- [ ] Start: paste the `/goal` line at the top of `GOAL.md` (repo is already created and pushed).
- [ ] Run Claude Code for the overnight session in **auto mode** (or pre-approve git/uv/node/soffice/gh), so it doesn't stall on a permission prompt.

## Done in the setup pass (Nov 3)
- [x] Repo skeleton, docs folder, course guide copied in
- [x] Research: prior art (incl. SpriteX32), datasets (emoji sets, PixelGPT, icon sets, GameTileNet; rejected Kaggle/tkarr/bghira/Nouns), local formula engines → `docs/02`, `docs/03`, `docs/05`
- [x] Data + CLIP downloaded; uv env + HyperFormula installed; pixelation previews in `artifacts/figures/`
- [x] Architecture, evaluation plan, risks → `docs/04`, `docs/06`, `docs/07`
- [x] Overnight plan / goal prompt → `GOAL.md`
- [x] Sheets pre-flight probe → `scripts/sheets_probe.py`

## Overnight (agent, via GOAL.md)
- [ ] Data pipeline (normalise, synth variations, splits) + CLIP word table
- [ ] Model training + numpy float64 reference
- [ ] Pixcell compiler → `.xlsx` + Google Sheets
- [ ] Gates A1–A5, benchmarks B–D → `artifacts/metrics/`
- [ ] Figures for slides → `artifacts/figures/`
- [ ] `PROGRESS.md` morning report (what passed, what didn't, what's NOT RUN)

## Wed Nov 4 morning: review the agent's work (Brian + team)
- [ ] Read `PROGRESS.md`; rerun `scripts/verify_all.sh` yourself and confirm the numbers.
- [ ] Open the real Google Sheet, type 3 prompts, and screen-record (the main demo footage).
- [ ] Each teammate reads one part of the code well enough to answer questions at the review (§11 warns about code "nobody on the team fully understands"). Suggested split: tokenizer + text encoder · decoder + training · Pixcell compiler · verification + benchmarks.

## Wed Nov 4: deliverables
- [ ] **Slides** (5–8, tasteful, in the style of the askjev MP1 deck: HTML → PDF via headless Chrome, same look). Required content (guide §7): problem + related work/competition · results · how we organized (board, cadence) · use of AI (what worked / what didn't / how we review generated code / what we didn't delegate) · lessons + next step (MP3 = Waterloo Goose Simulator?). Every number traced to `artifacts/`.
- [ ] **Presentation script**: ~4:40 at ~140 wpm, split across all four members, with handoffs marked.
- [ ] **Video**: record per speaker, then edit; private YouTube; link in the sign-up sheet **and** `abstract.md`.
- [ ] **Rewrite `abstract.md` in place** for SpriteSheet (guide §8: an abstract that no longer describes the work is penalized). Template items: opportunity/problem · objective · design approach · benefits over existing alternatives · expected results/evidence. Plus title, category (Advanced Technology), team name, members + WatIAM IDs, video link, and an AID statement.
- [ ] `AI-USE.md` final pass (tools, what for, review process).
- [ ] GitLab: branch → `mini-project2/TEAMNAME/{abstract.md, slides.pdf}` → merge to main before midnight.
- [ ] Sign-up sheet: team name (exactly matching the folder), project name, WatIAM IDs one per line.

## Process evidence (Notion)
- [ ] New Notion page "SE 390 · SpriteSheet", modelled on "SE 390 · askjev": callout with review time, Start here, Links, Key dates, How we work, Team tabs.
- [ ] Kanban (Tasks DB: Backlog → To Do → In Progress → Blocked → Done; Owner; Human/Agent/Both; due date), populated with the **real** cards from this TODO and the agent's milestones.
- [ ] Real split of the remaining work among the four of us (slides, script, recording, code walkthroughs, expert outreach, GitLab/sign-up), recorded on the cards as people take them.
- [ ] Slide on "how we used AI" built from the real board + `PROGRESS.md` + `AI-USE.md`.

> Note on "simulated" progress: I've left out back-dated cards and randomly assigned contributions. The guide says the board should be "a log of what has been accomplished and by whom", and the instructor grades Process on it. Fabricating it is the kind of thing that turns into an academic-integrity problem. The honest version still looks good: one person ran the agent overnight, and the others own slides, script, video, code walkthroughs and outreach, all with real cards.

## External contact (Brian/team, not the agent)
- [ ] Send the demo clip to 2–3 people for a one-paragraph comment (ML TA/prof, a Nouns community member, a spreadsheet/GPT-2-in-Excel person). Log in `docs/feedback.md` with dates and verbatim replies (with permission).

## Open questions for Brian
- [ ] MP2 GitLab folder name / team name (MP1 was `whiplash`)?
- [ ] Is MP3 still Waterloo Goose Simulator (Notion MP2 page says so)? If yes, the last slide announces the pivot.
- [ ] The course recommends the `Assisted-by:` commit trailer, but your global rule bans Claude attribution in commits. Current choice: no trailers; disclose in `AI-USE.md` + slides. OK?
- [ ] Which electives to name on slides (ML, deep learning, algorithms confirmed; compilers? numerical methods?)

## Data / licences (later)
- [ ] `ATTRIBUTION.md`: Twemoji (CC-BY 4.0), OpenMoji (CC BY-SA 4.0), PixelGPT 20K (CC BY 4.0 per card), game-icons (CC BY 3.0, per author), GameTileNet tilesets used, Noto (Apache-2.0), Fluent (MIT), CLDR (Unicode License v3), CLIP (MIT).
- [ ] Optional: email unstonio about the PixelGPT 20K placeholder LICENSE.txt vs the CC BY 4.0 card.
- [ ] Optional: buy PixelGPT full (216k, $24) for v48/v64. It must stay local (no redistribution).
- [ ] Decide on OpenMoji (share-alike) before publishing weights.
