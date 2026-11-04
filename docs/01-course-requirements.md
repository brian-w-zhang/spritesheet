# 01 · Course requirements (SE 390 MP2)

Source of truth: the SE 390 course guide v1.1, available locally at `docs/course/` (gitignored: course material is not published, per guide §15).
Brian's Notion page "SE 390 MP2" also quotes guide v1.2 and the Capstone Handbook §14.2.5. If this summary and the guide disagree, **the guide wins**.

## Deadlines (America/Toronto)

| What | When |
| --- | --- |
| `slides.pdf` in GitLab `secapstone/se2028-390/mini-project2/TEAMNAME/` (merged to main by us) | **end of Wed Nov 4, 2026** |
| 5-min narrated video, **private YouTube**, link in the sign-up sheet **and** in `abstract.md` | end of Wed Nov 4 |
| `abstract.md` updated **in place** to describe what we actually did (guide §8) | before the review |
| MP2 review (video played, then Q&A), MS Teams | Thu Nov 5 |
| MP3 abstract | Mon Nov 9 |

Late submissions are not accepted.

## Weights

MP2 is 35% of the course: Process 10, Results 15, Presentation 5, Video 5. Grading is holistic (A+ 100, A 90, A− 85, B+ 80 …).

## What MP2 must show (guide §7 calibration, MP2 column)

- The question: *"Can we actually build this, and does anyone outside our team care?"*
- **Technical artifact:** "A working prototype or MVP of the core idea: the part that would fail if the project is not viable."
- **Problem definition:** "Explicit comparison against competition or related work."
- **A-range:** "Substantial visible progress on the hard part, not the easy part."
- **External contact:** "First real contact." For this project that means a few quick expert comments (see TODO.md). We cannot run user studies in time, and we should say so honestly.
- "Ambition is rewarded, but so is judgement. A team that attempts something hard and reports honestly on what did not work will generally do better than a team that does something safe and polishes it."

## Advanced Technology rubric (Handbook §14.2.5, symposium standard)

A = "a problem was solved convincingly" **and** excellence on ≥2 of: importance of the problem · technical difficulty and scope · novelty of the solution · applies knowledge from multiple upper-year courses. A+ also needs "positive mention from an industry professional".
"A possible way to demonstrate that the problem was solved convincingly is by comparing the solution to the alternatives."

How SpriteSheet maps onto this:
- **Difficulty and scope:** a neural network (text encoder + image decoder), plus a compiler that lowers it into spreadsheet formulas, checked against a float64 reference.
- **Novelty:** as far as our prior-art search found, no text-to-image model runs as native spreadsheet formulas (see `docs/02-prior-art.md` for what was and was not checked). The closest functional competitor is SpriteX32, a GPU model.
- **Multiple courses:** ML / deep learning (model design and training), algorithms and data structures (layout, scheduling, tokenizer), compiler ideas (IR, lowering, codegen), numerical computing (float agreement).

## Every presentation must cover (5–8 slides, 5 minutes)

1. The problem, the audience, related past capstones and work, and how we differ.
2. Results of the iteration.
3. How we organized: meeting cadence, task management, the kanban board.
4. **Use of AI:** tools, what worked, what did not, how generated code is reviewed, what we did not delegate (§11).
5. Lessons learned, and the next step. Include the MP3 pivot if there is one.

The video is graded "as for presentation, plus narration flow and audio quality". Every team member should take part.

## AI rules that affect this repo (§11)

- Keep `AI-USE.md` at the repo root: which tools we use, for what, and how generated code is reviewed.
- Disclose documents with an AID Framework statement (https://subjectguides.uwaterloo.ca/AIDframework).
- The course suggests the `Assisted-by:` commit trailer. **Brian's global instructions forbid putting any Claude attribution in commits**, so this repo does not use trailers. AI use is disclosed in `AI-USE.md` and on the slides instead. Brian decides whether to change this.
- "You are accountable for everything you submit." No fabricated citations or numbers. Every number on a slide must trace back to a file in `artifacts/`.
- The instructor will ask whether the team understands the code: "a team that ships large volumes of code nobody on the team fully understands is accumulating risk".

## Team

Team Whiplash (MP1 folder `whiplash`): Brian Zhang, Sabrina Wang, Adrian Tang, Mayank Upadhyay. (WatIAM IDs are kept out of this public repo.) The MP2 folder name is unconfirmed (see TODO.md).
