# Start here

**SpriteSheet:** type a prompt into a Google Sheet, and a 32×32 pixel-art character appears in the cells. It is computed by a neural network that exists only as spreadsheet formulas.
**Pixcell:** our compiler from a trained model to a workbook.

| Doc | What's in it |
| --- | --- |
| `../GOAL.md` | The overnight mission: definition of done, phases, gates, what to report |
| `../TODO.md` | Everything outside the overnight build: slides, script, video, abstract, Notion, outreach |
| `10-how-it-works.md` | **Plain-language explainer at 4 levels** (tweet → prof → teammates → builder), with real formulas |
| `01-course-requirements.md` | What SE 390 MP2 requires and how it's graded (summary of the course guide) |
| `02-prior-art.md` | Freak in the Sheets, GPT-2 in Excel, others; Google Sheets limits; novelty wording; sources |
| `03-dataset.md` | Data sources (emoji, PixelGPT, icons), licences, size normalisation, synthetic variations, splits |
| `04-architecture.md` | Model, compiler, workbook layout, code layout |
| `05-sheets-engine-and-verification.md` | Which formula engines to verify with, which functions are safe, how to push to Sheets |
| `06-evaluation-and-benchmarks.md` | Gates A1–A5, quality metrics, baselines, systems benchmarks, comparison table |
| `07-risks-and-fallbacks.md` | Risk register, spikes to run first, the fallback ladder |
| `09-experiment-log.md` | Iterations, numbers, and how we got to the final model (slide source) |
| `decisions.md` | Append-only decisions log |
| `course/` | (local only, gitignored) course guide, AskJev MP1 slides text |
