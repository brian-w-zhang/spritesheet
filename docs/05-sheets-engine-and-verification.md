# 05 · Formula engines, safe function subset, verification stack

Tested hands-on 2026-11-03 (prototype scripts in `tools/verify-prototypes/`; they were written in a scratch dir, so fix the paths when reusing them). Real Google Sheets was **not** tested in the setup pass; that is gate A4 (`scripts/sheets_probe.py` is the pre-flight).

## Verdict
- **Primary oracle: HyperFormula 3.5.0** (npm, `GPL-3.0-only`, `licenseKey: 'gpl-v3'`), run in-process in node.
  **Required config:** `{licenseKey: 'gpl-v3', smartRounding: false, useArrayArithmetic: true}`. With the default `smartRounding`, `EXP(1)` came back as `2.7182818285` and the MLP error was 3.4e-12.
- **Secondary oracle: LibreOffice 26.8 headless** on the real `.xlsx` artifact. This also tests the file format.
- **Third (small cases only): Python `formulas` 1.3.4** (slow: 7 s per 10k cells). pycel lacks MMULT, so skip it.

## Measured (8→16→4 ReLU MLP, weights in cells, vs numpy float64)

| Engine | max abs err |
| --- | --- |
| HyperFormula (smartRounding off) | 5.6e-17 |
| LibreOffice | 3.3e-16 (its 15-significant-digit output is the floor) |
| formulas | 1.9e-15 |

Scale, with a grid of `=SUMPRODUCT(W!$A$r:$BL$r, act!$A$a:$BL$a)` (64-term dot products):

| Formula cells | HyperFormula build+eval | HF memory | HF recompute after 1 input edit | LibreOffice convert |
| --- | --- | --- | --- | --- |
| 102k | 1.65 s | 571 MB | 5 ms | 1.76 s |
| 1.02M | 17.4 s | 2.1 GB (`node --max-old-space-size=12000`) | 5 ms | 10.4 s (+2.1 s read) |

Errors at scale were 1e-14 (summation order on values around 10). **SpriteSheet's expected size (~300k–600k formula cells) is comfortably in range locally.**

## Function support matrix (✓ = correct value)

| Function | HyperFormula | LibreOffice | formulas | Google Sheets (docs) |
| --- | --- | --- | --- | --- |
| SUMPRODUCT (incl. `(cond)*range`) | ✓ | ✓ | ✓ | ✓ |
| MMULT inside INDEX | ✓ | ✓ | ✓ | ✓ |
| EXP, TANH, MAX, MAX(0,x), IF | ✓ | ✓ | ✓ | ✓ |
| INDEX, MATCH(…,0), LOWER | ✓ | ✓ | ✓ | ✓ |
| XLOOKUP | ✓ | needs `_xlfn.` | ✓ | ✓ |
| LET | ✗ #NAME | needs `_xlfn.`/`_xlpm.` | ✓ | ✓ |
| LAMBDA / MAP / MAKEARRAY / BYROW | ✗ | ✗ | partly | ✓ |
| SPLIT (Sheets semantics) | ✗ (different meaning) | ✗ | ✗ | ✓ |
| REGEXMATCH / REGEXREPLACE | ✗ | ✗ (LO has REGEX) | ✗ | ✓ |
| ARRAYFORMULA / bare spill | ✓ | ✗ | ✗ | ✓ |
| `SUM(A*B)` not array-entered | ✓ (with useArrayArithmetic) | ✗ #VALUE | ✓ | ✓ (needs ARRAYFORMULA) |

### The allowed subset (Pixcell may only emit these)
`SUMPRODUCT`, `MMULT` (only inside `INDEX`, or avoid it), `INDEX`, `MATCH(…,0)`, `MAX`, `MIN`, `SUM` over ranges, `EXP`, `TANH`, `IF`, `MOD`, `INT`, `+ − * /`, and, for the tokenizer only, `LOWER`, `SUBSTITUTE`, `TRIM`, `MID`, `REPT`, `LEN`, `IFERROR`.
One **scalar formula per cell.** No multi-cell arrays, spills, LET/LAMBDA, or volatile functions (`RAND`, `NOW`, `INDIRECT`, `OFFSET`).
Sheets-only functions (`SPARKLINE`, optionally `REGEXREPLACE`) are allowed **only in the render/UI layer**, with their own real-Sheets check, and the local oracles skip those cells.

### Portable tokenizer (no SPLIT/REGEX)
1. `clean = LOWER(prompt)`, then a fixed chain of `SUBSTITUTE(…, ch, " ")` for each punctuation char in the allowed set. This must include `* ? ~`, because `MATCH(…,0)` treats them as wildcards in Sheets/Excel.
2. Word *i* (1-based) = `TRIM(MID(SUBSTITUTE(TRIM(clean)," ",REPT(" ",100)), (i-1)*100+1, 100))`. This is the classic fixed-width split, using only portable functions.
3. `id_i = IF(word_i="", PAD, IFERROR(MATCH(word_i, Vocab!$A$1:$A$V, 0), UNK))`
4. **Spike this early** (A1): run all engines on adversarial prompts. Watch `TRIM` (it collapses internal spaces in all engines?) and `MATCH` case-insensitivity. The Python tokenizer must replicate whatever the formulas do, not the other way round.

## Numerical rules
- **Round every weight to 15 significant digits** (`float('%.15g' % w)`) *before* computing the numpy reference. openpyxl writes `%.16g`, which changed 43% of random float64 weights (e.g. `0.30000000000000004` → `0.3`). Pre-rounded values round-trip exactly. Use the same rounded values for the Sheets API upload.
- Softmax is always `EXP(x − MAX(range))`. `EXP(800)` gives `#NUM`. Any `EXP` input must be bounded by construction.
- Tolerances: logits `max|engine − numpy| ≤ 1e-12·max(1,|y|)` locally (achieved: 1e-14). **Argmax outputs must be identical.** Report the minimum top-1/top-2 margin; if a margin is < 1e-9, flag that prompt.
- No error objects anywhere: the check scans every formula cell for `#NAME`, `#VALUE`, `#REF`, `#NUM`, `#DIV/0`.

## LibreOffice gotcha: it does not recalculate by default
openpyxl files claim to come from "Microsoft Excel", so LibreOffice trusts the cached values. Measured: the default and "never" settings returned planted fake values (999) in every cell. You **must**:
1. use an isolated profile (`-env:UserInstallation=file://<profile>`) whose `registrymodifications.xcu` sets `/org.openoffice.Office.Calc/Formula/Load` `OOXMLRecalcMode` and `ODFRecalcMode` = `0` (always);
2. plant stale values (`make_stale.py` writes `<v>999</v>` into formula cells) and assert none survive.

Convert with: `soffice -env:UserInstallation=file://$PROFILE --headless --norestore --convert-to "xlsx:Calc MS Excel 2007 XML" --outdir OUT in.xlsx`, then read back with `openpyxl.load_workbook(..., data_only=True)`. UNO scripting with LibreOffice's bundled Python is SIGKILLed on this Mac (exit 137), so don't use it.

## Writing the workbook
- **Local artifact:** openpyxl, plain formula strings, one per cell. (`ArrayFormula` works in LibreOffice, but avoid it.) openpyxl writes no cached values.
- **Google Sheets:** prefer the Sheets API directly over uploading xlsx. Create the spreadsheet, add sheets, write weights as `values` (RAW, numbers) and formulas as `USER_ENTERED` strings in ≤2 MB batches at ≤1 req/s. For repeated formula patterns, use `batchUpdate` `repeatCell`/`copyPaste` with relative references, so you send one formula per block instead of thousands.
  - *Fallback:* Drive upload with conversion of the `.xlsx`. Unverified: how Sheets treats `t="array"` and `_xlfn.` prefixes, and whether a very large xlsx converts at all.
- **Unverified until A4:** Sheets' recalc time at our size, whether its `SUMPRODUCT` summation order stays within 1e-12, and whether 15-digit numbers parse exactly. Run the cheap check first: one small compiled layer pushed and compared against numpy.

## Rendering (Sheets only; verify in R3)
`=SPARKLINE(1,{"charttype","bar";"max",1;"color1",<hex cell>})` per display cell. Fallback: conditional-format rules per class. Pixel-art rendering through conditional formatting has prior art (Doom.xlsx), so it is not a novelty claim.
