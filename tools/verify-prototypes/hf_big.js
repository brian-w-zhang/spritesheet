// node --max-old-space-size=8192 hf_big.js  -> timings + max abs err vs numpy for big.json
const { HyperFormula } = require('hyperformula');
const t0 = performance.now();
const B = JSON.parse(require('fs').readFileSync('big.json'));
const t1 = performance.now();
const hf = HyperFormula.buildFromSheets({ act: B.act, W: B.W, out: B.out }, { licenseKey: 'gpl-v3', smartRounding: false });
const t2 = performance.now();
const vals = hf.getSheetValues(hf.getSheetId('out'));
let err = 0, bad = 0;
for (let i = 0; i < vals.length; i++) for (let j = 0; j < vals[i].length; j++) {
  const v = vals[i][j]; if (typeof v !== 'number') { bad++; continue; }
  err = Math.max(err, Math.abs(v - B.ref[i][j]));
}
const t3 = performance.now();
// incremental: change one activation -> R dependent cells recompute
hf.setCellContents({ sheet: hf.getSheetId('act'), row: 0, col: 0 }, [[0.5]]);
const t4 = performance.now();
const mem = process.memoryUsage().rss / 2 ** 20;
console.log(JSON.stringify({ cells: B.out.length * B.out[0].length, parseJSON_s: (t1 - t0) / 1e3, build_and_eval_s: (t2 - t1) / 1e3,
  read_s: (t3 - t2) / 1e3, incremental_s: (t4 - t3) / 1e3, maxAbsErr: err, nonNumeric: bad, rssMB: Math.round(mem) }));
