// HyperFormula oracle for Pixcell workbooks (GPL-3.0 licence key; config per docs/05).
// Usage: node --max-old-space-size=12000 hf_eval.js in.json out.json
// in.json:  {"sheets": {name: [[cell|null,...],...]}, "runs": [{"set": [[sheet,row0,col0,value],...], "read": [sheet,...]}]}
//           Strings starting with "=" are formulas; text is sent with a leading apostrophe by the Python side.
// out.json: {"version", "build_s", "runs": [{"eval_s", "values": {sheet: grid}}]}; errors become "#ERR:<type>".
const fs = require('fs');
const path = require('path');
const { HyperFormula } = require(path.join(__dirname, 'node_modules', 'hyperformula'));

const inp = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const t0 = Date.now();
const hf = HyperFormula.buildFromSheets(inp.sheets, {
  licenseKey: 'gpl-v3', smartRounding: false, useArrayArithmetic: true, maxColumns: 18278, maxRows: 100000,
});
const build_s = (Date.now() - t0) / 1000;

function clean(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'object') return '#ERR:' + (v.type || 'UNKNOWN');
  return v;
}

const out = { version: HyperFormula.version, build_s, runs: [] };
for (const run of inp.runs) {
  const t1 = Date.now();
  if (run.set && run.set.length) {
    hf.batch(() => {
      for (const [sheet, row, col, value] of run.set) {
        hf.setCellContents({ sheet: hf.getSheetId(sheet), row, col }, [[value]]);
      }
    });
  }
  const eval_s = (Date.now() - t1) / 1000;
  const values = {};
  for (const name of run.read) {
    values[name] = hf.getSheetValues(hf.getSheetId(name)).map(r => r.map(clean));
  }
  out.runs.push({ eval_s, values });
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
