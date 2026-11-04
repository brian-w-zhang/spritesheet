// Evaluate cases.json with HyperFormula (GPLv3 licence key).
const { HyperFormula } = require('hyperformula');
const C = require('./cases.json');

function a1(ref) { const m = ref.match(/^([A-Z]+)(\d+)$/); let c = 0; for (const ch of m[1]) c = c * 26 + ch.charCodeAt(0) - 64; return { col: c - 1, row: +m[2] - 1 }; }
function grid(cells) { const g = []; for (const [k, v] of Object.entries(cells)) { const { row, col } = a1(k); (g[row] ||= [])[col] = v; } for (let r = 0; r < g.length; r++) { g[r] ||= []; for (let c = 0; c < g[r].length; c++) if (g[r][c] === undefined) g[r][c] = null; } return g; }

const F = { ...C.data };
for (const p of C.meta.probes) F[p.plain] = p.formula;
F[C.meta.spill.cell] = C.meta.spill.formula;
const hf = HyperFormula.buildFromSheets({ F: grid(F), M: grid(C.mlp_cells) }, { licenseKey: 'gpl-v3', smartRounding: false, useArrayArithmetic: true });
const sF = hf.getSheetId('F'), sM = hf.getSheetId('M');
const val = (s, ref) => { const { row, col } = a1(ref); const v = hf.getCellValue({ sheet: s, row, col }); return (v && typeof v === 'object') ? `ERR:${v.type}:${v.message || ''}` : v; };
const out = { engine: 'hyperformula ' + HyperFormula.version, probes: {}, mlp: {} };
for (const p of C.meta.probes) out.probes[p.name] = val(sF, p.plain);
out.probes[C.meta.spill.name] = val(sF, C.meta.spill.read);
out.mlp.y = C.meta.mlp.ycells.map(r => val(sM, r));
out.mlp.y_mmult = C.meta.mlp.ycells_mmult.map(r => val(sM, r));
console.log(JSON.stringify(out));
