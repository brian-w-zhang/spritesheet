"use client";
// The generation bank: every prompt x 8 seeds x 3 models, with the visual-review verdicts.
// Data: public/data/meta.json + sprites_<model>.gz (written by scripts/export_bank_site.py).
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

const AXES = [
  ["", "Any"], ["plain", "None (base nouns)"], ["seen", "Seen with this noun in training"],
  ["common", "Common caption adjective"], ["colour", "Colour"], ["material", "Material"],
  ["state", "State"], ["size", "Size"], ["pose", "Pose"],
];
const AXIS_NAME = { seen: "seen in training", common: "common adjective" };
const AXO = { plain: 0, seen: 1, common: 2, colour: 3, material: 4, state: 5, size: 6, pose: 7 };
const VRANK = { good: 2, ok: 1, blob: 0 };
const axisOf = (p) => (p.axis === "base" || p.axis === "demo" ? "plain" : p.axis);
const catName = (c) => c.replace(/^v_/, "variant · ").replace(/^\d+_/, "").replaceAll("_", " ");

async function inflate(url) {
  const r = await fetch(url);
  const s = r.body.pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(s).arrayBuffer());
}
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

const Sprite = memo(function Sprite({ m, j, cls, title, onToggle }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current, n = m.n;
    const ctx = c.getContext("2d"), im = ctx.createImageData(n, n), off = j * n * n;
    for (let i = 0; i < n * n; i++) { const [r, g, b] = m.rgb[m.px[off + i]]; im.data[i * 4] = r; im.data[i * 4 + 1] = g; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = 255; }
    ctx.putImageData(im, 0, 0);
  }, [m, j]);
  return (
    <canvas ref={ref} width={m.n} height={m.n} className={`sp ${cls}`} title={title} tabIndex={0} role="button"
      aria-label={title} onClick={onToggle}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onToggle(); } }} />
  );
});

export default function Bank() {
  const [meta, setMeta] = useState(null);
  const [models, setModels] = useState(null);
  const [status, setStatus] = useState("Loading prompts…");
  const [view, setView] = useState("rows");
  const [mk, setMk] = useState("all");
  const [sort, setSort] = useState("verdict");
  const [cat, setCat] = useState("");
  const [ax, setAx] = useState("");
  const [q, setQ] = useState("");
  const [stars, setStars] = useState(() => new Set());
  const [shown, setShown] = useState(0);
  const sentinel = useRef(null);

  useEffect(() => {
    (async () => {
      const m = await (await fetch("data/meta.json")).json();
      setMeta(m);
      const out = {};
      for (const k of Object.keys(m.models)) {
        setStatus(`Loading ${m.models[k].label} sprites…`);
        out[k] = { ...m.models[k], px: await inflate(`data/sprites_${k}.gz`), rgb: m.models[k].palette.map(hex) };
      }
      setModels(out);
    })().catch((e) => setStatus(`Could not load the bank: ${e.message}`));
    try { setStars(new Set(JSON.parse(localStorage.getItem("bank-stars") || "[]"))); } catch {}
    const u = new URLSearchParams(location.search);
    if (u.get("q")) setQ(u.get("q"));
    if (u.get("view")) setView(u.get("view"));
  }, []);

  const toggle = useCallback((id) => setStars((s) => {
    const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id);
    localStorage.setItem("bank-stars", JSON.stringify([...n])); return n;
  }), []);

  const S = meta ? meta.seeds.length : 8;
  const keys = models ? (mk === "all" ? Object.keys(models) : [mk]) : [];
  const rv = useCallback((k, i) => meta?.review?.[k]?.[i] || null, [meta]);
  const isPick = useCallback((k, j) => { const r = rv(k, Math.floor(j / S)); return !!r && (r.good_seeds || []).includes(meta.seeds[j % S]); }, [rv, S, meta]);
  const isShort = useCallback((k, j) => { const r = rv(k, Math.floor(j / S)); return !!r && !!r.shortlist && r.best_seed === meta.seeds[j % S]; }, [rv, S, meta]);
  const val = useCallback((k, j) => {
    if (sort === "verdict") {
      const r = rv(k, Math.floor(j / S)), v = r ? VRANK[r.verdict] ?? 0 : -1;
      return v * 10 + (r?.shortlist ? 5 : 0) + (isPick(k, j) ? 2 : 0) + (r?.best_seed === meta.seeds[j % S] ? 1 : 0) + models[k].score[j] / 100;
    }
    return sort === "worst" ? -models[k].score[j] : models[k].score[j];
  }, [sort, rv, isPick, S, meta, models]);
  const bestOf = useCallback((k, i) => { let b = i * S; for (let j = i * S; j < i * S + S; j++) if (val(k, j) > val(k, b)) b = j; return b; }, [val, S]);

  const items = useMemo(() => {
    if (!meta || !models) return [];
    const ql = q.trim().toLowerCase();
    const ok = (i) => { const p = meta.prompts[i];
      return (!cat || p.category === cat) && (!ax || axisOf(p) === ax) && (!ql || p.prompt.includes(ql) || p.base === ql); };
    const P = meta.prompts.length;
    if (view === "rows") {
      const it = [...Array(P).keys()].filter(ok);
      const key = (i) => { const p = meta.prompts[i];
        if (sort === "group") return `${p.base || p.prompt}|${AXO[axisOf(p)]}|${p.prompt}`;
        if (sort === "prompt") return p.prompt;
        if (sort === "n_train") return -(p.n_train || 0);
        return -Math.max(...keys.map((k) => val(k, bestOf(k, i)))); };
      const kk = new Map(it.map((i) => [i, key(i)]));
      return it.sort((a, b) => (kk.get(a) < kk.get(b) ? -1 : kk.get(a) > kk.get(b) ? 1 : 0));
    }
    const it = [];
    for (const k of keys) for (let j = 0; j < P * S; j++) {
      if (!ok(Math.floor(j / S))) continue;
      if (view === "stars" && !stars.has(`${k}|${j}`)) continue;
      if (view === "picks" && !isPick(k, j)) continue;
      if (view === "short" && !isShort(k, j)) continue;
      it.push([k, j]);
    }
    const key = ([k, j]) => { const p = meta.prompts[Math.floor(j / S)];
      if (sort === "prompt" || sort === "group") return p.prompt;
      if (sort === "n_train") return -(p.n_train || 0);
      return -val(k, j); };
    const kk = new Map(it.map((x) => [x, key(x)]));
    return it.sort((a, b) => (kk.get(a) < kk.get(b) ? -1 : kk.get(a) > kk.get(b) ? 1 : 0));
    // stars only matter for the starred view
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, models, view, mk, sort, cat, ax, q, view === "stars" ? stars : null]);

  const chunk = view === "rows" ? 25 : 240;
  useEffect(() => { setShown(chunk); window.scrollTo({ top: 0 }); }, [items, chunk]);
  useEffect(() => {
    const el = sentinel.current; if (!el) return;
    const ob = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) setShown((n) => Math.min(items.length, n + chunk)); }, { rootMargin: "2000px" });
    ob.observe(el); return () => ob.disconnect();
  }, [items, chunk]);

  const cats = useMemo(() => (meta ? [...new Set(meta.prompts.map((p) => p.category))].sort() : []), [meta]);
  const exportStars = () => {
    const out = [...stars].map((id) => { const [k, j] = id.split("|"); const jj = +j;
      return { model: k, prompt: meta.prompts[Math.floor(jj / S)].prompt, seed: meta.seeds[jj % S] }; });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
    a.download = "spritesheet_favourites.json"; a.click();
  };

  const tip = (k, j) => {
    const i = Math.floor(j / S), r = rv(k, i), p = meta.prompts[i], seed = meta.seeds[j % S];
    return `${models[k].label}: "${p.prompt}", seed ${seed}` + (r ? ` · reviewer: ${r.verdict}${r.best_seed === seed ? " (best seed)" : ""}${r.note ? ` · ${r.note}` : ""}` : "");
  };
  const cls = (k, j, best) => [best ? "best" : "", isPick(k, j) ? "pick" : "", stars.has(`${k}|${j}`) ? "star" : ""].join(" ");
  const nP = meta?.prompts.length ?? 0;

  return (
    <>
      <div className="bar"><span className="ref">Bank!A1</span><span className="fx">fx</span>
        <span className="f">=GENERATE(prompt, seed)   ← the same sprites the Google Sheets draw, computed headless</span></div>
      <section className="intro">
        <div>
          <h1>SpriteSheet generation bank</h1>
          <p>
            Every sprite here was drawn by one of our three text-to-image models that run as <strong>native Google Sheets
            formulas</strong>: the same model, the same seed, the same pixels as the sheet, computed in Python for speed.
            {nP ? <> <strong>{nP.toLocaleString()} prompts × 8 seeds × 3 models.</strong></> : null} Agents reviewed every row
            by eye; blue borders mark the seeds they rated good.
          </p>
        </div>
        <div className="legend">
          <span><i className="sw" style={{ borderColor: "var(--sel)" }} /> reviewer rated this seed good</span>
          <span><i className="sw" style={{ borderColor: "#c9c5cf" }} /> best seed for the row (by the current sort)</span>
          <span><i className="sw" style={{ borderColor: "var(--star)" }} /> your star (click a sprite; saved in this browser)</span>
          <span>v3 = patch transformer · v2 = masked diffusion · v1 = one-pass conv decoder</span>
        </div>
      </section>

      <div className="controls"><div className="in">
        <label>View <select value={view} onChange={(e) => setView(e.target.value)}>
          <option value="rows">One row per prompt</option><option value="short">Shortlist (slide-worthy)</option>
          <option value="picks">Every seed rated good</option><option value="grid">All sprites</option><option value="stars">Your stars</option></select></label>
        <label>Model <select value={mk} onChange={(e) => setMk(e.target.value)}>
          <option value="all">All models</option>
          {models && Object.keys(models).map((k) => <option key={k} value={k}>{models[k].label}</option>)}</select></label>
        <label>Sort <select value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="verdict">Reviewer verdict</option><option value="group">Noun, with its variants</option>
          <option value="score">CLIP score</option><option value="n_train">Most training data</option>
          <option value="prompt">Prompt A–Z</option><option value="worst">Worst CLIP score</option></select></label>
        <label>Category <select value={cat} onChange={(e) => setCat(e.target.value)}>
          <option value="">All</option>{cats.map((c) => <option key={c} value={c}>{catName(c)}</option>)}</select></label>
        <label>Adjective <select value={ax} onChange={(e) => setAx(e.target.value)}>
          {AXES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
        <input type="search" placeholder="Filter prompts, e.g. potion" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter prompts" />
        <button onClick={exportStars} disabled={!stars.size}>Export stars</button>
        <span className="count">{models ? `${items.length.toLocaleString()} ${view === "rows" ? "prompts" : "sprites"} · ${stars.size} starred` : ""}</span>
      </div></div>

      <main>
        {!models ? <div className="loading">{status}</div> : view === "rows" ? (
          items.slice(0, shown).map((i) => { const p = meta.prompts[i]; const a = axisOf(p); return (
            <div className="row" key={i}>
              <div>
                <h2>{p.prompt}</h2>
                <div className="meta">
                  {a === "plain" ? catName(p.category) : <>{AXIS_NAME[a] || a} variant of <button onClick={() => { setQ(p.base); setSort("group"); }}>{p.base}</button></>}
                  <br />{(p.n_train || 0).toLocaleString()} training captions
                </div>
              </div>
              <div className="sets">
                {keys.map((k) => { const b = bestOf(k, i), r = rv(k, i); return (
                  <div className="set" key={k}><span className="lab">{k}</span>
                    {[...Array(S).keys()].map((s) => { const j = i * S + s; return (
                      <Sprite key={j} m={models[k]} j={j} cls={cls(k, j, j === b)} title={tip(k, j)} onToggle={() => toggle(`${k}|${j}`)} />); })}
                    {r && <span className="note"><span className={`v-${r.verdict}`}>{r.verdict}</span>{r.shortlist ? " · shortlist" : ""}{r.best_seed ? `, best seed ${r.best_seed}` : ""}<br />{r.note}</span>}
                  </div>); })}
              </div>
            </div>); })
        ) : (
          <div className="grid">
            {items.slice(0, shown).map(([k, j]) => (
              <figure key={`${k}${j}`}>
                <Sprite m={models[k]} j={j} cls={cls(k, j, false)} title={tip(k, j)} onToggle={() => toggle(`${k}|${j}`)} />
                <figcaption>{k} · {meta.prompts[Math.floor(j / S)].prompt} · s{meta.seeds[j % S]}</figcaption>
              </figure>))}
          </div>
        )}
        <div ref={sentinel} style={{ height: 1 }} />
      </main>
      <footer>
        Team Whiplash · SE 390 MP2 · University of Waterloo. Models, compiler and this bank:{" "}
        <a href="https://github.com/brian-w-zhang/spritesheet">github.com/brian-w-zhang/spritesheet</a>. Training data is
        licensed and is not shown here; every image on this page is our models' output.
      </footer>
    </>
  );
}
