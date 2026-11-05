// SpriteSheet deck: sheet chrome per slide, and sprites drawn as cells from sprites.js (our model's outputs).
(function () {
  const S = window.SPRITES;
  const slides = [...document.querySelectorAll(".slide")];
  const tabs = slides.map((s) => s.dataset.tab);
  const col = (i) => { let s = ""; i += 1; while (i > 0) { const m = (i - 1) % 26; s = String.fromCharCode(65 + m) + s; i = Math.floor((i - 1) / 26); } return s; };
  const NS = "http://www.w3.org/2000/svg";

  slides.forEach((sl, k) => {
    const top = document.createElement("div"); top.className = "ruler-top";
    for (let i = 0; i < 13; i++) { const c = document.createElement("span"); c.textContent = col(i); top.appendChild(c); }
    const left = document.createElement("div"); left.className = "ruler-left";
    for (let r = 1; r <= 27; r++) { const c = document.createElement("span"); c.textContent = r; left.appendChild(c); }
    const corner = document.createElement("div"); corner.className = "corner";
    const tb = document.createElement("div"); tb.className = "tabs";
    tabs.forEach((t, j) => { const e = document.createElement("span"); e.textContent = t; if (j === k) e.className = "on"; tb.appendChild(e); });
    const meta = document.createElement("span"); meta.className = "meta"; meta.textContent = "Team Whiplash, SE 390 MP2"; tb.appendChild(meta);
    const pg = document.createElement("span"); pg.className = "pg"; pg.textContent = `${k + 1} of ${slides.length}`; tb.appendChild(pg);
    sl.prepend(corner, top, left); sl.appendChild(tb);
  });

  function svgSprite(grid, pal, cell, opts = {}) {
    const n = grid.length, R = opts.ruler ? 22 : 0, W = n * cell + R, H = n * cell + R;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", W); svg.setAttribute("height", H); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const add = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
    if (R) {
      for (let x = 0; x < n; x++) { const t = add("text", { x: R + x * cell + cell / 2, y: 15, "text-anchor": "middle", "font-size": 10, "font-family": "JetBrains Mono", fill: "#85827f" }); t.textContent = col(x); }
      for (let y = 0; y < n; y++) { const t = add("text", { x: R - 5, y: R + y * cell + cell / 2 + 3.5, "text-anchor": "end", "font-size": 10, "font-family": "JetBrains Mono", fill: "#85827f" }); t.textContent = y + 1; }
    }
    add("rect", { x: R, y: R, width: n * cell, height: n * cell, fill: pal[0] });
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const v = grid[y][x]; if (!v) continue;
      add("rect", { x: R + x * cell, y: R + y * cell, width: cell, height: cell, fill: pal[v] });
    }
    if (cell >= 8) {   // cell lines: the pixels are cells
      let d = "";
      for (let i = 0; i <= n; i++) { d += `M${R + i * cell} ${R}V${R + n * cell}M${R} ${R + i * cell}H${R + n * cell}`; }
      add("path", { d, stroke: "rgba(255,255,255,0.38)", "stroke-width": 1, fill: "none" });
    }
    if (opts.sel) {
      const [sy, sx] = opts.sel;
      add("rect", { x: R + sx * cell - 1, y: R + sy * cell - 1, width: cell + 2, height: cell + 2, fill: "none", stroke: "#2d56ce", "stroke-width": 3 });
      add("rect", { x: R + sx * cell + cell - 3, y: R + sy * cell + cell - 3, width: 6, height: 6, fill: "#2d56ce", stroke: "#fff", "stroke-width": 1 });
    }
    return svg;
  }

  document.querySelectorAll(".sprite").forEach((el) => {
    const m = S.models[el.dataset.model], g = m.grids[+el.dataset.i];
    const cell = el.classList.contains("hero") ? 19 : 18;
    el.appendChild(svgSprite(g, m.palette, cell, { ruler: !!el.dataset.ruler, sel: el.dataset.sel ? el.dataset.sel.split(",").map(Number) : null }));
  });

  document.querySelectorAll(".strip").forEach((el) => {
    const m = S.models[el.dataset.model];
    const from = +(el.dataset.from || 0), n = +(el.dataset.n || 8);
    const all = S.cases.concat(S.extra || []);
    for (let i = from; i < from + n; i++) {
      const g = m.grids[i], cell = Math.round(96 / g.length);
      const f = document.createElement("figure");
      f.appendChild(svgSprite(g, m.palette, cell));
      if (el.dataset.labels) { const c = document.createElement("figcaption"); c.textContent = all[i].prompt.replace(/^an? /, ""); f.appendChild(c); }
      if (el.parentElement.classList.contains("demo-right") && !el.dataset.labels) { const c = document.createElement("figcaption"); c.textContent = `seed ${all[i].seed}`; f.appendChild(c); }
      el.appendChild(f);
    }
  });

  document.querySelectorAll(".cap[data-caps]").forEach((el) => {
    S.cases.forEach((c) => { const s = document.createElement("span"); s.textContent = `${c.prompt}, seed ${c.seed}`; el.appendChild(s); });
  });
})();
