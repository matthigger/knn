// Page state, drawing and controls. The stage is one SVG: the feature plane
// with shaded decision regions (classification) or the x-y scatter with the
// k-NN curve (regression), plus the query and its k neighbors. The chart
// SVG in the side panel plots training and testing error against k; it,
// the testing samples and the error readout show only while validating.

const SVGNS = "http://www.w3.org/2000/svg";
const COLORS = ["#2563eb", "#ea580c", "#9333ea"];
// Region shade per class (RGB).
const PALE = [[191, 219, 254], [254, 215, 170], [233, 213, 255]];
const CURVE = "#ea580c";
// Testing samples: per class (classification) or in all (regression).
const N_TEST = { class: 150, reg: 300 };
const N_RANGE = { class: [5, 150], reg: [5, 200] };
// Region grid cells per side, and the coarser grid used mid-drag.
const GRID = 150;
const GRID_FAST = 60;

const VIEW = {
  class: { w: 700, h: 700, x0: 40, x1: 680, y0: 20, y1: 660,
    xdom: [-1.15, 1.15], ydom: [-1.15, 1.15] },
  reg: { w: 1000, h: 560, x0: 70, x1: 980, y0: 20, y1: 500,
    xdom: [-0.15, 1.15], ydom: [-2.2, 2.2] },
};
// Chart plot box, in its 340 x 176 viewBox.
const CH = { x0: 46, x1: 330, y0: 10, y1: 140 };

// Each tab keeps its own settings.
const state = {
  mode: "class",
  validate: false,
  class: { set: "swiss", n: 60, noise: 0.3, k: 5, metric: "l2", seed: 1,
    q: [0.3, 0.35] },
  reg: { set: "sine", n: 40, noise: 0.3, k: 5, metric: "l2", seed: 1,
    q: [0.4] },
};
// Derived by regen() and refit(): the samples, and err[k] for k = 1..n.
let train, test, errTrain, errTest;
// The query's neighbors and estimate, from drawQuery().
let qres;

const svg = document.getElementById("stage");
const chart = document.getElementById("chart");
const els = {};
const canvas = document.createElement("canvas");

function cur() { return state[state.mode]; }
function numClasses() {
  return state.mode === "class" ? CLASS_SETS[cur().set].C : 0;
}
function V() { return VIEW[state.mode]; }
function sx(v) {
  const p = V();
  return p.x0 + (v - p.xdom[0]) / (p.xdom[1] - p.xdom[0]) * (p.x1 - p.x0);
}
function sy(v) {
  const p = V();
  return p.y1 - (v - p.ydom[0]) / (p.ydom[1] - p.ydom[0]) * (p.y1 - p.y0);
}
function ix(px) {
  const p = V();
  return p.xdom[0] + (px - p.x0) / (p.x1 - p.x0) * (p.xdom[1] - p.xdom[0]);
}
function iy(py) {
  const p = V();
  return p.ydom[0] + (p.y1 - py) / (p.y1 - p.y0) * (p.ydom[1] - p.ydom[0]);
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function node(tag, attrs = {}) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/** Pointer position in the viewBox units of an SVG. */
function svgPoint(el, e) {
  const pt = el.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(el.getScreenCTM().inverse());
}

function fmt(x, digits = 2) {
  return (x < 0 ? "−" : "") + Math.abs(x).toFixed(digits);
}

/** Error as shown: a percentage (classification) or an MSE. */
function fmtErr(v) {
  if (v === 0) return state.mode === "class" ? "0%" : "0";
  if (state.mode === "class") {
    const p = 100 * v;
    return p.toFixed(p < 10 && Math.round(10 * p) % 10 ? 1 : 0) + "%";
  }
  return v < 0.1 ? v.toFixed(3) : v.toFixed(2);
}

/** Chart tick label: the nice value exactly. */
function fmtTick(v) {
  if (state.mode !== "class") return fmtErr(v);
  return `${+(100 * v).toFixed(1)}%`;
}

function dot(c) {
  return `<span class="dot" style="background:${COLORS[c]}"></span>`;
}

// ------------------------------------------------------------------ data

/** Draw fresh samples from the current settings (drags are lost). */
function regen() {
  const s = cur();
  if (state.mode === "class") {
    train = makeClass(s.set, s.n, s.noise, s.seed);
    test = makeClass(s.set, N_TEST.class, s.noise, s.seed + 100003);
  } else {
    train = makeReg(s.set, s.n, s.noise, s.seed);
    test = makeReg(s.set, N_TEST.reg, s.noise, s.seed + 100003);
  }
  s.k = Math.min(s.k, train.y.length);
  invalidate({ fit: true });
}

function refit() {
  const C = numClasses(), m = cur().metric;
  errTrain = errorByK(train.X, train.y, train.X, train.y, C, m);
  errTest = errorByK(test.X, test.y, train.X, train.y, C, m);
}

/** Smallest k with the lowest testing error. */
function bestK() {
  let b = 1;
  for (let k = 2; k < errTest.length; k++) if (errTest[k] < errTest[b]) b = k;
  return b;
}

// ------------------------------------------------------------- scheduling

// What the next frame must redo: refit the errors, redraw the regions or
// curve (coarsely if fast). Samples, query, readout and chart always redraw.
const todo = { fit: false, model: false, fast: false };
let queued = false;

function invalidate({ fit = false, model = false, fast = false } = {}) {
  todo.fast = queued ? todo.fast && fast : fast;
  todo.fit = todo.fit || fit;
  todo.model = todo.model || model || fit;
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => {
    queued = false;
    draw();
  });
}

function draw() {
  if (todo.fit) refit();
  if (todo.model) {
    if (state.mode === "class") drawRegions(todo.fast ? GRID_FAST : GRID);
    else drawCurve(todo.fast ? 300 : 700);
  }
  todo.fit = todo.model = false;
  updateLegend();
  drawSamples();
  drawQuery();
  updateReadout();
  if (state.validate) drawChart();
  updateControls();
}

// ------------------------------------------------------------------ build

function build() {
  const p = V();
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${p.w} ${p.h}`);
  const w = p.x1 - p.x0, h = p.y1 - p.y0;
  svg.innerHTML = `<clipPath id="plotclip"><rect x="${p.x0}" y="${p.y0}"
    width="${w}" height="${h}"/></clipPath>`;

  const axes = node("g", { class: "axes" });
  let a = "";
  if (state.mode === "class") {
    for (const v of [-1, 0, 1]) {
      a += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${p.y0}"
        y2="${p.y1}"/><line class="grid" x1="${p.x0}" x2="${p.x1}"
        y1="${sy(v)}" y2="${sy(v)}"/>
        <text class="tick" x="${sx(v)}" y="${p.y1 + 16}"
        text-anchor="middle">${fmt(v, 0)}</text>
        <text class="tick" x="${p.x0 - 7}" y="${sy(v) + 4}"
        text-anchor="end">${fmt(v, 0)}</text>`;
    }
    a += `<text class="axis-label" x="${(p.x0 + p.x1) / 2}" y="${p.y1 + 34}"
      text-anchor="middle"><tspan class="sym">x</tspan><tspan
      baseline-shift="sub" font-size="10">1</tspan></text>
      <text class="axis-label" text-anchor="middle"
      transform="translate(14 ${(p.y0 + p.y1) / 2}) rotate(-90)"><tspan
      class="sym">x</tspan><tspan baseline-shift="sub"
      font-size="10">2</tspan></text>`;
  } else {
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      a += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${p.y0}"
        y2="${p.y1}"/><text class="tick" x="${sx(v)}" y="${p.y1 + 18}"
        text-anchor="middle">${v}</text>`;
    }
    for (const v of [-2, -1, 0, 1, 2]) {
      a += `<line class="grid" x1="${p.x0}" x2="${p.x1}" y1="${sy(v)}"
        y2="${sy(v)}"/><text class="tick" x="${p.x0 - 8}" y="${sy(v) + 4}"
        text-anchor="end">${fmt(v, 0)}</text>`;
    }
    a += `<text class="axis-label" x="${(p.x0 + p.x1) / 2}" y="${p.y1 + 44}"
      text-anchor="middle"><tspan class="sym">x</tspan></text>
      <text class="axis-label" x="22" y="${(p.y0 + p.y1) / 2 + 5}"
      text-anchor="middle"><tspan class="sym">y</tspan></text>`;
  }
  axes.innerHTML = a;

  els.region = node("image", { x: p.x0, y: p.y0, width: w, height: h,
    preserveAspectRatio: "none" });
  els.curve = node("g", { class: "curve", "clip-path": "url(#plotclip)" });
  els.test = node("g", { class: "test", "clip-path": "url(#plotclip)" });
  els.nbhd = node("g", { class: "nbhd", "clip-path": "url(#plotclip)" });
  els.train = node("g", { class: "train" });
  els.query = node("g", { class: "query" });
  const frame = node("rect", { class: "frame", x: p.x0, y: p.y0, width: w,
    height: h });
  if (state.mode === "class") svg.append(els.region);
  svg.append(axes, els.curve, els.test, frame, els.nbhd, els.train,
    els.query);
}

function buildSets() {
  const sets = state.mode === "class" ? CLASS_SETS : REG_SETS;
  document.getElementById("sets").innerHTML = Object.entries(sets)
    .map(([key, s]) => `<button data-set="${key}" role="radio">${s.name}
      </button>`).join("");
  for (const b of document.querySelectorAll("[data-set]")) {
    b.onclick = () => {
      cur().set = b.dataset.set;
      regen();
    };
  }
}

function updateLegend() {
  let h = "";
  if (state.mode === "class") {
    for (let c = 0; c < numClasses(); c++) {
      h += `<span class="key">${dot(c)}<span><i>y</i> = ${c}</span></span>`;
    }
  } else {
    h += `<span class="key"><span class="line-key"></span><span>k-NN
        <i>&ycirc;</i>(<i>x</i>)</span></span>
      <span class="key"><span class="line-key dash"></span><span>true
        <i>f</i>(<i>x</i>)</span></span>`;
  }
  // Classification points take their class color, so the key is neutral.
  const fill = state.mode === "class" ? `<span class="dot fill"></span>`
    : dot(0);
  h += `<span class="key">${fill}training sample</span>`;
  if (state.validate) {
    h += `<span class="key"><span class="dot ring"></span>test sample</span>`;
  }
  h += `<span class="key"><span class="dot qkey"></span>query</span>`;
  document.getElementById("legend").innerHTML = h;
}

// ------------------------------------------------------------------ draw

function drawRegions(G) {
  const C = numClasses(), s = cur(), p = V();
  const winner = gridVotes(G, p.xdom, p.ydom, train.X, train.y, C, s.k,
    s.metric);
  canvas.width = canvas.height = G;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(G, G);
  for (let j = 0; j < G * G; j++) {
    img.data.set([...PALE[winner[j]], 255], 4 * j);
  }
  ctx.putImageData(img, 0, 0);
  els.region.setAttribute("href", canvas.toDataURL());
}

function drawCurve(G) {
  const s = cur(), p = V(), f = REG_SETS[s.set].f;
  let est = "", truth = "";
  for (let j = 0; j <= G; j++) {
    const x = p.xdom[0] + j / G * (p.xdom[1] - p.xdom[0]);
    const { order } = neighborOrder([x], train.X, s.metric);
    const cmd = j ? "L" : "M";
    est += `${cmd}${sx(x).toFixed(1)} ${sy(average(order, train.y, s.k))
      .toFixed(1)}`;
    truth += `${cmd}${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)}`;
  }
  els.curve.innerHTML = `<path class="truth" d="${truth}"/>
    <path class="est" d="${est}" stroke="${CURVE}"/>`;
}

function drawSamples() {
  const isClass = state.mode === "class";
  const col = i => (isClass ? COLORS[train.y[i]] : COLORS[0]);
  els.train.innerHTML = train.X.map((x, i) => `<circle class="s"
    data-i="${i}" cx="${sx(x[0]).toFixed(1)}"
    cy="${sy(isClass ? x[1] : train.y[i]).toFixed(1)}" r="5.5"
    fill="${col(i)}"/>`).join("");
  els.test.innerHTML = !state.validate ? "" : test.X.map((x, i) => `<circle
    class="t" cx="${sx(x[0]).toFixed(1)}"
    cy="${sy(isClass ? x[1] : test.y[i]).toFixed(1)}" r="3.6"
    stroke="${isClass ? COLORS[test.y[i]] : "#8a93a3"}"/>`).join("");
}

function drawQuery() {
  const s = cur(), q = s.q, k = s.k;
  const { order, d } = neighborOrder(q, train.X, s.metric);
  const nb = order.slice(0, k), r = d[order[k - 1]];
  const isClass = state.mode === "class";
  const ptY = i => (isClass ? train.X[i][1] : train.y[i]);
  let h = "";
  const ring = (i, cls) => `<circle class="${cls}" cx="${sx(train.X[i][0])}"
    cy="${sy(ptY(i))}" r="9.5"/>`;
  let rings = nb.map(i => ring(i, "ring")).join("");

  if (isClass) {
    const C = numClasses();
    qres = { ...vote(order, train.y, k, C), nb, r };
    rings = nb.map((i, j) => ring(i, j < qres.kUsed ? "ring" : "ring dropped"))
      .join("");
    const cx = sx(q[0]), cy = sy(q[1]), rp = sx(r) - sx(0);
    h += s.metric === "l1"
      ? `<polygon class="ball" points="${cx},${cy - rp} ${cx + rp},${cy}
        ${cx},${cy + rp} ${cx - rp},${cy}"/>`
      : `<circle class="ball" cx="${cx}" cy="${cy}" r="${rp}"/>`;
    h += nb.map(i => `<line class="spoke" x1="${cx}" y1="${cy}"
      x2="${sx(train.X[i][0])}" y2="${sy(train.X[i][1])}"/>`).join("");
    els.nbhd.innerHTML = h + rings;
    els.query.innerHTML = `<g transform="translate(${cx} ${cy})">
      <circle class="q-mark" r="11" fill="${COLORS[qres.winner]}"/>
      <circle r="3" fill="#fff"/></g>`;
  } else {
    const yhat = average(order, train.y, k);
    qres = { yhat, nb, r };
    const p = V(), a = sx(q[0] - r), b = sx(q[0] + r);
    const cx = sx(q[0]), cy = sy(yhat);
    h += `<rect class="band" x="${a}" y="${p.y0}" width="${b - a}"
      height="${p.y1 - p.y0}"/>
      <line class="guide" x1="${cx}" x2="${cx}" y1="${p.y0}" y2="${p.y1}"/>
      <line class="avg" x1="${a}" x2="${b}" y1="${cy}" y2="${cy}"/>`;
    els.nbhd.innerHTML = h + rings;
    els.query.innerHTML = `<g transform="translate(${cx} ${cy})">
      <circle class="q-mark" r="9" fill="${CURVE}"/>
      <circle r="2.6" fill="#fff"/></g>
      <path class="q-handle" d="M${cx} ${p.y1} l-7 12 h14 z"/>`;
  }
}

function drawChart() {
  const n = train.y.length, P = CH, k = cur().k, best = bestK();
  const kx = j => P.x0 + Math.log(j) / Math.log(n) * (P.x1 - P.x0);
  // Sized to the low-k dip rather than the large-k end, which is clipped.
  let top = 0;
  for (let j = 1; j <= n; j++) top = Math.max(top, errTrain[j], errTest[j]);
  top = niceCeil(Math.min(top, Math.max(3 * errTest[best], 0.05)));
  const ey = v => P.y1 - Math.min(v, top) / top * (P.y1 - P.y0);
  const path = err => {
    let d = "";
    for (let j = 1; j <= n; j++) {
      d += `${j > 1 ? "L" : "M"}${kx(j).toFixed(1)} ${ey(err[j]).toFixed(1)}`;
    }
    return d;
  };

  let h = `<clipPath id="chartclip"><rect x="${P.x0}" y="${P.y0 - 4}"
    width="${P.x1 - P.x0 + 4}" height="${P.y1 - P.y0 + 4}"/></clipPath>`;
  for (const t of [1, 2, 5, 10, 20, 50, 100, 200, 500]) {
    if (t > n) break;
    h += `<line class="grid" x1="${kx(t)}" x2="${kx(t)}" y1="${P.y0}"
      y2="${P.y1}"/><text class="tick" x="${kx(t)}" y="${P.y1 + 14}"
      text-anchor="middle">${t}</text>`;
  }
  for (const v of [0, top / 2, top]) {
    h += `<line class="grid" x1="${P.x0}" x2="${P.x1}" y1="${ey(v)}"
      y2="${ey(v)}"/><text class="tick" x="${P.x0 - 5}" y="${ey(v) + 4}"
      text-anchor="end">${fmtTick(v)}</text>`;
  }
  h += `<rect class="frame" x="${P.x0}" y="${P.y0}" width="${P.x1 - P.x0}"
      height="${P.y1 - P.y0}"/>
    <text class="axis-label" x="${(P.x0 + P.x1) / 2}" y="${P.y1 + 32}"
      text-anchor="middle"><tspan class="sym">k</tspan> (log scale)</text>
    <g clip-path="url(#chartclip)">
      <path class="c-train" d="${path(errTrain)}"/>
      <path class="c-test" d="${path(errTest)}"/></g>
    <circle class="c-best" cx="${kx(best)}" cy="${ey(errTest[best])}" r="5"/>
    <line class="c-k" x1="${kx(k)}" x2="${kx(k)}" y1="${P.y0}"
      y2="${P.y1}"/>
    <circle class="c-dot train" cx="${kx(k)}" cy="${ey(errTrain[k])}"
      r="3.5"/>
    <circle class="c-dot" cx="${kx(k)}" cy="${ey(errTest[k])}" r="3.5"/>`;
  chart.innerHTML = h;

  const what = state.mode === "class" ? "Error rate" : "MSE";
  document.getElementById("chart-title").innerHTML = `${what} against
    <i>k</i> <span class="lg test"></span>testing
    <span class="lg train"></span>training`;
  document.getElementById("chart-note").innerHTML = `Lowest testing
    ${state.mode === "class" ? "error" : "MSE"} (green ring):
    <i>k</i>&nbsp;=&nbsp;${best}, ${fmtErr(errTest[best])}. Click or drag
    the chart to set <i>k</i>.`;
}

/** Round v up to 1, 2, 2.5 or 5 times a power of ten. */
function niceCeil(v) {
  if (!(v > 0)) return 1;
  const e = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (v <= m * e + 1e-12) return m * e;
}

// --------------------------------------------------------------- readout

function updateReadout() {
  const s = cur(), k = s.k, n = train.y.length, nt = test.y.length;
  const isClass = state.mode === "class";
  let h = `<h3>Query</h3>`;
  if (isClass) {
    h += `<div class="row"><span>Position <i>x</i></span><span class="val">
        (${fmt(s.q[0])}, ${fmt(s.q[1])})</span></div>
      <div class="row"><span>Votes of the ${k} nearest</span>
        <span class="val">${qres.counts.map((c, j) => `${dot(j)}${c}`)
          .join(" &nbsp;")}</span></div>
      <div class="row"><span>Estimate <i>&ycirc;</i></span><span
        class="val">${dot(qres.winner)}<i>y</i> = ${qres.winner}</span>
      </div>`;
    if (qres.kUsed < k) {
      h += `<p class="warn">Tied vote: the farthest
        ${k - qres.kUsed === 1 ? "neighbor is" : `${k - qres.kUsed}
        neighbors are`} dropped (dashed rings), and the
        ${qres.kUsed} nearest decide.</p>`;
    }
  } else {
    const f = REG_SETS[s.set].f;
    const xs = qres.nb.map(i => train.X[i][0]);
    h += `<div class="row"><span>Position <i>x</i></span><span class="val">
        ${fmt(s.q[0])}</span></div>
      <div class="row"><span>The ${k} nearest <i>x<sub>i</sub></i></span>
        <span class="val">${fmt(Math.min(...xs))} to
        ${fmt(Math.max(...xs))}</span></div>
      <div class="row"><span>Estimate <i>&ycirc;</i> (their mean
        <i>y<sub>i</sub></i>)</span><span class="val">${fmt(qres.yhat)}
        </span></div>
      <div class="row"><span>True <i>f</i>(<i>x</i>)</span><span
        class="val">${fmt(f(s.q[0]))}</span></div>`;
  }
  const what = isClass ? "error" : "MSE";
  if (!state.validate) {
    h += `<p class="muted">Check <b>Validate on test samples</b> to measure
      the training and testing ${what}.</p>`;
  }
  document.getElementById("readout").innerHTML = h;
  if (!state.validate) return;
  document.getElementById("errs").innerHTML = `<h3>At <i>k</i> = ${k}</h3>
    <div class="row"><span>Training ${what} (${n} samples)</span>
      <span class="val">${fmtErr(errTrain[k])}</span></div>
    <div class="row"><span>Testing ${what} (${nt} new samples)</span>
      <span class="val">${fmtErr(errTest[k])}</span></div>`;
}

// -------------------------------------------------------------- controls

const kInput = document.getElementById("k");
const nInput = document.getElementById("n");
const noiseInput = document.getElementById("noise");

function updateControls() {
  const s = cur(), n = train.y.length, C = numClasses();
  kInput.max = n;
  kInput.value = s.k;
  document.getElementById("kval").innerHTML = s.k === n
    ? `<i>k</i> = ${n} (every sample)` : `<i>k</i> = ${s.k}`;
  [nInput.min, nInput.max] = N_RANGE[state.mode];
  nInput.value = s.n;
  document.getElementById("nval").textContent = C
    ? `${s.n} per class (${n} in all)` : `${n}`;
  noiseInput.value = s.noise;
  document.getElementById("noiseval").textContent = s.noise.toFixed(2);
  document.getElementById("validate").checked = state.validate;
  document.getElementById("chart-card").hidden = !state.validate;
  for (const b of document.querySelectorAll("[data-set]")) {
    b.setAttribute("aria-checked", b.dataset.set === s.set);
  }
  for (const b of document.querySelectorAll("[data-metric]")) {
    b.setAttribute("aria-checked", b.dataset.metric === s.metric);
  }
  for (const t of document.querySelectorAll(".tab")) {
    t.setAttribute("aria-selected", t.dataset.mode === state.mode);
  }
  for (const el of document.querySelectorAll("[data-show]")) {
    el.hidden = el.dataset.show !== state.mode;
  }
}

function setMode(mode) {
  state.mode = mode;
  history.replaceState(null, "", mode === "reg" ? "#regression"
    : "#classification");
  build();
  buildSets();
  regen();
}

function setK(k, fast) {
  cur().k = clamp(Math.round(k), 1, train.y.length);
  invalidate({ model: true, fast });
}

// Sliders redraw coarsely while moving and in full once released.
kInput.oninput = () => setK(+kInput.value, true);
kInput.onchange = () => setK(+kInput.value, false);
nInput.oninput = () => { cur().n = +nInput.value; regen(); };
noiseInput.oninput = () => { cur().noise = +noiseInput.value; regen(); };
for (const b of document.querySelectorAll("[data-metric]")) {
  b.onclick = () => {
    cur().metric = b.dataset.metric;
    invalidate({ fit: true });
  };
}
document.getElementById("validate").onchange = e => {
  state.validate = e.target.checked;
  invalidate();
};
document.getElementById("resample").onclick = () => {
  cur().seed += 1;
  regen();
};
document.getElementById("reset").onclick = regen;
for (const t of document.querySelectorAll(".tab")) {
  t.onclick = () => {
    if (t.dataset.mode !== state.mode) setMode(t.dataset.mode);
  };
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey) return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    e.preventDefault();
    setK(cur().k + (e.key === "ArrowRight" ? 1 : -1), false);
  }
});

// Dragging: a training sample moves it (and refits); anywhere else moves
// the query. Capture on the SVG survives the redraw replacing the target.
let drag = null;

function dragTo(e) {
  const p = svgPoint(svg, e), v = V(), s = cur();
  const x = clamp(ix(p.x), ...v.xdom), y = clamp(iy(p.y), ...v.ydom);
  if (drag.query) {
    s.q = state.mode === "class" ? [x, y] : [x];
    invalidate();
    return;
  }
  if (state.mode === "class") {
    train.X[drag.i] = [x, y];
  } else {
    train.X[drag.i] = [x];
    train.y[drag.i] = y;
  }
  invalidate({ fit: true, fast: true });
}

svg.addEventListener("pointerdown", e => {
  if (e.button !== 0) return;
  const t = e.target.closest(".s");
  drag = t ? { i: +t.dataset.i } : { query: true };
  svg.setPointerCapture(e.pointerId);
  e.preventDefault();
  dragTo(e);
});
svg.addEventListener("pointermove", e => { if (drag) dragTo(e); });
function endDrag() {
  if (drag && !drag.query) invalidate({ model: true });
  drag = null;
}
svg.addEventListener("pointerup", endDrag);
svg.addEventListener("pointercancel", endDrag);

// The chart sets k from the pointer's position on its log axis.
let chartDrag = false;
function chartK(e, fast) {
  const p = svgPoint(chart, e);
  const t = clamp((p.x - CH.x0) / (CH.x1 - CH.x0), 0, 1);
  setK(Math.exp(t * Math.log(train.y.length)), fast);
}
chart.addEventListener("pointerdown", e => {
  chartDrag = true;
  chart.setPointerCapture(e.pointerId);
  chartK(e, true);
});
chart.addEventListener("pointermove", e => {
  if (chartDrag) chartK(e, true);
});
chart.addEventListener("pointerup", e => {
  chartDrag = false;
  chartK(e, false);
});

// Footer build stamp: the deployed commit (linked) and build time.
(function () {
  const el = document.getElementById("build");
  const repo = "https://github.com/matthigger/knn";
  if (!BUILD) { el.textContent = "local copy"; return; }
  const when = new Date(BUILD.time).toLocaleString("en-US", {
    dateStyle: "medium", timeStyle: "short" });
  el.innerHTML = `build <a href="${repo}/commit/${BUILD.sha}">${
    BUILD.sha.slice(0, 7)}</a>, ${when}`;
})();

setMode(location.hash === "#regression" ? "reg" : "class");
