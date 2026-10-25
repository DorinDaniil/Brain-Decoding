"""Build index.html (an interactive literature map) from knowledge_graph.json."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
graph = json.loads((here / "knowledge_graph.json").read_text(encoding="utf-8"))
data = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")

page = r"""<title>Литература римановского леса</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=PT+Serif:wght@400;700&family=PT+Sans:wght@400;700&family=PT+Mono&display=swap">
<style>
/* Layout: header with search and topic chips; graph canvas beside a reading card; full list of works below. */
:root {
  --bg: #f4f6f8; --surface: #ffffff; --ink: #1c2430; --muted: #5d6877; --line: #d9dee5;
  --accent: #2f56c7; --barachant: #b86e00; --focus: #2f56c7;
  --g-geometry: #6c5aa8; --g-riemannian: #2f7d8c; --g-transfer: #3a8a4f; --g-ensemble: #9a4a6c;
  --g-eeg: #b2582f; --g-clustered: #5e7290; --g-data: #7a7a3a; --g-concept: #8a94a3; --g-author: #b86e00;
  --display: "PT Serif", Georgia, "Times New Roman", serif;
  --body: "PT Sans", "Segoe UI", Arial, sans-serif;
  --mono: "PT Mono", Consolas, "Courier New", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12161c; --surface: #1a2029; --ink: #e6eaf0; --muted: #9aa5b4; --line: #2c3542;
  --accent: #7f9cff; --barachant: #f0a640; --focus: #7f9cff;
  --g-geometry: #a596e0; --g-riemannian: #5fbfd0; --g-transfer: #6fc78a; --g-ensemble: #d98bb0;
  --g-eeg: #e8946b; --g-clustered: #9fb2cf; --g-data: #c4c47a; --g-concept: #7d8796; --g-author: #f0a640;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12161c; --surface: #1a2029; --ink: #e6eaf0; --muted: #9aa5b4; --line: #2c3542;
  --accent: #7f9cff; --barachant: #f0a640; --focus: #7f9cff;
  --g-geometry: #a596e0; --g-riemannian: #5fbfd0; --g-transfer: #6fc78a; --g-ensemble: #d98bb0;
  --g-eeg: #e8946b; --g-clustered: #9fb2cf; --g-data: #c4c47a; --g-concept: #7d8796; --g-author: #f0a640;
  color-scheme: dark }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 20px 40px; }
.wrap { max-width: 1320px; margin: 0 auto; display: grid; gap: 18px; }
header { display: grid; gap: 6px; }
h1 { font-family: var(--display); font-size: clamp(24px, 3.2vw, 34px); font-weight: 700; margin: 0; text-wrap: balance; }
.lede { color: var(--muted); margin: 0; max-width: 72ch; }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.search { flex: 1 1 260px; min-width: 0; display: flex; }
.search input { width: 100%; font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; padding: 8px 12px; }
.search input:focus-visible, button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font: inherit; font-size: 13px; color: var(--ink); background: var(--surface); border: 1px solid var(--line);
  border-radius: 999px; padding: 4px 11px 4px 9px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
.chip .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--c); }
.chip[aria-pressed="false"] { opacity: .45; }
.chip.bar[aria-pressed="true"] { border-color: var(--barachant); box-shadow: inset 0 0 0 1px var(--barachant); }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 18px; align-items: start; }
@media (max-width: 900px) { .main { grid-template-columns: minmax(0, 1fr); } }
.canvas { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; position: relative; min-width: 0; overflow: hidden; }
svg { display: block; width: 100%; height: min(72vh, 680px); min-height: 420px; touch-action: none; }
.hint { position: absolute; left: 12px; bottom: 10px; font-size: 12px; color: var(--muted); pointer-events: none; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; display: grid; gap: 10px; min-width: 0; }
.card .kind { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.card h2 { font-family: var(--display); font-size: 20px; margin: 0; line-height: 1.3; text-wrap: balance; }
.card p { margin: 0; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 13px; color: var(--muted); }
.key { font-family: var(--mono); font-size: 12.5px; background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px; color: var(--ink); }
.card a { color: var(--accent); }
.nbrs { display: grid; gap: 4px; max-height: 300px; overflow-y: auto; margin: 0; padding: 0; list-style: none; }
.nbrs button { font: inherit; font-size: 13.5px; text-align: left; width: 100%; background: none; border: 0; color: var(--ink);
  padding: 4px 6px; border-radius: 4px; cursor: pointer; display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 8px; }
.nbrs button:hover { background: var(--bg); }
.rel { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.shape { width: 11px; height: 11px; display: inline-block; border: 1.5px solid var(--muted); }
section.list { display: grid; gap: 10px; }
section.list h2 { font-family: var(--display); font-size: 21px; margin: 0; }
.table { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 12px; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); font-weight: 700; position: sticky; top: 0; background: var(--surface); }
td.y { font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; }
tr.pick { cursor: pointer; }
tr.pick:hover td { background: var(--bg); }
.gtag { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; font-size: 13px; }
.gtag .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
.node-label { font-family: var(--body); font-size: 11px; fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; pointer-events: none; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <h1>Литература римановского леса</h1>
    <p class="lede">Карта работ, связанных с римановым лесом, ансамблем касательных классификаторов на случайных подмножествах каналов: римановы методы в ИМК, перенос между испытуемыми, теория ансамблей, ансамбли в декодировании ЭЭГ. Все 110 работ сверены по метаданным издателя; цвет вершины означает тему, оранжевая обводка — работы А. Барашана и его группы.</p>
  </header>

  <div class="toolbar">
    <label class="search" for="q"><input id="q" type="search" placeholder="Поиск по названию, году или ключу BibTeX" autocomplete="off"></label>
    <div class="chips" id="chips" role="group" aria-label="Темы"></div>
  </div>

  <div class="main">
    <div class="canvas">
      <svg id="graph" role="img" aria-label="Граф работ и понятий"></svg>
      <div class="hint">Колесо или щипок — масштаб, перетаскивание — сдвиг, щелчок по вершине — карточка</div>
    </div>
    <aside class="card" id="card" aria-live="polite"></aside>
  </div>

  <div class="legend" id="legend">
    <span><i class="shape" style="border-radius:50%"></i>работа</span>
    <span><i class="shape" style="border-radius:2px"></i>понятие или метод</span>
    <span><i class="shape" style="transform:rotate(45deg) scale(.85)"></i>автор</span>
    <span><i class="shape" style="border-radius:2px;border-style:dashed"></i>набор данных</span>
    <span>размер вершины — число связей</span>
  </div>

  <section class="list">
    <h2>Все работы по годам</h2>
    <div class="table"><table>
      <thead><tr><th>Год</th><th>Работа</th><th>Тема</th><th>Ключ</th></tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
  </section>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const DATA = __DATA__;
const GROUPS = {
  "barachant": ["Барашан и группа", "--g-author"],
  "riemannian-bci": ["Римановы методы в ИМК", "--g-riemannian"],
  "transfer": ["Перенос между испытуемыми", "--g-transfer"],
  "ensemble-theory": ["Теория ансамблей", "--g-ensemble"],
  "eeg-ensembles": ["Ансамбли в ЭЭГ", "--g-eeg"],
  "clustered-dg": ["Сгруппированные данные, новые домены", "--g-clustered"],
  "geometry": ["Геометрия SPD", "--g-geometry"],
  "datasets": ["Данные", "--g-data"], "dataset": ["Данные", "--g-data"],
  "concept": ["Понятия", "--g-concept"], "method": ["Методы", "--g-concept"], "author": ["Авторы", "--g-author"]
};
const REL = { extends: "развивает", uses: "использует", proves: "обосновывает", applies: "применяет", compares: "сравнивает", cites: "цитирует", related: "связано" };
const KIND = { paper: "Работа", concept: "Понятие", method: "Метод", author: "Автор", dataset: "Набор данных" };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const groupKey = g => (g === "dataset" ? "datasets" : g === "method" ? "concept" : g);
const color = n => css((GROUPS[n.group] || GROUPS.concept)[1]);

const nodes = DATA.nodes.map(n => ({ ...n }));
const byId = new Map(nodes.map(n => [n.id, n]));
const links = DATA.edges.filter(e => byId.has(e.source) && byId.has(e.target)).map(e => ({ ...e }));
const nbr = new Map(nodes.map(n => [n.id, []]));
links.forEach(e => { nbr.get(e.source).push([e.target, e.relation, "out"]); nbr.get(e.target).push([e.source, e.relation, "in"]); });
nodes.forEach(n => { n.deg = nbr.get(n.id).length; n.r = n.id === "m_rrf" ? 13 : 4 + Math.sqrt(n.deg) * 1.6; });

// topic chips
const active = new Set(Object.keys(GROUPS).map(groupKey));
let onlyBarachant = false;
const chips = document.getElementById("chips");
const chipGroups = ["riemannian-bci", "transfer", "ensemble-theory", "eeg-ensembles", "clustered-dg", "geometry", "datasets", "concept"];
chipGroups.forEach(g => {
  const b = document.createElement("button");
  b.className = "chip"; b.type = "button"; b.setAttribute("aria-pressed", "true");
  b.style.setProperty("--c", `var(${GROUPS[g][1]})`);
  b.innerHTML = `<span class="dot"></span>${GROUPS[g][0]}`;
  b.onclick = () => { const on = b.getAttribute("aria-pressed") === "true"; b.setAttribute("aria-pressed", String(!on)); on ? active.delete(g) : active.add(g); refresh(); };
  chips.appendChild(b);
});
const bar = document.createElement("button");
bar.className = "chip bar"; bar.type = "button"; bar.setAttribute("aria-pressed", "false");
bar.style.setProperty("--c", "var(--barachant)");
bar.innerHTML = `<span class="dot"></span>Только Барашан`;
bar.onclick = () => { onlyBarachant = !onlyBarachant; bar.setAttribute("aria-pressed", String(onlyBarachant)); refresh(); };
chips.appendChild(bar);

// graph
const svg = d3.select("#graph");
const root = svg.append("g");
const box = () => svg.node().getBoundingClientRect();
const zoom = d3.zoom().scaleExtent([0.2, 5]).on("zoom", ev => root.attr("transform", ev.transform));
svg.call(zoom);
function fit() {
  const shown = nodes.filter(n => visible(n) && Number.isFinite(n.x));
  if (!shown.length) return;
  const { width, height } = box();
  const xs = shown.map(n => n.x), ys = shown.map(n => n.y);
  const x0 = Math.min(...xs) - 30, x1 = Math.max(...xs) + 30, y0 = Math.min(...ys) - 30, y1 = Math.max(...ys) + 30;
  const k = Math.min(width / (x1 - x0), height / (y1 - y0), 2);
  svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
}
const linkSel = root.append("g").selectAll("line").data(links).join("line")
  .attr("stroke", "var(--line)").attr("stroke-width", 1);
const nodeSel = root.append("g").selectAll("g").data(nodes).join("g").style("cursor", "pointer")
  .on("click", (ev, d) => { ev.stopPropagation(); select(d.id); });
nodeSel.each(function (d) {
  const g = d3.select(this);
  const fill = color(d);
  if (d.type === "paper") g.append("circle").attr("r", d.r);
  else if (d.type === "author") g.append("rect").attr("width", d.r * 1.7).attr("height", d.r * 1.7).attr("x", -d.r * .85).attr("y", -d.r * .85).attr("transform", "rotate(45)");
  else g.append("rect").attr("width", d.r * 1.9).attr("height", d.r * 1.9).attr("x", -d.r * .95).attr("y", -d.r * .95).attr("rx", 2)
    .attr("stroke-dasharray", d.type === "dataset" ? "3 2" : null);
  g.select(":first-child").attr("fill", d.type === "paper" ? fill : "var(--surface)").attr("stroke", d.group === "barachant" ? "var(--barachant)" : fill)
    .attr("stroke-width", d.group === "barachant" ? 2.5 : 1.6);
  if (d.id === "m_rrf") g.select(":first-child").attr("fill", "var(--accent)").attr("stroke", "var(--accent)");
});
const labelSel = root.append("g").selectAll("text").data(nodes.filter(n => n.type !== "paper" || n.deg >= 9 || n.group === "barachant"))
  .join("text").attr("class", "node-label").attr("dy", d => -d.r - 4).attr("text-anchor", "middle")
  .text(d => (d.type === "paper" ? `${d.bibkey}` : d.label));
svg.on("click", () => {});

const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(l => (l.source.type === "paper" && l.target.type === "paper" ? 55 : 75)).strength(0.25))
  .force("charge", d3.forceManyBody().strength(-120))
  .force("collide", d3.forceCollide().radius(d => d.r + 3))
  .force("x", d3.forceX(0).strength(0.05))
  .force("y", d3.forceY(0).strength(0.07));
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
sim.stop();
for (let i = 0; i < 300; i++) sim.tick();
draw();
requestAnimationFrame(fit);
sim.on("tick", draw);
if (!reduce) sim.alpha(0.05).restart();
window.addEventListener("resize", () => fit());
nodeSel.call(d3.drag()
  .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
  .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
  .on("end", (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));
function draw() {
  linkSel.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
  labelSel.attr("x", d => d.x).attr("y", d => d.y);
}

// visibility and highlight
let selected = "m_rrf";
const q = document.getElementById("q");
q.addEventListener("input", refresh);
function visible(n) {
  if (onlyBarachant && n.group !== "barachant" && n.id !== "m_rrf" && n.type === "paper") return false;
  if (n.group === "barachant" || n.id === "m_rrf" || n.type === "author") return true;
  return active.has(groupKey(n.group));
}
function matches(n) {
  const t = q.value.trim().toLowerCase();
  if (!t) return true;
  return [n.label, n.bibkey, String(n.year || ""), n.summary].some(v => (v || "").toLowerCase().includes(t));
}
function refresh() {
  const near = new Set([selected, ...nbr.get(selected).map(x => x[0])]);
  const searching = q.value.trim() !== "";
  nodeSel.style("display", d => (visible(d) ? null : "none"))
    .style("opacity", d => (searching ? (matches(d) ? 1 : 0.12) : (near.has(d.id) ? 1 : 0.35)));
  labelSel.style("display", d => (visible(d) ? null : "none"))
    .style("opacity", d => (searching ? (matches(d) ? 1 : 0.1) : (near.has(d.id) ? 1 : 0.3)));
  if (refresh.fitOnChange) fit();
  linkSel.style("display", d => (visible(d.source) && visible(d.target) ? null : "none"))
    .attr("stroke", d => (d.source.id === selected || d.target.id === selected ? "var(--accent)" : "var(--line)"))
    .attr("stroke-width", d => (d.source.id === selected || d.target.id === selected ? 1.6 : 1));
  renderRows();
}

// card
const card = document.getElementById("card");
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function select(id) {
  selected = id;
  const n = byId.get(id);
  const g = GROUPS[n.group] || GROUPS.concept;
  const list = nbr.get(id).map(([o, rel, dir]) => [byId.get(o), rel, dir])
    .sort((a, b) => (b[0].deg - a[0].deg));
  card.innerHTML = `
    <div class="kind">${KIND[n.type] || n.type} · ${esc(g[0])}</div>
    <h2>${esc(n.label)}</h2>
    <div class="meta">${n.year ? `<span>${n.year}</span>` : ""}${n.bibkey ? `<span class="key">${esc(n.bibkey)}</span>` : ""}<span>${n.deg} связей</span></div>
    ${n.summary ? `<p>${esc(n.summary)}</p>` : ""}
    ${n.url ? `<p><a href="${esc(n.url)}" target="_blank" rel="noopener">Открыть источник</a></p>` : ""}
    <ul class="nbrs">${list.map(([o, rel, dir]) => `<li><button type="button" data-id="${esc(o.id)}"><span class="rel">${dir === "out" ? "" : "← "}${REL[rel] || rel}</span><span>${esc(o.type === "paper" ? `${o.label} (${o.year || "—"})` : o.label)}</span></button></li>`).join("")}</ul>`;
  card.querySelectorAll(".nbrs button").forEach(b => b.onclick = () => select(b.dataset.id));
  refresh();
}

// list of works
const rows = document.getElementById("rows");
function renderRows() {
  const papers = nodes.filter(n => n.type === "paper" && visible(n) && matches(n)).sort((a, b) => (a.year || 0) - (b.year || 0) || a.label.localeCompare(b.label));
  rows.innerHTML = papers.map(n => {
    const g = GROUPS[n.group] || GROUPS.concept;
    return `<tr class="pick" data-id="${esc(n.id)}"><td class="y">${n.year || "—"}</td><td>${esc(n.label)}</td><td><span class="gtag" style="--c:var(${g[1]})"><span class="dot"></span>${esc(g[0])}</span></td><td><span class="key">${esc(n.bibkey || "")}</span></td></tr>`;
  }).join("");
  rows.querySelectorAll("tr").forEach(r => r.onclick = () => { select(r.dataset.id); document.querySelector(".canvas").scrollIntoView({ block: "nearest" }); });
}
select("m_rrf");
</script>
"""

page = page.replace("__DATA__", data)
head, body = page.split("</style>", 1)
doc_head = '<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
out = here / "index.html"
out.write_text(doc_head + head + "</style>\n</head>\n<body>\n" + body + "</body>\n</html>\n", encoding="utf-8")
print(out, out.stat().st_size)
