/** The non-game pages: how Sinsan works, training progress, and the rules it plays by.
 * Diagrams are inline SVG; every number is measured (docs/REBUILD_LOG.md) unless marked as an example. */

const PIECE_IMG = (side: 'blue' | 'red', piece: string) => `/assets/pieces/janggihb-ka-wooden/${side}_${piece}.svg`;

// ---------- small SVG helpers (markup strings for static diagrams) ----------

/** A straight arrow with a drawn (not marker) head, ending exactly at (x2, y2). */
function arrowLine(x1: number, y1: number, x2: number, y2: number, cls = 'dg-arrow', head = 9): string {
  const len = Math.hypot(x2 - x1, y2 - y1);
  const ux = (x2 - x1) / len;
  const uy = (y2 - y1) / len;
  const bx = x2 - ux * head;
  const by = y2 - uy * head;
  const px = -uy * head * 0.55;
  const py = ux * head * 0.55;
  return `<line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}" class="${cls}"/>` +
    `<polygon points="${x2},${y2} ${bx + px},${by + py} ${bx - px},${by - py}" class="${cls}-head"/>`;
}

interface MiniBoard {
  cols: number;
  rows: number;
  cell?: number;
  /** Top-left point of a 3x3 palace inside the window, to draw its diagonals. */
  palace?: [number, number][];
  pieces?: { c: number; r: number; img: string; faded?: boolean }[];
  dots?: [number, number][];
  captures?: [number, number][];
  crosses?: [number, number][];
  paths?: [number, number][][];
  caption?: string;
}

/** A window of the Janggi board: lines, palace diagonals, pieces, and move markers. */
function miniBoard(b: MiniBoard): string {
  const cell = b.cell ?? 34;
  const m = cell * 0.6;
  const w = (b.cols - 1) * cell + 2 * m;
  const h = (b.rows - 1) * cell + 2 * m;
  const X = (c: number) => m + c * cell;
  const Y = (r: number) => m + r * cell;
  const parts: string[] = [`<rect x="0" y="0" width="${w}" height="${h}" rx="6" class="mb-wood"/>`];
  for (let c = 0; c < b.cols; c++) parts.push(`<line x1="${X(c)}" y1="${Y(0)}" x2="${X(c)}" y2="${Y(b.rows - 1)}" class="mb-line"/>`);
  for (let r = 0; r < b.rows; r++) parts.push(`<line x1="${X(0)}" y1="${Y(r)}" x2="${X(b.cols - 1)}" y2="${Y(r)}" class="mb-line"/>`);
  for (const [pc, pr] of b.palace ?? []) {
    parts.push(`<line x1="${X(pc)}" y1="${Y(pr)}" x2="${X(pc + 2)}" y2="${Y(pr + 2)}" class="mb-line"/>`);
    parts.push(`<line x1="${X(pc + 2)}" y1="${Y(pr)}" x2="${X(pc)}" y2="${Y(pr + 2)}" class="mb-line"/>`);
  }
  for (const path of b.paths ?? []) {
    parts.push(`<polyline points="${path.map(([c, r]) => `${X(c)},${Y(r)}`).join(' ')}" class="mb-path"/>`);
  }
  for (const [c, r] of b.dots ?? []) parts.push(`<circle cx="${X(c)}" cy="${Y(r)}" r="${cell * 0.16}" class="mb-dot"/>`);
  const ps = cell * 1.04;
  for (const p of b.pieces ?? []) {
    parts.push(`<image href="${p.img}" x="${X(p.c) - ps / 2}" y="${Y(p.r) - ps / 2}" width="${ps}" height="${ps}"${p.faded ? ' opacity="0.55"' : ''}/>`);
  }
  for (const [c, r] of b.captures ?? []) parts.push(`<circle cx="${X(c)}" cy="${Y(r)}" r="${cell * 0.46}" class="mb-capture"/>`);
  for (const [c, r] of b.crosses ?? []) {
    const d = cell * 0.17;
    parts.push(`<path d="M${X(c) - d},${Y(r) - d} L${X(c) + d},${Y(r) + d} M${X(c) + d},${Y(r) - d} L${X(c) - d},${Y(r) + d}" class="mb-cross"/>`);
  }
  return `<figure class="mini"><svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img">${parts.join('')}</svg>` +
    (b.caption ? `<figcaption>${b.caption}</figcaption>` : '') + `</figure>`;
}

/** Horizontal bar chart for a few labelled values. */
function barChart(rows: { label: string; value: number; display?: string; cls?: string }[], opts: { min?: number; max: number; width?: number; unit?: string }): string {
  const w = opts.width ?? 560;
  const labelW = 190;
  const rowH = 28;
  const min = opts.min ?? 0;
  const x = (v: number) => labelW + ((v - min) / (opts.max - min)) * (w - labelW - 70);
  const h = rows.length * rowH + 8;
  const zero = x(Math.max(min, 0));
  const parts = rows.map((r, i) => {
    const y = i * rowH + 4;
    const x0 = Math.min(zero, x(r.value));
    const bw = Math.max(2, Math.abs(x(r.value) - zero));
    return `<text x="${labelW - 8}" y="${y + 17}" class="ch-label" text-anchor="end">${r.label}</text>` +
      `<rect x="${x0}" y="${y + 5}" width="${bw}" height="${rowH - 10}" rx="3" class="${r.cls ?? 'ch-bar'}"/>` +
      `<text x="${Math.max(x(r.value), zero) + 6}" y="${y + 17}" class="ch-val">${r.display ?? `${r.value}${opts.unit ?? ''}`}</text>`;
  });
  if (min < 0) parts.unshift(`<line x1="${zero}" x2="${zero}" y1="0" y2="${h}" class="ch-zero"/>`);
  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg" role="img">${parts.join('')}</svg>`;
}

// ---------- How it works ----------

function pipelineDiagram(): string {
  const box = (x: number, title: string, sub: string, icon: string) =>
    `<g transform="translate(${x},0)"><rect x="0" y="0" width="170" height="150" rx="10" class="dg-box"/>${icon}` +
    `<text x="85" y="112" class="dg-title" text-anchor="middle">${title}</text><text x="85" y="132" class="dg-sub" text-anchor="middle">${sub}</text></g>`;
  const arrow = (x: number) => arrowLine(x + 2, 75, x + 24, 75);
  let grid = '';
  for (let i = 0; i < 5; i++) grid += `<line x1="${55 + i * 15}" y1="22" x2="${55 + i * 15}" y2="82" class="dg-grid"/><line x1="55" y1="${22 + i * 15}" x2="115" y2="${22 + i * 15}" class="dg-grid"/>`;
  grid += `<circle cx="70" cy="52" r="6" class="dg-cho"/><circle cx="100" cy="37" r="6" class="dg-han"/><circle cx="85" cy="67" r="6" class="dg-cho"/>`;
  let layers = '';
  for (let i = 0; i < 6; i++) layers += `<rect x="${44 + i * 14}" y="24" width="10" height="58" rx="3" class="dg-layer"/>`;
  let tree = `<circle cx="85" cy="24" r="6" class="dg-node-hot"/>`;
  for (const kx of [55, 85, 115]) tree += `<line x1="85" y1="24" x2="${kx}" y2="52" class="dg-edge${kx === 85 ? ' dg-edge-hot' : ''}"/><circle cx="${kx}" cy="52" r="5" class="${kx === 85 ? 'dg-node-hot' : 'dg-node'}"/>`;
  for (const kx of [72, 98]) tree += `<line x1="85" y1="52" x2="${kx}" y2="80" class="dg-edge${kx === 72 ? ' dg-edge-hot' : ''}"/><circle cx="${kx}" cy="80" r="4" class="${kx === 72 ? 'dg-node-hot' : 'dg-node'}"/>`;
  const move = `<circle cx="58" cy="72" r="7" class="dg-cho"/>${arrowLine(64, 67, 112, 30, 'dg-move', 12)}`;
  return `<svg viewBox="0 0 776 150" class="diagram" role="img" aria-label="Position to network to search to move">
    ${box(0, 'Position', 'board + last 3 moves', grid)}${arrow(176)}
    ${box(202, 'Network', '60M transformer', layers)}${arrow(378)}
    ${box(404, 'Search', '800 visits, tree reused', tree)}${arrow(580)}
    ${box(606, 'Move', 'most-visited', move)}
  </svg>`;
}

function networkDiagram(): string {
  let tokens = '';
  for (let r = 0; r < 10; r++) for (let c = 0; c < 9; c++) tokens += `<circle cx="${20 + c * 11}" cy="${38 + r * 11}" r="3.2" class="dg-token"/>`;
  let layers = '';
  for (let i = 0; i < 12; i++) layers += `<rect x="${178 + i * 15}" y="40" width="11" height="96" rx="3" class="dg-layer"/>`;
  const head = (y: number, title: string, chart: string) =>
    `<g transform="translate(470,${y})"><rect x="0" y="0" width="250" height="44" rx="8" class="dg-box"/>` +
    `<text x="12" y="27" class="dg-title2">${title}</text><g transform="translate(150,6)">${chart}</g></g>`;
  const policyBars = [0.42, 0.24, 0.12, 0.08, 0.05, 0.04].map((p, i) => `<rect x="${i * 14}" y="${32 - p * 70}" width="10" height="${p * 70}" class="dg-bar"/>`).join('');
  const wdl = `<rect x="0" y="9" width="54" height="14" rx="2" class="dg-cho"/><rect x="54" y="9" width="38" height="14" rx="2" class="dg-han"/>`;
  const margin = [2, 5, 10, 16, 18, 13, 7, 3].map((v, i) => `<rect x="${i * 11}" y="${32 - v * 1.6}" width="8" height="${v * 1.6}" class="dg-bar2"/>`).join('');
  const midY = 88;
  return `<svg viewBox="0 0 730 176" class="diagram" role="img" aria-label="Network diagram">
    <text x="64" y="22" class="dg-sub" text-anchor="middle">90 points → 90 tokens</text>${tokens}
    ${arrowLine(126, midY, 166, midY)}
    <text x="267" y="22" class="dg-sub" text-anchor="middle">12 attention layers</text>${layers}
    ${arrowLine(362, midY, 462, 30)}${arrowLine(362, midY, 462, midY)}${arrowLine(362, midY, 462, 146)}
    ${head(8, 'Move probabilities', policyBars)}
    ${head(66, 'Win / loss', wdl)}
    ${head(124, 'Point margin', margin)}
  </svg>`;
}

function mctsDiagram(): string {
  const pos: Record<string, [number, number]> = { r: [85, 36], a: [45, 72], b: [85, 72], c: [125, 72], ba: [68, 106], bb: [102, 106], n: [68, 138] };
  const step = (x: number, title: string, sub: string, body: string) =>
    `<g transform="translate(${x},0)"><text x="85" y="16" class="dg-title" text-anchor="middle">${title}</text>${body}` +
    `<text x="85" y="166" class="dg-sub" text-anchor="middle">${sub}</text></g>`;
  const tree = (withNew: boolean) => {
    const edges: [string, string][] = [['r', 'a'], ['r', 'b'], ['r', 'c'], ['b', 'ba'], ['b', 'bb']];
    const hot = ['r', 'b', 'ba'];
    let t = '';
    for (const [p, c] of edges) {
      const h = hot.includes(p) && hot.includes(c);
      t += `<line x1="${pos[p]![0]}" y1="${pos[p]![1]}" x2="${pos[c]![0]}" y2="${pos[c]![1]}" class="dg-edge${h ? ' dg-edge-hot' : ''}"/>`;
    }
    if (withNew) t += `<line x1="68" y1="106" x2="68" y2="138" class="dg-edge dg-edge-new"/>`;
    for (const k of ['r', 'a', 'b', 'c', 'ba', 'bb']) {
      const [x, y] = pos[k]!;
      t += `<circle cx="${x}" cy="${y}" r="${k === 'r' ? 8 : 6}" class="${hot.includes(k) ? 'dg-node-hot' : 'dg-node'}"/>`;
    }
    if (withNew) t += `<circle cx="68" cy="138" r="6" class="dg-node-new"/>`;
    return t;
  };
  // Back-up: arrows running parallel to the path, offset to its left so they never cross a node.
  const up = arrowLine(52, 136, 52, 112, 'dg-arrow-up', 7) + arrowLine(56, 100, 71, 76, 'dg-arrow-up', 7) + arrowLine(73, 64, 73, 44, 'dg-arrow-up', 7);
  const badge = `<rect x="80" y="129" width="56" height="19" rx="9" class="dg-badge"/><text x="108" y="142" class="dg-badge-t" text-anchor="middle">net 62%</text>`;
  return `<svg viewBox="0 0 700 172" class="diagram" role="img" aria-label="The four steps of one search visit">
    ${step(0, '1 · Select', 'follow the best-scoring child', tree(false))}
    ${step(175, '2 · Expand', 'add the new position', tree(true))}
    ${step(350, '3 · Evaluate', 'the network scores it', tree(true) + badge)}
    ${step(525, '4 · Back up', 'update visits and Q to the root', tree(true) + up)}
  </svg>`;
}

function trainingDiagram(): string {
  const box = (x: number, w: number, title: string, num: string, sub: string) =>
    `<g transform="translate(${x},0)"><rect width="${w}" height="100" rx="10" class="dg-box"/>` +
    `<text x="${w / 2}" y="24" class="dg-title" text-anchor="middle">${title}</text>` +
    `<text x="${w / 2}" y="58" class="dg-num" text-anchor="middle">${num}</text>` +
    `<text x="${w / 2}" y="82" class="dg-sub" text-anchor="middle">${sub}</text></g>`;
  const arrow = (x: number) => arrowLine(x + 2, 50, x + 24, 50);
  // Self-play cycle: four stages around a ring, arrows between them.
  const cx = 690;
  const cy = 66;
  const r = 38;
  const stages = ['play', 'train', 'match', 'promote'];
  let ring = `<circle cx="${cx}" cy="${cy}" r="${r}" class="dg-loop"/>`;
  stages.forEach((st, i) => {
    const a = -Math.PI / 2 + (i * Math.PI) / 2;
    const tx = cx + Math.cos(a) * (r + 2);
    const ty = cy + Math.sin(a) * (r + 2);
    const lx = cx + Math.cos(a) * (r + 30);
    const ly = cy + Math.sin(a) * (r + 18) + 4;
    ring += `<circle cx="${tx}" cy="${ty}" r="4" class="dg-node-hot"/><text x="${lx}" y="${ly}" class="dg-sub" text-anchor="middle">${st}</text>`;
    const a2 = a + Math.PI / 4;
    const hx = cx + Math.cos(a2) * r;
    const hy = cy + Math.sin(a2) * r;
    ring += arrowLine(hx - Math.cos(a2 + Math.PI / 2) * 6, hy - Math.sin(a2 + Math.PI / 2) * 6, hx, hy, 'dg-arrow', 7);
  });
  ring += `<text x="${cx}" y="${cy + 5}" class="dg-title" text-anchor="middle">Self-play</text>`;
  return `<svg viewBox="0 0 790 136" class="diagram" role="img" aria-label="Training pipeline"><g transform="translate(0,16)">
    ${box(0, 150, 'Exact rules', '845K', 'positions, 0 mismatches')}${arrow(150)}
    ${box(178, 170, 'Teacher games', '50M', 'positions (FSF NNUE)')}${arrow(348)}
    ${box(376, 160, 'Supervised', '6.5 h', '52K steps, one GPU')}${arrow(536)}
    </g>${ring}
  </svg>`;
}

/** Example position: how the old and new label temperatures turn 8 candidate scores into targets. */
function labelChart(): string {
  const scores = [40, 22, 8, -6, -25, -60, -110, -180];
  const soft = (t: number) => {
    const e = scores.map((s) => Math.exp(s / t));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  };
  const oldP = soft(800);
  const newP = soft(30);
  const w = 560;
  const h = 170;
  const bw = 22;
  let bars = '';
  scores.forEach((s, i) => {
    const x = 50 + i * 62;
    bars += `<rect x="${x}" y="${140 - oldP[i]! * 220}" width="${bw}" height="${oldP[i]! * 220}" class="ch-bar-old"/>`;
    bars += `<rect x="${x + bw + 2}" y="${140 - newP[i]! * 220}" width="${bw}" height="${newP[i]! * 220}" class="ch-bar-new"/>`;
    bars += `<text x="${x + bw}" y="158" class="ch-tick" text-anchor="middle">${s > 0 ? '+' : ''}${s}</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h + 6}" class="chart-svg" role="img" aria-label="Label sharpness example">
    <line x1="40" x2="${w - 10}" y1="140" y2="140" class="ch-zero"/>
    ${bars}
    <text x="40" y="174" class="ch-tick">teacher's score for each of its 8 candidate moves (centipawns, example)</text>
    <rect x="330" y="8" width="12" height="12" class="ch-bar-old"/><text x="348" y="18" class="ch-label">original target (÷800)</text>
    <rect x="330" y="28" width="12" height="12" class="ch-bar-new"/><text x="348" y="38" class="ch-label">new target (÷30)</text>
  </svg>`;
}

function classLegend(): string {
  const rows: [string, string, string][] = [
    ['best', '★ Best', 'the expected move, or < 0.5%'],
    ['excellent', '! Excellent', '< 2%'],
    ['good', '✓ Good', '< 5%'],
    ['inaccuracy', '?! Inaccuracy', '< 10%'],
    ['mistake', '? Mistake', '< 20%'],
    ['blunder', '?? Blunder', '20% or more'],
  ];
  return `<div class="legend-grid">${rows.map(([c, l, t]) => `<span class="ap-cls ap-cls-${c}">${l}</span><span>${t}</span>`).join('')}</div>`;
}

/** Ladder ratings at 800 visits (docs/REBUILD_LOG.md). */
const STRENGTH_GPU = '2,520 (round 13)';
const STRENGTH_BROWSER = '~1,825 at 400 visits (v2, estimate)';

export function renderAbout(container: HTMLElement): void {
  container.innerHTML = `
  <article class="page">
    <h2>How Sinsan works</h2>
    <p class="lead">A neural network for judgement, plus tree search to check it, the recipe of AlphaZero, Leela Chess Zero and KataGo.</p>
    ${pipelineDiagram()}

    <h3>1 · The network</h3>
    ${networkDiagram()}
    <h3 class="sub">Inside the network <span class="badge-live">live</span></h3>
    <p class="caption">The real network on a real position. Try <i>Attention</i> and click pieces.</p>
    <div id="network-viz" class="nv"></div>
    <table class="page-table compact">
      <thead><tr><th>Output</th><th>What it predicts</th><th>Where you see it</th></tr></thead>
      <tbody>
        <tr><td>Move probabilities</td><td>which moves look good at a glance</td><td><i>Prior</i> column</td></tr>
        <tr><td>Win / loss</td><td>who is winning</td><td>the marker on the win bar ("first instinct")</td></tr>
        <tr><td>Point margin</td><td>who wins on points, by how much</td><td>used by <i>Point greed</i></td></tr>
      </tbody>
    </table>

    <h3>2 · The search</h3>
    ${mctsDiagram()}
    <p class="caption">One <b>visit</b> = these four steps. 800 visits per move by default; the most-visited move is played.
      No fixed depth: it reads deep where play is forcing.</p>
    <div class="chart">${barChart([
      { label: 'CNN, instinct only', value: -330, display: '−330', cls: 'ch-bar-old' },
      { label: 'CNN, 800 visits', value: 1079, display: '+1079' },
      { label: 'Transformer, instinct only', value: -40, display: '−40', cls: 'ch-bar-old' },
      { label: 'Transformer, 800 visits', value: 1690, display: '+1690' },
    ], { min: -500, max: 1900 })}</div>
    <p class="caption">Same network, with and without search: search adds about 1,400-1,700 Elo (ladder ratings of the two early networks).</p>

    <h3>3 · Training</h3>
    ${trainingDiagram()}
    <div class="chart">${labelChart()}</div>
    <p class="caption">The biggest fix: the original turned the teacher's scores into almost flat targets, so the network never learned
      which move was best. The new targets follow the teacher's real score gaps.</p>

    <h3>4 · Two engines</h3>
    <table class="page-table compact">
      <thead><tr><th></th><th>GPU server</th><th>In your browser</th></tr></thead>
      <tbody>
        <tr><td>Network</td><td>12-layer transformer, 60M parameters</td><td>6-layer transformer, 10.7M parameters</td></tr>
        <tr><td>Trained on</td><td>teacher games, then its own self-play</td><td>the big network's self-play and the teacher games</td></tr>
        <tr><td>Runs on</td><td>PyTorch, CUDA</td><td>onnxruntime-web, int8 weights (12 MB)</td></tr>
        <tr><td>Search</td><td colspan="2" class="span-cell">the same Rust code: compiled natively for the server, to WebAssembly for the browser</td></tr>
        <tr><td>Speed</td><td>thousands of visits per second</td><td>a few hundred visits per second</td></tr>
        <tr><td>Strength</td><td>${STRENGTH_GPU}</td><td>${STRENGTH_BROWSER}</td></tr>
      </tbody>
    </table>
    <p class="caption">Pick "In-browser" as the network (or the In-browser preset) to play with no server at all; it is also used
      automatically when no server is running.</p>

    <h3>Then vs now</h3>
    <div class="stat-row">
      <div class="stat"><b>−1,424</b><span>original V3, 16 visits</span></div>
      <div class="stat"><b>2,520</b><span>now, 800 visits</span></div>
      <div class="stat"><b>~3,900</b><span>Elo gained</span></div>
    </div>
    <table class="page-table compact">
      <thead><tr><th></th><th>Original</th><th>Now</th></tr></thead>
      <tbody>
        <tr><td>Move targets</td><td>scores ÷ 800 (nearly flat)</td><td>scores ÷ 30</td></tr>
        <tr><td>Teacher</td><td>Fairy-Stockfish classical, depth 5-6</td><td>Fairy-Stockfish NNUE, 100K nodes</td></tr>
        <tr><td>Data</td><td>first 60 plies of 1,800 games</td><td>284,000 full games, 50M positions</td></tr>
        <tr><td>Rules</td><td>endings differed from Fairy-Stockfish</td><td>verified identical</td></tr>
        <tr><td>Model</td><td>56×7 CNN, 0.4 MB</td><td>12-layer transformer, 60M parameters</td></tr>
        <tr><td>After training</td><td>-</td><td>self-play with promotion matches</td></tr>
      </tbody>
    </table>

    <h3>Settings</h3>
    <table class="page-table compact">
      <thead><tr><th>Setting</th><th>Effect</th><th>Default</th></tr></thead>
      <tbody>
        <tr><td>Network</td><td>latest self-play / before self-play / small</td><td>latest</td></tr>
        <tr><td>Visits or time</td><td>more = deeper and stronger</td><td>800</td></tr>
        <tr><td>Randomness</td><td>plays moves in proportion to visits: weaker, varied</td><td>off</td></tr>
        <tr><td>Exploration (cpuct)</td><td>↑ wider search · ↓ digs into favourites</td><td>1.2</td></tr>
        <tr><td>Unvisited-move penalty</td><td>↑ narrower search</td><td>0.33</td></tr>
        <tr><td>Policy softness</td><td>↑ gives disliked moves a look</td><td>1.36</td></tr>
        <tr><td>Point greed</td><td>↑ prefers bigger point wins</td><td>off</td></tr>
      </tbody>
    </table>

    <h3>Reading the panel</h3>
    <div class="two-col">
      <div>
        <div class="legend-title">Win colour (for the side that searched)</div>
        <div class="win-gradient"></div>
        <div class="legend-ends"><span>losing</span><span>even</span><span>winning</span></div>
        <div class="legend-title">Decision tree</div>
        <p class="caption">node size = visits · colour = win chance · italics = the opponent's moves · hover to draw the line</p>
      </div>
      <div>
        <div class="legend-title">Move ratings (win chance given away)</div>
        ${classLegend()}
        <p class="caption">Rated by the opponent Sinsan's searches before and after the move. Accuracy uses Lichess's formula.</p>
      </div>
    </div>

    <h3>Name and origins</h3>
    <p class="caption">Started from <b>Moka</b>, which distilled a strong Go engine into a tiny browser model. <b>神算 (신산)</b>, "calculation of
      divine level", is the nickname of Go professional Lee Chang-ho (이창호); it is a Go term, not a Janggi title.</p>
    <p class="caption"><b>Limits:</b> ratings are against Fairy-Stockfish, not people; the 1단 goal (≈1,700-1,800 on pychess) still needs human games.
      The browser-only version comes later.</p>
  </article>`;
}

// ---------- Rules ----------

export function renderRules(container: HTMLElement): void {
  const B = (p: string) => PIECE_IMG('blue', p);
  const R = (p: string) => PIECE_IMG('red', p);
  const material = [
    ['chariot', 13, 2], ['cannon', 7, 2], ['horse', 5, 2], ['elephant', 3, 2], ['guard', 3, 2], ['soldier', 2, 5],
  ] as const;
  const total = material.reduce((a, [, v, n]) => a + v * n, 0);
  const pointsBar = (side: 'cho' | 'han') => {
    let x = 0;
    const segs = material.map(([, v, n]) => {
      const w = ((v * n) / 75) * 520;
      const seg = `<rect x="${x}" y="0" width="${w - 2}" height="26" rx="3" class="pt-${side}"/><text x="${x + w / 2 - 1}" y="17" class="pt-t" text-anchor="middle">${v * n}</text>`;
      x += w;
      return seg;
    }).join('');
    const extra = side === 'han' ? `<rect x="${x}" y="0" width="${(1.5 / 75) * 520}" height="26" rx="2" class="pt-komi"/>` : '';
    return segs + extra;
  };
  container.innerHTML = `
  <article class="page">
    <h2>Rules</h2>
    <p class="lead">Fairy-Stockfish <code>janggi</code>, as on pychess.org. Games are never drawn.</p>

    <h3>How the pieces move</h3>
    <div class="mini-grid">
      ${miniBoard({ cols: 7, rows: 6, pieces: [{ c: 2, r: 3, img: B('chariot') }, { c: 5, r: 3, img: R('pawn') }],
        dots: [[0, 3], [1, 3], [3, 3], [4, 3], [2, 0], [2, 1], [2, 2], [2, 4], [2, 5]], captures: [[5, 3]],
        caption: '<b>車 Chariot</b> · 13 points<br>any distance in a line' })}
      ${miniBoard({ cols: 7, rows: 5, pieces: [{ c: 0, r: 2, img: B('cannon') }, { c: 2, r: 2, img: B('pawn'), faded: true }, { c: 6, r: 2, img: R('chariot') }],
        dots: [[3, 2], [4, 2], [5, 2]], captures: [[6, 2]], crosses: [[1, 2]],
        caption: '<b>包 Cannon</b> · 7 points<br>must jump exactly one piece; never jumps or takes a cannon' })}
      ${miniBoard({ cols: 5, rows: 5, pieces: [{ c: 2, r: 2, img: B('horse') }, { c: 2, r: 1, img: R('pawn'), faded: true }],
        dots: [[0, 1], [4, 1], [0, 3], [4, 3], [1, 4], [3, 4]], crosses: [[1, 0], [3, 0]],
        caption: '<b>馬 Horse</b> · 5 points<br>one straight, one diagonal; blocked by an adjacent piece' })}
      ${miniBoard({ cols: 7, rows: 7, cell: 28, pieces: [{ c: 3, r: 3, img: B('elephant') }, { c: 4, r: 3, img: R('pawn'), faded: true }],
        dots: [[1, 0], [5, 0], [0, 1], [0, 5], [1, 6], [5, 6]], crosses: [[6, 1], [6, 5]], paths: [[[3, 3], [3, 2], [1, 0]]],
        caption: '<b>象 Elephant</b> · 3 points<br>one straight, two diagonal; any piece in the way blocks it' })}
      ${miniBoard({ cols: 5, rows: 5, palace: [[1, 1]], pieces: [{ c: 2, r: 2, img: B('advisor') }],
        dots: [[1, 1], [2, 1], [3, 1], [1, 2], [3, 2], [1, 3], [2, 3], [3, 3]],
        caption: '<b>士 Guard</b> · 3 points / <b>楚 漢 General</b><br>one step along the palace lines' })}
      ${miniBoard({ cols: 5, rows: 4, palace: [[1, 0]], pieces: [{ c: 1, r: 2, img: B('pawn') }],
        dots: [[1, 1], [0, 2], [2, 2], [2, 1]], paths: [[[1, 2], [2, 1]]],
        caption: '<b>卒 Soldier</b> · 2 points<br>forward or sideways; diagonally forward in the enemy palace' })}
    </div>

    <h3>Passing and bikjang</h3>
    <div class="two-col">
      ${miniBoard({ cols: 9, rows: 10, cell: 20, palace: [[3, 0], [3, 7]], pieces: [{ c: 4, r: 1, img: R('king') }, { c: 4, r: 8, img: B('king') }],
        paths: [[[4, 2], [4, 7]]], caption: '<b>Bikjang</b>: generals face each other on an open file' })}
      <table class="page-table compact">
        <tbody>
          <tr><td>Pass</td><td>allowed any time except in check</td></tr>
          <tr><td>Bikjang on the board</td><td>the side to move must break it or pass</td></tr>
          <tr><td>Still standing a move later</td><td>game ends on points</td></tr>
        </tbody>
      </table>
    </div>

    <h3>How games end</h3>
    <div class="end-grid">
      <div class="end-card"><b>Checkmate</b><span>or no legal moves: the side to move loses</span></div>
      <div class="end-card"><b>Bikjang left standing</b><span>points</span></div>
      <div class="end-card"><b>Both sides pass</b><span>points</span></div>
      <div class="end-card"><b>100 plies, no capture</b><span>points</span></div>
      <div class="end-card"><b>Same position 3 times</b><span>points, unless one side checked every move: that side loses</span></div>
    </div>

    <h3>Points</h3>
    <svg viewBox="0 0 640 80" class="chart-svg" role="img" aria-label="Starting points">
      <text x="0" y="22" class="ch-label">Cho</text><g transform="translate(60,4)">${pointsBar('cho')}</g>
      <text x="${60 + 520 * (total / 75) + 8}" y="22" class="ch-val">${total}</text>
      <text x="0" y="62" class="ch-label">Han</text><g transform="translate(60,44)">${pointsBar('han')}</g>
      <text x="${60 + 520 * ((total + 1.5) / 75) + 8}" y="62" class="ch-val">${total} + 1.5</text>
    </svg>
    <p class="caption">Chariots, cannons, horses, elephants, guards, soldiers at the start. Han gets 1.5 for moving second,
      so Cho must lead by 2 to win on points.</p>
  </article>`;
}

// ---------- Progress ----------

interface Progress {
  ladder: { id: string; label: string; elo: number; kind: string }[];
  nets: { id: string; label: string; elo: number; visits: number; matches: { opponent: string; games: number; score: number; elo: number }[] }[];
  rl: { iteration: number; promoted: boolean; score: number | null; elo: number | null; low: number | null; high: number | null }[];
  selfplay: string[];
  curves: { label: string; points: { samples: number; top1: number; value_acc: number }[] }[];
  endings: Record<string, number>;
}

const SVG = 'http://www.w3.org/2000/svg';
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, text?: string): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

function ladderChart(data: Progress): SVGSVGElement {
  const rows = [
    ...data.ladder.filter((l) => l.kind !== 'fsf' || /nnue/.test(l.id)).map((l) => ({ ...l, sinsan: l.id.startsWith('oldnet') })),
    ...data.nets.map((n) => ({ id: n.id, label: `Sinsan: ${n.label}`, elo: n.elo, kind: 'net', sinsan: true })),
  ].sort((a, b) => b.elo - a.elo);
  const rowH = 22;
  const labelW = 300;
  const w = 760;
  const h = rows.length * rowH + 30;
  const lo = -2400;
  const hi = 2600;
  const x = (elo: number) => labelW + ((elo - lo) / (hi - lo)) * (w - labelW - 50);
  const s = svg('svg', { viewBox: `0 0 ${w} ${h}`, class: 'ladder-svg', role: 'img', 'aria-label': 'Strength ladder' });
  for (const tick of [-2000, -1000, 0, 1000, 2000]) {
    s.append(svg('line', { x1: x(tick), x2: x(tick), y1: 0, y2: h - 22, class: 'ladder-grid' }));
    s.append(svg('text', { x: x(tick), y: h - 6, class: 'ladder-tick', 'text-anchor': 'middle' }, String(tick)));
  }
  rows.forEach((r, i) => {
    const y = i * rowH + 4;
    s.append(svg('text', { x: labelW - 8, y: y + 14, class: `ladder-label${r.kind === 'net' ? ' ladder-new' : ''}`, 'text-anchor': 'end' }, r.label));
    s.append(svg('rect', {
      x: Math.min(x(0), x(r.elo)), y: y + 4, width: Math.max(2, Math.abs(x(r.elo) - x(0))), height: rowH - 8, rx: 2,
      class: r.kind === 'net' ? 'ladder-bar-new' : r.sinsan ? 'ladder-bar-old' : 'ladder-bar',
    }));
    s.append(svg('text', { x: x(Math.max(r.elo, 0)) + 6, y: y + 14, class: 'ladder-val' }, String(r.elo)));
  });
  return s;
}

function rlChart(data: Progress): SVGSVGElement {
  const w = 760;
  const h = 200;
  const pad = 36;
  const its = data.rl.filter((r) => r.elo !== null);
  const maxIt = Math.max(5, ...its.map((r) => r.iteration + 1));
  const lo = -250;
  const hi = 120;
  const x = (i: number) => pad + (i / maxIt) * (w - pad - 10);
  const y = (e: number) => 10 + ((hi - Math.max(lo, Math.min(hi, e))) / (hi - lo)) * (h - 40);
  const s = svg('svg', { viewBox: `0 0 ${w} ${h}`, class: 'rl-svg', role: 'img', 'aria-label': 'Self-play rounds' });
  for (const tick of [-200, -100, 0, 100]) {
    s.append(svg('line', { x1: pad, x2: w - 10, y1: y(tick), y2: y(tick), class: tick === 0 ? 'ladder-zero' : 'ladder-grid' }));
    s.append(svg('text', { x: pad - 6, y: y(tick) + 4, class: 'ladder-tick', 'text-anchor': 'end' }, tick > 0 ? `+${tick}` : String(tick)));
  }
  for (const r of its) {
    const cx = x(r.iteration + 0.5);
    s.append(svg('line', { x1: cx, x2: cx, y1: y(r.low!), y2: y(r.high!), class: 'rl-ci' }));
    const c = svg('circle', { cx, cy: y(r.elo!), r: 5, class: r.promoted ? 'rl-pt rl-promoted' : 'rl-pt' });
    c.append(svg('title', {}, `Round ${r.iteration}: ${r.score}% (${r.elo! > 0 ? '+' : ''}${r.elo} Elo, ${r.low} to ${r.high}) - ${r.promoted ? 'promoted' : 'rejected'}`));
    s.append(c);
    s.append(svg('text', { x: cx, y: h - 8, class: 'ladder-tick', 'text-anchor': 'middle' }, String(r.iteration)));
  }
  return s;
}

/** Validation top-1 agreement with the teacher against training samples seen. */
function curvesChart(data: Progress): SVGSVGElement {
  const w = 760;
  const h = 230;
  const pad = 44;
  const maxS = Math.max(...data.curves.flatMap((c) => c.points.map((p) => p.samples)), 1);
  const lo = 0.28;
  const hi = 0.48;
  const x = (v: number) => pad + (v / maxS) * (w - pad - 150);
  const y = (v: number) => 10 + ((hi - v) / (hi - lo)) * (h - 40);
  const s = svg('svg', { viewBox: `0 0 ${w} ${h}`, class: 'rl-svg', role: 'img', 'aria-label': 'Learning curves' });
  for (const t of [0.3, 0.35, 0.4, 0.45]) {
    s.append(svg('line', { x1: pad, x2: w - 150, y1: y(t), y2: y(t), class: 'ladder-grid' }));
    s.append(svg('text', { x: pad - 6, y: y(t) + 4, class: 'ladder-tick', 'text-anchor': 'end' }, `${Math.round(t * 100)}%`));
  }
  for (const m of [25e6, 50e6, 75e6, 100e6].filter((m) => m <= maxS)) {
    s.append(svg('text', { x: x(m), y: h - 8, class: 'ladder-tick', 'text-anchor': 'middle' }, `${m / 1e6}M`));
  }
  const cls = ['curve-a', 'curve-b', 'curve-c'];
  data.curves.forEach((c, i) => {
    s.append(svg('polyline', { points: c.points.map((p) => `${x(p.samples)},${y(p.top1)}`).join(' '), class: `curve ${cls[i % 3]}` }));
    const last = c.points[c.points.length - 1];
    if (last) s.append(svg('text', { x: x(last.samples) + 8, y: y(last.top1) + 4, class: `curve-label ${cls[i % 3]}` }, `${c.label} ${(last.top1 * 100).toFixed(1)}%`));
  });
  return s;
}

/** Progress data: live from the GPU server, else the snapshot shipped with the app
 * (`play_server.py --snapshot apps/web/public/progress.json`). */
export async function fetchProgress<T>(): Promise<T & { snapshot?: string }> {
  try {
    const r = await fetch('/api/progress');
    if (r.ok) return (await r.json()) as T & { snapshot?: string };
  } catch {
    // no server: fall through to the snapshot
  }
  const r = await fetch(`${import.meta.env.BASE_URL}progress.json`);
  if (!r.ok) throw new Error(`no server and no progress snapshot (${r.status})`);
  return (await r.json()) as T & { snapshot?: string };
}

export async function renderProgress(container: HTMLElement): Promise<void> {
  container.innerHTML = `<article class="page"><h2>Progress</h2><p class="lead">Loading…</p></article>`;
  let data: Progress & { snapshot?: string };
  try {
    data = await fetchProgress<Progress>();
  } catch (error) {
    container.innerHTML = `<article class="page"><h2>Progress</h2><p class="lead">Couldn't reach the server (${String(error)}).</p></article>`;
    return;
  }
  const promoted = data.rl.filter((r) => r.promoted).length;
  const best = [...data.nets].sort((a, b) => b.elo - a.elo)[0];
  const endTotal = Object.values(data.endings).reduce((a, b) => a + b, 0);
  const endNames: Record<string, string> = {
    checkmate: 'Checkmate', repetition: 'Repetition (points)', bikjang: 'Bikjang (points)', 'double-pass': 'Both passed (points)',
    'move-limit': 'Move limit (points)', 'perpetual-check': 'Perpetual check',
  };
  const endRows = Object.entries(data.endings).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ label: endNames[k] ?? k, value: Math.round((1000 * v) / Math.max(1, endTotal)) / 10, display: `${((100 * v) / Math.max(1, endTotal)).toFixed(0)}%` }));
  container.innerHTML = `
  <article class="page">
    <h2>Progress</h2>
    <p class="lead">${data.snapshot ? `Snapshot from ${data.snapshot} (the training server isn't reachable).` : 'Live from the training server.'} Elo against Fairy-Stockfish NNUE; 0 = 1,000 nodes per move. 200 Elo apart ≈ the stronger side wins 3 games in 4.</p>
    <div class="stat-row">
      <div class="stat"><b>${best ? best.elo : '—'}</b><span>best rating (800 visits)</span></div>
      <div class="stat"><b>${data.rl.length}</b><span>self-play rounds</span></div>
      <div class="stat"><b>${promoted}</b><span>promotions</span></div>
      <div class="stat"><b>50M</b><span>teacher positions</span></div>
    </div>
    <h3>Strength ladder</h3>
    <div class="chart ladder-chart"></div>
    <p class="page-note">Sinsan at 800 visits per move; each rating averages its 200-game matches. Rounds 2 and 9 had one match each (±50).</p>
    <h3>Learning curves</h3>
    <div class="chart curves-chart"></div>
    <p class="page-note">How often the network's top move matches the teacher's on held-out positions, against training samples seen.</p>
    <h3>Self-play rounds</h3>
    <div class="chart rl-chart"></div>
    <p class="page-note">New network vs current best, 400 games at 128 visits: Elo with 95% interval. Filled = promoted.</p>
    <h3>How games end</h3>
    <div class="chart">${barChart(endRows, { max: Math.max(10, ...endRows.map((r) => r.value)) * 1.1, width: 760 })}</div>
    <p class="page-note">${endTotal.toLocaleString()} games from the promotion matches.</p>
    <h3>Self-play now</h3>
    <pre class="page-pre">${data.selfplay.join('\n') || 'No self-play running.'}</pre>
  </article>`;
  container.querySelector('.ladder-chart')!.append(ladderChart(data));
  container.querySelector('.curves-chart')!.append(curvesChart(data));
  container.querySelector('.rl-chart')!.append(rlChart(data));
}
