import { applyMove, findGeneral, pieceAt } from '@sinsan/rules';
import type { GameResult, Move, Position, Side } from '@sinsan/rules';
import type { BoardArrow } from '@sinsan/ui';

/** What training/play_server.py returns for one Sinsan search. Each q is from the point of view of
 * the side that played that move (-1 = certain loss, +1 = certain win). */
export interface ServerAnalysis {
  move: string;
  best: string;
  cp: number;
  value: number;
  net_value: number | null;
  visits: number;
  ms: number;
  depth: number;
  seldepth: number;
  pv: string[];
  legal: number;
  candidates: { move: string; visits: number; q: number | null; prior: number }[];
  tree: [number, number, string, number, number | null, number, boolean][];
}

/** What /api/fsf returns: Fairy-Stockfish's move and its own evaluation (side to move). */
export interface FsfInfo {
  move: string;
  cp: number | null;
  mate: number | null;
  depth: number;
  nodes: number;
  pv: string[];
  ms: number;
}

interface TreeNode {
  move: string;
  label: string;
  depth: number;
  visits: number;
  /** Win probability (0..1) for the side that searched, or null if never evaluated. */
  win: number | null;
  prior: number;
  terminal: boolean;
  mover: Side;
  parent: TreeNode | null;
  children: TreeNode[];
  x: number;
  y: number;
}

/** One Sinsan search, made by `side` before it played `analysis.move` at `ply`. */
interface Entry {
  ply: number;
  side: Side;
  before: Position;
  analysis: ServerAnalysis;
}

type MoveClass = 'best' | 'excellent' | 'good' | 'inaccuracy' | 'mistake' | 'blunder';

/** A move in the game record. A move is rated when Sinsan searched for the opponent both before
 * and after it (you vs Sinsan: your moves; Fairy-Stockfish vs Sinsan: Fairy-Stockfish's moves). */
interface Played {
  ply: number;
  side: Side;
  uci: string;
  label: string;
  entry?: number; // Sinsan's moves: index into entries
  fsf?: FsfInfo;
  /** The mover's win chance (0..1) before the move, from the opponent's preceding search. */
  before?: number;
  expected?: string; // the reply that search expected
  expectedLabel?: string;
  loss?: number;
  cls?: MoveClass;
}

export interface AnalysisPanel {
  reset(names: Record<Side, string>, viewer: Side, humans: readonly Side[]): void;
  addAnalysis(ply: number, before: Position, analysis: ServerAnalysis): void;
  noteMove(ply: number, before: Position, move: Move, fsf?: FsfInfo): void;
  gameOver(result: GameResult): void;
  showHint(kind: 'hint' | 'threat', before: Position, analysis: ServerAnalysis): void;
  setWaiting(waiting: boolean, what?: string): void;
}

export interface PanelCallbacks {
  onArrows(arrows: BoardArrow[]): void;
  /** Win chance (0..1) of the side at the bottom of the board, or null when unknown. */
  onEval(viewerWin: number | null): void;
}

const GLYPHS: Record<string, [string, string]> = {
  chariot: ['車', '車'],
  cannon: ['包', '包'],
  horse: ['馬', '馬'],
  elephant: ['象', '象'],
  guard: ['士', '士'],
  general: ['楚', '漢'],
  soldier: ['卒', '兵'],
};

const CLASS_INFO: Record<MoveClass, { label: string; symbol: string }> = {
  best: { label: 'Best', symbol: '★' },
  excellent: { label: 'Excellent', symbol: '!' },
  good: { label: 'Good', symbol: '✓' },
  inaccuracy: { label: 'Inaccuracy', symbol: '?!' },
  mistake: { label: 'Mistake', symbol: '?' },
  blunder: { label: 'Blunder', symbol: '??' },
};

function classify(loss: number, matchedExpectation: boolean): MoveClass {
  if (matchedExpectation || loss < 0.005) return 'best';
  if (loss < 0.02) return 'excellent';
  if (loss < 0.05) return 'good';
  if (loss < 0.1) return 'inaccuracy';
  if (loss < 0.2) return 'mistake';
  return 'blunder';
}

/** Lichess's per-move accuracy from the win-percentage drop (0..100 scale). */
function moveAccuracy(loss: number): number {
  return Math.max(0, Math.min(100, 103.1668 * Math.exp(-0.04354 * loss * 100) - 3.1669));
}

/** Fairy-Stockfish centipawns to a win chance, with the scale fitted on its own games (sigmoid(cp/105)). */
function cpToWin(cp: number | null, mate: number | null): number | null {
  if (mate !== null) return mate > 0 ? 1 : 0;
  if (cp === null) return null;
  return 1 / (1 + Math.exp(-cp / 105));
}

function parseSquare(text: string): number {
  return (10 - Number(text.slice(1))) * 9 + (text.charCodeAt(0) - 97);
}

export function parseUci(uci: string): Move {
  const split = [...uci].findIndex((ch, i) => i > 0 && /[a-i]/.test(ch));
  const from = parseSquare(uci.slice(0, split));
  const to = parseSquare(uci.slice(split));
  return { from, to, isPass: from === to };
}

function squareName(square: number): string {
  return String.fromCharCode(97 + (square % 9)) + String(10 - Math.floor(square / 9));
}

/** Fairy-Stockfish notation; a pass is written as the mover's general square twice. */
export function moveToUci(move: Move, position: Position): string {
  if (move.isPass) {
    const g = squareName(findGeneral(position, position.sideToMove));
    return g + g;
  }
  return squareName(move.from) + squareName(move.to);
}

function moveLabel(position: Position, uci: string): string {
  const move = parseUci(uci);
  if (move.isPass) return 'pass';
  const piece = pieceAt(position, move.from);
  const glyph = piece ? GLYPHS[piece.type]![piece.side === 'cho' ? 0 : 1] : '?';
  return `${glyph}${squareName(move.to)}`;
}

/** Labels for a line of moves played from `position` (stops at the first move that doesn't fit). */
function lineLabels(position: Position, ucis: string[]): string[] {
  const out: string[] = [];
  let pos = position;
  for (const u of ucis) {
    const m = parseUci(u);
    if (!m.isPass && !pieceAt(pos, m.from)) break;
    out.push(moveLabel(pos, u));
    pos = applyMove(pos, m);
  }
  return out;
}

function pct(p: number | null): string {
  return p === null ? '—' : `${(p * 100).toFixed(0)}%`;
}

/** Red (bad for the searching side) through amber to green (good). */
function winColor(p: number | null): string {
  if (p === null) return 'var(--panel-muted)';
  const hue = Math.max(0, Math.min(120, p * 120));
  return `hsl(${hue.toFixed(0)} 60% 45%)`;
}

const other = (side: Side): Side => (side === 'cho' ? 'han' : 'cho');
const SIDE_NAME: Record<Side, string> = { cho: 'Cho', han: 'Han' };
const SIDE_BRUSH: Record<Side, string> = { cho: 'blue', han: 'red' };
const PALE_BRUSH: Record<Side, string> = { cho: 'paleBlue', han: 'paleRed' };

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const SVG = 'http://www.w3.org/2000/svg';
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

export function createAnalysisPanel(container: HTMLElement, callbacks: PanelCallbacks): AnalysisPanel {
  container.classList.add('analysis-panel');
  container.innerHTML = `
    <section class="ap-eval">
      <div class="ap-eval-sides"><span class="ap-side ap-side-cho"></span><span class="ap-side ap-side-han"></span></div>
      <div class="ap-bar"><div class="ap-bar-fill"></div><div class="ap-bar-net" title="Network's first instinct, before search"></div></div>
      <div class="ap-sub ap-eval-detail">Waiting for Sinsan's first search.</div>
      <div class="ap-sub ap-stats"></div>
    </section>
    <section class="ap-hint-box" hidden>
      <div class="ap-title-row"><span class="ap-title ap-hint-title"></span><button class="ap-close" title="Close">×</button></div>
      <div class="ap-hint-text"></div>
    </section>
    <section>
      <div class="ap-title-row"><span class="ap-title">Engine lines</span><span class="ap-sub ap-lines-who"></span></div>
      <div class="ap-lines"></div>
    </section>
    <section>
      <div class="ap-title-row"><span class="ap-title">Candidate moves</span><span class="ap-sub">hover to show</span></div>
      <table class="ap-cands"><thead><tr><th>Move</th><th>Visits</th><th>Win</th><th>Prior</th></tr></thead><tbody></tbody></table>
    </section>
    <section>
      <div class="ap-title-row"><span class="ap-title">Decision tree</span>
        <select class="ap-detail"><option value="compact">Compact</option><option value="detailed">Detailed</option></select></div>
      <div class="ap-tree"></div>
      <div class="ap-sub ap-line">Hover a node to see that line on the board.</div>
    </section>
    <section>
      <div class="ap-title-row"><span class="ap-title">Win chance over the game</span><span class="ap-sub ap-graph-legend"></span></div>
      <div class="ap-graph"></div>
    </section>
    <section>
      <div class="ap-title-row"><span class="ap-title">Moves</span><span class="ap-sub ap-accuracy"></span></div>
      <div class="ap-moves"></div>
    </section>
    <section class="ap-review" hidden>
      <div class="ap-title">Game review</div>
      <div class="ap-review-body"></div>
    </section>
  `;
  const q = <T extends Element>(s: string) => container.querySelector<T>(s)!;
  const sideCho = q<HTMLSpanElement>('.ap-side-cho');
  const sideHan = q<HTMLSpanElement>('.ap-side-han');
  const barFill = q<HTMLDivElement>('.ap-bar-fill');
  const barNet = q<HTMLDivElement>('.ap-bar-net');
  const evalDetail = q<HTMLDivElement>('.ap-eval-detail');
  const stats = q<HTMLDivElement>('.ap-stats');
  const hintBox = q<HTMLElement>('.ap-hint-box');
  const hintTitle = q<HTMLSpanElement>('.ap-hint-title');
  const hintText = q<HTMLDivElement>('.ap-hint-text');
  const linesBox = q<HTMLDivElement>('.ap-lines');
  const linesWho = q<HTMLSpanElement>('.ap-lines-who');
  const graph = q<HTMLDivElement>('.ap-graph');
  const graphLegend = q<HTMLSpanElement>('.ap-graph-legend');
  const candBody = q<HTMLTableSectionElement>('.ap-cands tbody');
  const treeBox = q<HTMLDivElement>('.ap-tree');
  const lineBox = q<HTMLDivElement>('.ap-line');
  const detailSelect = q<HTMLSelectElement>('.ap-detail');
  const movesBox = q<HTMLDivElement>('.ap-moves');
  const accuracyEl = q<HTMLSpanElement>('.ap-accuracy');
  const reviewBox = q<HTMLElement>('.ap-review');
  const reviewBody = q<HTMLDivElement>('.ap-review-body');

  let entries: Entry[] = [];
  let played: Played[] = [];
  let selected = -1; // index into entries
  let names: Record<Side, string> = { cho: 'Cho', han: 'Han' };
  let viewer: Side = 'cho';
  let humans: readonly Side[] = [];
  let live = true; // the selected entry is the latest move: arrows match the board
  let hintArrows: BoardArrow[] | null = null;

  const who = (side: Side) => names[side];

  function showArrows(arrows: BoardArrow[]): void {
    callbacks.onArrows(hintArrows ?? (live ? arrows : []));
  }

  /** The searching side's win chance for the move it played (its Q), else the root value. */
  function chosenWin(entry: Entry): number {
    const c = entry.analysis.candidates.find((x) => x.move === entry.analysis.move);
    return c && c.q !== null ? (c.q + 1) / 2 : (entry.analysis.value + 1) / 2;
  }

  /** Win chance for `side` according to an entry's search. */
  function winFor(entry: Entry, side: Side): number {
    const w = (entry.analysis.value + 1) / 2;
    return entry.side === side ? w : 1 - w;
  }

  function buildTree(entry: Entry, detailed: boolean): TreeNode {
    const { analysis, before, side } = entry;
    const total = Math.max(1, analysis.visits);
    const minShare = detailed ? 0.004 : 0.015;
    const root: TreeNode = {
      move: '', label: 'now', depth: 0, visits: total, win: (analysis.value + 1) / 2, prior: 1, terminal: false,
      mover: other(side), parent: null, children: [], x: 0, y: 0,
    };
    const nodes: (TreeNode | null)[] = [];
    const positions: Position[] = [];
    for (const [parentRow, depth, move, visits, qv, prior, terminal] of analysis.tree) {
      const parent = parentRow < 0 ? root : nodes[parentRow];
      const parentPos = parentRow < 0 ? before : positions[parentRow];
      const limit = depth === 1 ? (detailed ? 5 : 3) : detailed ? 3 : 2;
      if (!parent || !parentPos || visits / total < minShare || depth > (detailed ? 6 : 5) || parent.children.length >= limit) {
        nodes.push(null);
        positions.push(before);
        continue;
      }
      const mover: Side = depth % 2 === 1 ? side : other(side);
      const winForMover = qv === null ? null : (qv + 1) / 2;
      const node: TreeNode = {
        move, label: moveLabel(parentPos, move), depth, visits,
        win: winForMover === null ? null : mover === side ? winForMover : 1 - winForMover,
        prior, terminal, mover, parent, children: [], x: 0, y: 0,
      };
      parent.children.push(node);
      nodes.push(node);
      positions.push(applyMove(parentPos, parseUci(move)));
    }
    return root;
  }

  function linePath(node: TreeNode): TreeNode[] {
    const path: TreeNode[] = [];
    for (let n: TreeNode | null = node; n && n.parent; n = n.parent) path.unshift(n);
    return path;
  }

  function arrowsFor(ucis: string[], firstMover: Side, pale = false): BoardArrow[] {
    const out: BoardArrow[] = [];
    ucis.forEach((u, i) => {
      const m = parseUci(u);
      if (m.isPass) return;
      const mover = i % 2 === 0 ? firstMover : other(firstMover);
      out.push({ from: m.from, to: m.to, brush: (pale ? PALE_BRUSH : SIDE_BRUSH)[mover], lineWidth: Math.max(4, 12 - i * 2) });
    });
    return out;
  }

  function renderTree(entry: Entry): void {
    const root = buildTree(entry, detailSelect.value === 'detailed');
    const colW = 66;
    const rowH = 24;
    let leaf = 0;
    let maxDepth = 0;
    const all: TreeNode[] = [];
    const place = (n: TreeNode): void => {
      all.push(n);
      maxDepth = Math.max(maxDepth, n.depth);
      n.x = 14 + n.depth * colW;
      if (n.children.length === 0) {
        n.y = 14 + leaf++ * rowH;
      } else {
        n.children.forEach(place);
        n.y = (n.children[0]!.y + n.children[n.children.length - 1]!.y) / 2;
      }
    };
    place(root);
    const width = 14 + maxDepth * colW + 70;
    const height = Math.max(40, leaf * rowH + 8);
    const s = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, class: 'ap-tree-svg' });
    const total = root.visits;
    for (const n of all) {
      for (const c of n.children) {
        const w = 1 + 7 * Math.sqrt(c.visits / total);
        const midX = (n.x + c.x) / 2;
        s.appendChild(svg('path', {
          d: `M${n.x},${n.y} C${midX},${n.y} ${midX},${c.y} ${c.x},${c.y}`,
          class: 'ap-edge', 'stroke-width': w.toFixed(1),
        }));
      }
    }
    for (const n of all) {
      const g = svg('g', { class: `ap-node ${n.depth === 0 ? 'ap-root' : `ap-${n.mover === entry.side ? 'own' : 'opp'}`}` });
      const r = n.depth === 0 ? 6 : 3 + 6 * Math.sqrt(n.visits / total);
      g.appendChild(svg('circle', { cx: n.x, cy: n.y, r: r.toFixed(1), fill: winColor(n.win) }));
      const t = svg('text', { x: n.x + r + 3, y: n.y - 4, class: 'ap-node-label' });
      t.textContent = n.label;
      g.appendChild(t);
      if (n.depth > 0) {
        const v = svg('text', { x: n.x + r + 3, y: n.y + 9, class: 'ap-node-sub' });
        v.textContent = `${n.visits} · ${pct(n.win)}`;
        g.appendChild(v);
      }
      const title = svg('title', {});
      title.textContent = n.depth === 0 ? 'Position searched' :
        `${who(n.mover)}: ${n.label} (${n.move})\n${n.visits} visits, prior ${pct(n.prior)}\n` +
        `${who(entry.side)} win ${pct(n.win)}${n.terminal ? '\ngame ends here' : ''}`;
      g.appendChild(title);
      if (n.depth > 0) {
        g.addEventListener('mouseenter', () => {
          const path = linePath(n);
          showArrows(arrowsFor(path.map((p) => p.move), entry.side));
          lineBox.textContent = path.map((p, i) => `${i + 1}. ${who(p.mover)} ${p.label}`).join('  ') +
            `  → ${who(entry.side)} ${pct(n.win)}`;
        });
        g.addEventListener('mouseleave', () => {
          showArrows(defaultArrows(entry));
          lineBox.textContent = 'Hover a node to see that line on the board.';
        });
      }
      s.appendChild(g);
    }
    treeBox.replaceChildren(s);
  }

  /** On the board after a Sinsan move: the reply it expects (and, faintly, its follow-up). */
  function defaultArrows(entry: Entry): BoardArrow[] {
    return arrowsFor(entry.analysis.pv.slice(1, 3), other(entry.side), true);
  }

  /** Top candidates, each continued by the most-visited child at every ply (the tree's MultiPV). */
  function renderLines(entry: Entry): void {
    const { analysis, before } = entry;
    linesWho.textContent = `${who(entry.side)}'s top 3, best play`;
    const rows = analysis.tree;
    const childrenOf = new Map<number, number[]>();
    rows.forEach((r, i) => childrenOf.set(r[0], [...(childrenOf.get(r[0]) ?? []), i]));
    const tops = (childrenOf.get(-1) ?? []).slice(0, 3);
    linesBox.replaceChildren(
      ...tops.map((rootRow) => {
        const line: string[] = [rows[rootRow]![2]];
        let at = rootRow;
        for (;;) {
          const kids = childrenOf.get(at);
          if (!kids || kids.length === 0) break;
          at = kids[0]!; // dump order is most-visited first
          line.push(rows[at]![2]);
        }
        const r = rows[rootRow]!;
        const win = r[4] === null ? null : (r[4] + 1) / 2;
        const div = el('div', `ap-pvline${r[2] === analysis.move ? ' ap-chosen' : ''}`);
        const badge = el('span', 'ap-pv-eval', pct(win));
        badge.style.background = winColor(win);
        div.append(badge, el('span', 'ap-pv-moves', lineLabels(before, line).join('  ')));
        div.title = `${r[3]} visits`;
        div.addEventListener('mouseenter', () => showArrows(arrowsFor(line.slice(0, 6), entry.side)));
        div.addEventListener('mouseleave', () => showArrows(defaultArrows(entry)));
        return div;
      }),
    );
  }

  function renderCandidates(entry: Entry): void {
    const { analysis, before } = entry;
    const total = Math.max(1, analysis.candidates.reduce((a, c) => a + c.visits, 0));
    candBody.replaceChildren(
      ...analysis.candidates.filter((c) => c.visits > 0 || c.prior > 0.02).slice(0, 8).map((c) => {
        const tr = el('tr', c.move === analysis.move ? 'ap-chosen' : '');
        const win = c.q === null ? null : (c.q + 1) / 2;
        const tag = c.move === analysis.move ? (c.move === analysis.best ? ' ✓' : ' ✓ (sampled)') : '';
        tr.append(
          el('td', 'ap-move', `${moveLabel(before, c.move)}${tag}`),
          (() => {
            const td = el('td', 'ap-visits');
            const bar = el('div', 'ap-vbar');
            bar.style.width = `${(100 * c.visits) / total}%`;
            td.append(bar, el('span', '', String(c.visits)));
            return td;
          })(),
          (() => {
            const td = el('td', '', pct(win));
            td.style.color = winColor(win);
            return td;
          })(),
          el('td', 'ap-prior', pct(c.prior)),
        );
        tr.addEventListener('mouseenter', () => showArrows(arrowsFor([c.move], entry.side)));
        tr.addEventListener('mouseleave', () => showArrows(defaultArrows(entry)));
        return tr;
      }),
    );
  }

  /** Viewer's win chance by Sinsan's searches, plus Fairy-Stockfish's own evaluations when it plays. */
  function renderGraph(): void {
    const w = 400;
    const h = 96;
    const s = svg('svg', { width: '100%', viewBox: `0 0 ${w} ${h}`, class: 'ap-graph-svg' });
    s.appendChild(svg('line', { x1: 0, x2: w, y1: h / 2, y2: h / 2, class: 'ap-mid' }));
    const lastPly = played.length ? played[played.length - 1]!.ply : 0;
    const maxPly = Math.max(30, lastPly + 2);
    const px = (ply: number) => (ply / maxPly) * (w - 10) + 5;
    const py = (win: number) => (1 - win) * (h - 10) + 5;
    const fsfPts = played.filter((p) => p.fsf && cpToWin(p.fsf.cp, p.fsf.mate) !== null).map((p) => {
      const win = cpToWin(p.fsf!.cp, p.fsf!.mate)!;
      return [px(p.ply), py(p.side === viewer ? win : 1 - win)] as const;
    });
    if (fsfPts.length > 1) {
      s.appendChild(svg('polyline', { points: fsfPts.map(([x, y]) => `${x},${y}`).join(' '), class: 'ap-curve-fsf' }));
    }
    if (entries.length > 0) {
      const pts = entries.map((e) => [px(e.ply), py(winFor(e, viewer))] as const);
      const area = `M${pts[0]![0]},${h / 2} ` + pts.map(([x, y]) => `L${x},${y}`).join(' ') + ` L${pts[pts.length - 1]![0]},${h / 2} Z`;
      s.appendChild(svg('path', { d: area, class: 'ap-area' }));
      s.appendChild(svg('polyline', { points: pts.map(([x, y]) => `${x},${y}`).join(' '), class: 'ap-curve' }));
      for (const p of played) {
        if (p.cls !== 'mistake' && p.cls !== 'blunder' && p.cls !== 'inaccuracy') continue;
        const mark = svg('line', { x1: px(p.ply), x2: px(p.ply), y1: 4, y2: h - 4, class: `ap-mark ap-mark-${p.cls}` });
        const t = svg('title', {});
        t.textContent = `Move ${p.ply + 1}: ${who(p.side)} ${p.label} - ${CLASS_INFO[p.cls].label}`;
        mark.appendChild(t);
        s.appendChild(mark);
      }
      pts.forEach(([x, y], i) => {
        const c = svg('circle', { cx: x, cy: y, r: i === selected ? 4.5 : 2.8, class: i === selected ? 'ap-pt ap-pt-sel' : 'ap-pt' });
        const t = svg('title', {});
        const e = entries[i]!;
        t.textContent = `Move ${e.ply + 1}: ${who(e.side)} ${moveLabel(e.before, e.analysis.move)}, ${who(viewer)} ${pct(winFor(e, viewer))} (click to inspect)`;
        c.appendChild(t);
        c.addEventListener('click', () => select(i));
        s.appendChild(c);
      });
    }
    graphLegend.textContent = `${who(viewer) === 'You' ? 'Your' : `${who(viewer)}'s`} chances${fsfPts.length > 1 ? ' · dashed: Fairy-Stockfish' : ''}`;
    graph.replaceChildren(s);
  }

  function renderMoves(): void {
    const rows: HTMLElement[] = [];
    for (const p of played) {
      const div = el('div', `ap-mv ap-mv-${p.side}`);
      div.append(el('span', 'ap-mv-num', `${p.ply + 1}.`), el('span', 'ap-mv-who', who(p.side)), el('span', 'ap-mv-label', p.label));
      const info = el('span', 'ap-mv-info');
      if (p.cls) {
        const badge = el('span', `ap-cls ap-cls-${p.cls}`, `${CLASS_INFO[p.cls].symbol} ${CLASS_INFO[p.cls].label}`);
        badge.title = p.expectedLabel && p.expected !== p.uci ? `Sinsan expected ${p.expectedLabel}` : 'The expected move';
        div.append(badge);
        if (p.loss !== undefined && p.loss >= 0.005) info.append(`−${(p.loss * 100).toFixed(0)}% `);
        if (p.cls !== 'best' && p.expectedLabel && p.expected !== p.uci) info.append(`best ${p.expectedLabel} `);
      } else if (p.before !== undefined && p.loss === undefined) {
        div.append(el('span', 'ap-mv-pending', '…'));
      } else {
        div.append(el('span', ''));
      }
      if (p.entry !== undefined) {
        const e = entries[p.entry]!;
        info.append(`${pct(chosenWin(e))} · ${e.analysis.visits.toLocaleString()} visits`);
        div.classList.add('ap-mv-click');
        if (p.entry === selected) div.classList.add('ap-mv-sel');
        div.addEventListener('click', () => select(p.entry!));
      } else if (p.fsf) {
        const win = cpToWin(p.fsf.cp, p.fsf.mate);
        const score = p.fsf.mate !== null ? `mate ${p.fsf.mate}` : p.fsf.cp !== null ? `${p.fsf.cp > 0 ? '+' : ''}${(p.fsf.cp / 100).toFixed(2)}` : '';
        info.append(`depth ${p.fsf.depth} · ${score} (${pct(win)})`);
      }
      div.append(info);
      rows.push(div);
    }
    movesBox.replaceChildren(...(rows.length ? rows : [el('div', 'ap-sub', 'No moves yet.')]));
    movesBox.scrollTop = movesBox.scrollHeight;
    const parts: string[] = [];
    for (const side of ['cho', 'han'] as const) {
      const rated = played.filter((p) => p.side === side && p.loss !== undefined);
      if (rated.length) parts.push(`${who(side)} ${(rated.reduce((a, p) => a + moveAccuracy(p.loss!), 0) / rated.length).toFixed(1)}%`);
    }
    accuracyEl.textContent = parts.length ? `accuracy: ${parts.join(' · ')}` : '';
  }

  function renderEvalHeader(choWin: number | null, netChoWin: number | null): void {
    sideCho.textContent = `${SIDE_NAME.cho} · ${who('cho')}  ${pct(choWin)}`;
    sideHan.textContent = `${pct(choWin === null ? null : 1 - choWin)}  ${who('han')} · ${SIDE_NAME.han}`;
    barFill.style.width = `${(choWin ?? 0.5) * 100}%`;
    barNet.style.left = netChoWin === null ? '-10px' : `calc(${netChoWin * 100}% - 1px)`;
  }

  function select(i: number): void {
    selected = i;
    const entry = entries[i]!;
    live = played[played.length - 1]?.entry === i;
    const { analysis } = entry;
    const choWin = winFor(entry, 'cho');
    const netWin = analysis.net_value === null ? null : (analysis.net_value + 1) / 2;
    renderEvalHeader(choWin, netWin === null ? null : entry.side === 'cho' ? netWin : 1 - netWin);
    evalDetail.textContent =
      `${i === entries.length - 1 ? 'Latest search' : `Viewing move ${entry.ply + 1}`} by ${who(entry.side)} - ` +
      (netWin === null ? '' : `first instinct ${pct(netWin)}, after search `) + `${pct((analysis.value + 1) / 2)} for ${who(entry.side)}`;
    stats.textContent = `${analysis.visits.toLocaleString()} visits · depth ${analysis.depth} (max ${analysis.seldepth}) · ` +
      `${(analysis.ms / 1000).toFixed(1)}s · ${Math.round(analysis.visits / Math.max(0.001, analysis.ms / 1000)).toLocaleString()}/s · ` +
      `${analysis.legal} legal`;
    renderLines(entry);
    renderCandidates(entry);
    renderTree(entry);
    renderGraph();
    renderMoves();
    showArrows(defaultArrows(entry));
  }

  /** Rate the move just before `ply` if the opponent searched both before and after it. */
  function rateMoveBefore(ply: number, moverWinAfter: number): void {
    const p = played.find((x) => x.ply === ply - 1);
    if (!p || p.before === undefined || p.loss !== undefined) return;
    p.loss = Math.max(0, p.before - moverWinAfter);
    p.cls = classify(p.loss, p.expected === p.uci);
  }

  /** Common bookkeeping for every move: the mover's win chance before it, if the opponent searched just before. */
  function record(ply: number, before: Position, uci: string, extra: Partial<Played>): void {
    const side = before.sideToMove;
    const prev = entries[entries.length - 1];
    const fresh = prev && prev.ply === ply - 1 && prev.side !== side;
    played.push({
      ply, side, uci, label: moveLabel(before, uci),
      before: fresh ? 1 - chosenWin(prev) : undefined,
      expected: fresh ? prev.analysis.pv[1] : undefined,
      expectedLabel: fresh && prev.analysis.pv[1] ? lineLabels(before, [prev.analysis.pv[1]])[0] : undefined,
      ...extra,
    });
  }

  function renderReview(result: GameResult): void {
    const counts = new Map<string, number>();
    for (const p of played) if (p.cls) counts.set(`${p.side}:${p.cls}`, (counts.get(`${p.side}:${p.cls}`) ?? 0) + 1);
    const ratedSides = (['cho', 'han'] as const).filter((side) => played.some((p) => p.side === side && p.cls));
    const winner = result.winner === null ? 'Draw' : `${who(result.winner)} (${SIDE_NAME[result.winner]}) won`;
    const table = el('table', 'ap-review-table');
    const head = el('tr');
    head.append(el('th', '', ''), ...ratedSides.map((s) => el('th', '', who(s))));
    table.append(head);
    for (const cls of Object.keys(CLASS_INFO) as MoveClass[]) {
      const tr = el('tr');
      tr.append(el('td', '', ''));
      tr.firstChild!.appendChild(el('span', `ap-cls ap-cls-${cls}`, `${CLASS_INFO[cls].symbol} ${CLASS_INFO[cls].label}`));
      tr.append(...ratedSides.map((s) => el('td', '', String(counts.get(`${s}:${cls}`) ?? 0))));
      table.append(tr);
    }
    const accRow = el('tr');
    accRow.append(el('td', 'ap-review-acc', 'Accuracy'), ...ratedSides.map((s) => {
      const rated = played.filter((p) => p.side === s && p.loss !== undefined);
      return el('td', 'ap-review-acc', `${(rated.reduce((a, p) => a + moveAccuracy(p.loss!), 0) / rated.length).toFixed(1)}%`);
    }));
    table.append(accRow);
    const worst = [...played].filter((p) => p.loss !== undefined).sort((a, b) => b.loss! - a.loss!)[0];
    reviewBody.replaceChildren(
      el('div', 'ap-review-head', `${winner} · ${result.kind}`),
      ...(ratedSides.length ? [table] : [el('div', 'ap-sub', 'No moves were rated (moves are rated by the opponent Sinsan\'s searches).')]),
      el('div', 'ap-sub', worst && worst.loss! >= 0.05
        ? `Biggest swing: move ${worst.ply + 1}, ${who(worst.side)} ${worst.label} (−${(worst.loss! * 100).toFixed(0)}%); best was ${worst.expectedLabel ?? '—'}.`
        : 'No big swings.'),
    );
    reviewBox.hidden = false;
  }

  detailSelect.addEventListener('change', () => {
    if (selected >= 0) renderTree(entries[selected]!);
  });
  q<HTMLButtonElement>('.ap-close').addEventListener('click', () => {
    hintBox.hidden = true;
    hintArrows = null;
    showArrows(selected >= 0 ? defaultArrows(entries[selected]!) : []);
  });

  function clearHint(): void {
    hintBox.hidden = true;
    hintArrows = null;
  }

  return {
    reset(newNames, newViewer, newHumans) {
      names = newNames;
      viewer = newViewer;
      humans = newHumans;
      entries = [];
      played = [];
      selected = -1;
      live = true;
      clearHint();
      renderEvalHeader(null, null);
      evalDetail.textContent = 'Waiting for the first search.';
      stats.textContent = '';
      linesWho.textContent = '';
      linesBox.replaceChildren();
      candBody.replaceChildren();
      treeBox.replaceChildren();
      reviewBox.hidden = true;
      renderGraph();
      renderMoves();
      callbacks.onArrows([]);
      callbacks.onEval(null);
    },
    addAnalysis(ply, before, analysis) {
      clearHint();
      const side = before.sideToMove;
      rateMoveBefore(ply, 1 - (analysis.value + 1) / 2);
      record(ply, before, analysis.move, { entry: entries.length }); // before pushing: compares with the opponent's search
      entries.push({ ply, side, before, analysis });
      select(entries.length - 1);
      const e = entries[entries.length - 1]!;
      callbacks.onEval(winFor(e, viewer));
    },
    noteMove(ply, before, move, fsf) {
      clearHint();
      record(ply, before, moveToUci(move, before), fsf ? { fsf } : {});
      live = false; // the board has moved on; arrows from the last search would point at the wrong position
      if (fsf) {
        const win = cpToWin(fsf.cp, fsf.mate);
        if (win !== null) callbacks.onEval(before.sideToMove === viewer ? win : 1 - win);
        if (selected >= 0) renderGraph();
      }
      renderMoves();
      callbacks.onArrows([]);
    },
    gameOver(result) {
      const last = played[played.length - 1];
      if (last) rateMoveBefore(last.ply + 1, result.winner === null ? 0.5 : result.winner === last.side ? 1 : 0);
      renderGraph();
      renderMoves();
      renderReview(result);
      callbacks.onEval(result.winner === null ? 0.5 : result.winner === viewer ? 1 : 0);
    },
    showHint(kind, before, analysis) {
      const searcher = before.sideToMove;
      const human = humans.includes(searcher) ? searcher : other(searcher);
      const win = (analysis.value + 1) / 2;
      const humanWin = searcher === human ? win : 1 - win;
      hintTitle.textContent = kind === 'hint' ? 'Hint: Sinsan would play this for you' : 'Threat: what your opponent could do if you passed';
      hintText.textContent = `${lineLabels(before, analysis.pv.slice(0, 6)).join('  ')}  · your chances ${pct(humanWin)} · ${analysis.visits.toLocaleString()} visits`;
      hintBox.hidden = false;
      hintArrows = arrowsFor(analysis.pv.slice(0, kind === 'hint' ? 3 : 2), searcher);
      callbacks.onArrows(hintArrows);
    },
    setWaiting(waiting, what) {
      container.classList.toggle('ap-waiting', waiting);
      if (waiting && what) stats.textContent = what;
    },
  };
}
