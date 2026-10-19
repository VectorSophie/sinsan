import { applyMove, findGeneral, pieceAt } from '@sinsan/rules';
import type { GameResult, Move, Position, Side } from '@sinsan/rules';
import type { BoardArrow } from '@sinsan/ui';

/** What training/play_server.py returns for one search. Each q is from the point of view of the
 * side that played that move (-1 = certain loss, +1 = certain win). */
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

interface TreeNode {
  move: string;
  label: string;
  depth: number;
  visits: number;
  /** Sinsan's win probability (0..1) after this line, or null if never evaluated. */
  win: number | null;
  prior: number;
  terminal: boolean;
  mover: Side;
  parent: TreeNode | null;
  children: TreeNode[];
  x: number;
  y: number;
}

/** One search by Sinsan, made before it played `analysis.move` at `ply`. */
interface Entry {
  ply: number;
  before: Position;
  analysis: ServerAnalysis;
}

type MoveClass = 'best' | 'excellent' | 'good' | 'inaccuracy' | 'mistake' | 'blunder';

/** A move in the game record. Your moves are classified once Sinsan has searched the reply. */
interface Played {
  ply: number;
  side: Side;
  uci: string;
  label: string;
  entry?: number; // Sinsan's moves: index into entries
  /** Your moves: your win chance (0..1) before the move, as Sinsan saw it after its own move. */
  before?: number;
  expected?: string; // the reply Sinsan expected (its principal variation)
  expectedLabel?: string;
  loss?: number; // win chance given away by the move
  cls?: MoveClass;
}

export interface AnalysisPanel {
  reset(sinsan: Side): void;
  addAnalysis(ply: number, before: Position, analysis: ServerAnalysis): void;
  noteOpponentMove(ply: number, before: Position, move: Move): void;
  gameOver(result: GameResult): void;
  showHint(kind: 'hint' | 'threat', before: Position, analysis: ServerAnalysis): void;
  setWaiting(waiting: boolean, what?: string): void;
}

export interface PanelCallbacks {
  onArrows(arrows: BoardArrow[]): void;
  /** Your win chance (0..1), or null when unknown. */
  onEval(humanWin: number | null): void;
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

/** Labels for a line of moves played from `position`. */
function lineLabels(position: Position, ucis: string[]): string[] {
  const out: string[] = [];
  let pos = position;
  for (const u of ucis) {
    out.push(moveLabel(pos, u));
    pos = applyMove(pos, parseUci(u));
  }
  return out;
}

function pct(p: number | null): string {
  return p === null ? '—' : `${(p * 100).toFixed(0)}%`;
}

/** Red (bad for Sinsan) through amber to green (good for Sinsan). */
function winColor(p: number | null): string {
  if (p === null) return 'var(--panel-muted)';
  const hue = Math.max(0, Math.min(120, p * 120));
  return `hsl(${hue.toFixed(0)} 60% 45%)`;
}

const other = (side: Side): Side => (side === 'cho' ? 'han' : 'cho');
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
      <div class="ap-eval-head"><span class="ap-title">Sinsan's win estimate</span><span class="ap-eval-num">—</span></div>
      <div class="ap-bar"><div class="ap-bar-fill"></div><div class="ap-bar-net" title="Network's first instinct, before search"></div></div>
      <div class="ap-sub ap-eval-detail">Waiting for Sinsan's first move.</div>
      <div class="ap-sub ap-stats"></div>
    </section>
    <section class="ap-hint-box" hidden>
      <div class="ap-title-row"><span class="ap-title ap-hint-title"></span><button class="ap-close" title="Close">×</button></div>
      <div class="ap-hint-text"></div>
    </section>
    <section>
      <div class="ap-title-row"><span class="ap-title">Engine lines</span><span class="ap-sub">top 3 candidates, best play</span></div>
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
      <div class="ap-title">Win estimate over the game</div>
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
  const evalNum = q<HTMLSpanElement>('.ap-eval-num');
  const barFill = q<HTMLDivElement>('.ap-bar-fill');
  const barNet = q<HTMLDivElement>('.ap-bar-net');
  const evalDetail = q<HTMLDivElement>('.ap-eval-detail');
  const stats = q<HTMLDivElement>('.ap-stats');
  const hintBox = q<HTMLElement>('.ap-hint-box');
  const hintTitle = q<HTMLSpanElement>('.ap-hint-title');
  const hintText = q<HTMLDivElement>('.ap-hint-text');
  const linesBox = q<HTMLDivElement>('.ap-lines');
  const graph = q<HTMLDivElement>('.ap-graph');
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
  let sinsan: Side = 'han';
  let live = true; // the selected entry is the latest one and nobody has moved since: arrows match the board
  let hintArrows: BoardArrow[] | null = null;

  function showArrows(arrows: BoardArrow[]): void {
    callbacks.onArrows(hintArrows ?? (live ? arrows : []));
  }

  /** Sinsan's win chance (0..1) for the move it played in an entry (its Q), else the root value. */
  function chosenWin(entry: Entry): number {
    const c = entry.analysis.candidates.find((x) => x.move === entry.analysis.move);
    return c && c.q !== null ? (c.q + 1) / 2 : (entry.analysis.value + 1) / 2;
  }

  function buildTree(entry: Entry, detailed: boolean): TreeNode {
    const { analysis, before } = entry;
    const total = Math.max(1, analysis.visits);
    const minShare = detailed ? 0.004 : 0.015;
    const root: TreeNode = {
      move: '', label: 'now', depth: 0, visits: total, win: (analysis.value + 1) / 2, prior: 1, terminal: false,
      mover: other(sinsan), parent: null, children: [], x: 0, y: 0,
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
      const mover: Side = depth % 2 === 1 ? sinsan : other(sinsan);
      const winForMover = qv === null ? null : (qv + 1) / 2;
      const node: TreeNode = {
        move, label: moveLabel(parentPos, move), depth, visits,
        win: winForMover === null ? null : mover === sinsan ? winForMover : 1 - winForMover,
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
      const g = svg('g', { class: `ap-node ${n.depth === 0 ? 'ap-root' : `ap-${n.mover === sinsan ? 'own' : 'opp'}`}` });
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
      title.textContent = n.depth === 0 ? 'Position Sinsan searched' :
        `${n.mover === sinsan ? 'Sinsan' : 'You'}: ${n.label} (${n.move})\n${n.visits} visits, prior ${pct(n.prior)}\n` +
        `Sinsan win ${pct(n.win)}${n.terminal ? '\ngame ends here' : ''}`;
      g.appendChild(title);
      if (n.depth > 0) {
        g.addEventListener('mouseenter', () => {
          const path = linePath(n);
          showArrows(arrowsFor(path.map((p) => p.move), sinsan));
          lineBox.textContent = path
            .map((p, i) => `${i + 1}. ${p.mover === sinsan ? 'Sinsan' : 'You'} ${p.label}`)
            .join('  ') + `  → Sinsan ${pct(n.win)}`;
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

  /** On the board after Sinsan's move: the reply it expects (and, faintly, its follow-up). */
  function defaultArrows(entry: Entry): BoardArrow[] {
    return arrowsFor(entry.analysis.pv.slice(1, 3), other(sinsan), true);
  }

  /** Top candidates, each continued by the most-visited child at every ply (the tree's MultiPV). */
  function renderLines(entry: Entry): void {
    const { analysis, before } = entry;
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
        const labels = lineLabels(before, line);
        div.append(badge, el('span', 'ap-pv-moves', labels.map((l, i) => (i % 2 === 0 ? l : `${l}`)).join('  ')));
        div.title = `${r[3]} visits`;
        div.addEventListener('mouseenter', () => showArrows(arrowsFor(line.slice(0, 6), sinsan)));
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
        tr.addEventListener('mouseenter', () => showArrows(arrowsFor([c.move], sinsan)));
        tr.addEventListener('mouseleave', () => showArrows(defaultArrows(entry)));
        return tr;
      }),
    );
  }

  function renderGraph(): void {
    const w = 400;
    const h = 96;
    const s = svg('svg', { width: '100%', viewBox: `0 0 ${w} ${h}`, class: 'ap-graph-svg' });
    s.appendChild(svg('line', { x1: 0, x2: w, y1: h / 2, y2: h / 2, class: 'ap-mid' }));
    if (entries.length > 0) {
      const maxPly = Math.max(30, entries[entries.length - 1]!.ply + 2);
      const xy = (e: Entry) => [(e.ply / maxPly) * (w - 10) + 5, (1 - (e.analysis.value + 1) / 2) * (h - 10) + 5] as const;
      const pts = entries.map(xy);
      const area = `M${pts[0]![0]},${h / 2} ` + pts.map(([x, y]) => `L${x},${y}`).join(' ') + ` L${pts[pts.length - 1]![0]},${h / 2} Z`;
      s.appendChild(svg('path', { d: area, class: 'ap-area' }));
      s.appendChild(svg('polyline', { points: pts.map(([x, y]) => `${x},${y}`).join(' '), class: 'ap-curve' }));
      // Your costly moves, marked where they happened.
      for (const p of played) {
        if (p.cls !== 'mistake' && p.cls !== 'blunder' && p.cls !== 'inaccuracy') continue;
        const x = (p.ply / maxPly) * (w - 10) + 5;
        const mark = svg('line', { x1: x, x2: x, y1: 4, y2: h - 4, class: `ap-mark ap-mark-${p.cls}` });
        const t = svg('title', {});
        t.textContent = `Move ${p.ply + 1}: your ${p.label} - ${CLASS_INFO[p.cls].label}`;
        mark.appendChild(t);
        s.appendChild(mark);
      }
      pts.forEach(([x, y], i) => {
        const c = svg('circle', { cx: x, cy: y, r: i === selected ? 4.5 : 2.8, class: i === selected ? 'ap-pt ap-pt-sel' : 'ap-pt' });
        const t = svg('title', {});
        const e = entries[i]!;
        t.textContent = `Move ${e.ply + 1}: Sinsan ${moveLabel(e.before, e.analysis.move)}, win ${pct((e.analysis.value + 1) / 2)} (click to inspect)`;
        c.appendChild(t);
        c.addEventListener('click', () => select(i));
        s.appendChild(c);
      });
    }
    graph.replaceChildren(s);
  }

  function renderMoves(): void {
    const rows: HTMLElement[] = [];
    for (const p of played) {
      const div = el('div', `ap-mv ap-mv-${p.side === sinsan ? 'sinsan' : 'you'}`);
      div.append(el('span', 'ap-mv-num', `${p.ply + 1}.`), el('span', 'ap-mv-who', p.side === sinsan ? 'Sinsan' : 'You'),
        el('span', 'ap-mv-label', p.label));
      if (p.entry !== undefined) {
        const e = entries[p.entry]!;
        div.append(el('span', 'ap-mv-info', `${pct(chosenWin(e))} · ${e.analysis.visits.toLocaleString()} visits`));
        div.classList.add('ap-mv-click');
        if (p.entry === selected) div.classList.add('ap-mv-sel');
        div.addEventListener('click', () => select(p.entry!));
      } else if (p.cls) {
        const badge = el('span', `ap-cls ap-cls-${p.cls}`, `${CLASS_INFO[p.cls].symbol} ${CLASS_INFO[p.cls].label}`);
        const lossText = p.loss !== undefined && p.loss >= 0.005 ? ` (−${(p.loss * 100).toFixed(0)}%)` : '';
        badge.title = p.expectedLabel && p.expected !== p.uci ? `Sinsan expected ${p.expectedLabel}` : 'Sinsan expected this move';
        div.append(badge, el('span', 'ap-mv-info', lossText + (p.expectedLabel && p.expected !== p.uci ? ` best ${p.expectedLabel}` : '')));
      } else if (p.before !== undefined) {
        div.append(el('span', 'ap-mv-info', 'rating…'));
      }
      rows.push(div);
    }
    movesBox.replaceChildren(...(rows.length ? rows : [el('div', 'ap-sub', 'No moves yet.')]));
    movesBox.scrollTop = movesBox.scrollHeight;
    const rated = played.filter((p) => p.loss !== undefined);
    accuracyEl.textContent = rated.length
      ? `your accuracy ${(rated.reduce((a, p) => a + moveAccuracy(p.loss!), 0) / rated.length).toFixed(1)}%`
      : '';
  }

  function select(i: number): void {
    selected = i;
    live = i === entries.length - 1 && played[played.length - 1]?.entry === i;
    const entry = entries[i]!;
    const { analysis } = entry;
    const win = (analysis.value + 1) / 2;
    const netWin = analysis.net_value === null ? null : (analysis.net_value + 1) / 2;
    evalNum.textContent = pct(win);
    evalNum.style.color = winColor(win);
    barFill.style.width = `${win * 100}%`;
    barFill.style.background = winColor(win);
    barNet.style.left = netWin === null ? '-10px' : `calc(${netWin * 100}% - 1px)`;
    evalDetail.textContent =
      `${i === entries.length - 1 ? 'Latest search' : `Viewing move ${entry.ply + 1}`} - ` +
      (netWin === null ? 'search' : `first instinct ${pct(netWin)}, after search`) + ` ${pct(win)}` +
      ` · ${analysis.cp > 0 ? '+' : ''}${(analysis.cp / 100).toFixed(2)}`;
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

  function rateLastHumanMove(humanWinAfter: number): void {
    const p = [...played].reverse().find((x) => x.side !== sinsan);
    if (!p || p.before === undefined || p.loss !== undefined) return;
    p.loss = Math.max(0, p.before - humanWinAfter);
    p.cls = classify(p.loss, p.expected === p.uci);
  }

  function renderReview(result: GameResult): void {
    const rated = played.filter((p) => p.cls);
    const counts = new Map<MoveClass, number>();
    for (const p of rated) counts.set(p.cls!, (counts.get(p.cls!) ?? 0) + 1);
    const worst = [...rated].sort((a, b) => b.loss! - a.loss!)[0];
    const youWon = result.winner !== null && result.winner !== sinsan;
    reviewBody.replaceChildren(
      el('div', 'ap-review-head', `${youWon ? 'You won' : result.winner === sinsan ? 'Sinsan won' : 'Draw'} · ` +
        (rated.length ? `your accuracy ${(rated.reduce((a, p) => a + moveAccuracy(p.loss!), 0) / rated.length).toFixed(1)}%` : 'no rated moves')),
      (() => {
        const grid = el('div', 'ap-review-grid');
        for (const cls of Object.keys(CLASS_INFO) as MoveClass[]) {
          grid.append(el('span', `ap-cls ap-cls-${cls}`, `${CLASS_INFO[cls].symbol} ${CLASS_INFO[cls].label}`), el('span', '', String(counts.get(cls) ?? 0)));
        }
        return grid;
      })(),
      el('div', 'ap-sub', worst && worst.loss! >= 0.05
        ? `Biggest swing: move ${worst.ply + 1}, your ${worst.label} (−${(worst.loss! * 100).toFixed(0)}%); Sinsan expected ${worst.expectedLabel ?? '—'}.`
        : 'No big swings - well played.'),
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
    reset(side: Side) {
      entries = [];
      played = [];
      selected = -1;
      sinsan = side;
      live = true;
      clearHint();
      evalNum.textContent = '—';
      evalNum.style.color = '';
      barFill.style.width = '50%';
      barFill.style.background = '';
      barNet.style.left = '-10px';
      evalDetail.textContent = `Sinsan plays ${sinsan === 'cho' ? 'Cho' : 'Han'}. Waiting for its first search.`;
      stats.textContent = '';
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
      rateLastHumanMove(1 - (analysis.value + 1) / 2);
      entries.push({ ply, before, analysis });
      played.push({ ply, side: sinsan, uci: analysis.move, label: moveLabel(before, analysis.move), entry: entries.length - 1 });
      select(entries.length - 1);
      callbacks.onEval(1 - chosenWin(entries[entries.length - 1]!));
    },
    noteOpponentMove(ply, before, move) {
      clearHint();
      const uci = moveToUci(move, before);
      const last = entries[entries.length - 1];
      const fresh = last && last.ply === ply - 1;
      played.push({
        ply, side: other(sinsan), uci, label: moveLabel(before, uci),
        before: fresh ? 1 - chosenWin(last) : undefined,
        expected: fresh ? last.analysis.pv[1] : undefined,
        expectedLabel: fresh && last.analysis.pv[1] ? moveLabel(before, last.analysis.pv[1]) : undefined,
      });
      live = false; // the board has moved on; arrows from the last search would point at the wrong position
      renderMoves();
      callbacks.onArrows([]);
    },
    gameOver(result) {
      const last = played[played.length - 1];
      if (last && last.side !== sinsan) rateLastHumanMove(result.winner === null ? 0.5 : result.winner === sinsan ? 0 : 1);
      if (selected >= 0) renderGraph();
      renderMoves();
      renderReview(result);
      callbacks.onEval(result.winner === null ? 0.5 : result.winner === sinsan ? 0 : 1);
    },
    showHint(kind, before, analysis) {
      const you = other(sinsan);
      const win = (analysis.value + 1) / 2; // for the side that was to move in the searched position
      const youWin = kind === 'hint' ? win : 1 - win;
      const labels = lineLabels(before, analysis.pv.slice(0, 6));
      hintTitle.textContent = kind === 'hint' ? 'Hint: Sinsan would play this for you' : 'Threat: what Sinsan would do if you passed';
      hintText.textContent = `${labels.join('  ')}  · your win ${pct(youWin)} · ${analysis.visits.toLocaleString()} visits`;
      hintBox.hidden = false;
      hintArrows = arrowsFor(analysis.pv.slice(0, kind === 'hint' ? 3 : 2), kind === 'hint' ? you : sinsan);
      callbacks.onArrows(hintArrows);
    },
    setWaiting(waiting, what) {
      container.classList.toggle('ap-waiting', waiting);
      if (waiting && what) stats.textContent = what;
    },
  };
}
