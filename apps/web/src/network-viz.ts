/** "Inside the network": Sinsan's real network on a real position, from /api/inspect and /api/move.
 * Instinct = the policy head as a heatmap; Attention = what one board point looks at in a chosen
 * layer (heads averaged); After search = where 800 visits went. */
import { applyMove, createInitialPosition, pieceAt } from '@sinsan/rules';
import type { Formation, Position } from '@sinsan/rules';
import { parseUci } from './analysis-panel.ts';

interface Inspect {
  side: 'cho' | 'han';
  priors: [string, number][];
  wdl: [number, number, number];
  margin: number[];
  attention: number[][][];
}

interface Searched {
  candidates: { move: string; visits: number; q: number | null; prior: number }[];
  visits: number;
}

interface GameRef {
  cho: Formation;
  han: Formation;
  moves: string[];
}

const SAMPLES: { id: string; label: string; game: GameRef }[] = [
  { id: 'opening', label: 'Opening (after Cho\'s first move)', game: { cho: 'masang-sangma', han: 'masang-sangma', moves: ['b1c3'] } },
  {
    id: 'midgame', label: 'Midgame (move 25, from Fairy-Stockfish self-play)', game: {
      cho: 'masang-sangma', han: 'masang-sangma', moves: ['b1c3', 'h10g8', 'b3e3', 'i7h7', 'e4d4', 'h8e8', 'a4b4', 'b10c8', 'h1g3',
        'b8d8', 'h3f3', 'e9e10', 'g4g5', 'e7f7', 'c3e4', 'd10d9', 'e2e1', 'f10e9', 'a1a3', 'd8f10', 'f1f2', 'g8e7', 'd1e2', 'i10i9'],
    },
  },
];

const PIECE_FILE: Record<string, string> = {
  chariot: 'chariot', cannon: 'cannon', horse: 'horse', elephant: 'elephant', guard: 'advisor', general: 'king', soldier: 'pawn',
};
const GLYPH: Record<string, string> = { chariot: '車', cannon: '包', horse: '馬', elephant: '象', guard: '士', general: '將', soldier: '兵' };

const CELL = 46;
const M = 30;
const X = (sq: number) => M + (sq % 9) * CELL;
const Y = (sq: number) => M + Math.floor(sq / 9) * CELL;
const sqName = (sq: number) => String.fromCharCode(97 + (sq % 9)) + String(10 - Math.floor(sq / 9));

function arrow(x1: number, y1: number, x2: number, y2: number, cls: string, width: number): string {
  const len = Math.hypot(x2 - x1, y2 - y1);
  const ux = (x2 - x1) / len;
  const uy = (y2 - y1) / len;
  const head = Math.max(10, width * 2.6);
  const bx = x2 - ux * head;
  const by = y2 - uy * head;
  const px = -uy * head * 0.55;
  const py = ux * head * 0.55;
  return `<line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}" class="${cls}" stroke-width="${width}"/>` +
    `<polygon points="${x2},${y2} ${bx + px},${by + py} ${bx - px},${by - py}" class="${cls}-head"/>`;
}

function positionOf(game: GameRef): Position {
  let pos = createInitialPosition({ setupCho: game.cho, setupHan: game.han });
  for (const m of game.moves) pos = applyMove(pos, parseUci(m));
  return pos;
}

function label(pos: Position, uci: string): string {
  const m = parseUci(uci);
  if (m.isPass) return 'pass';
  const p = pieceAt(pos, m.from);
  return `${p ? GLYPH[p.type] : '?'}${sqName(m.to)}`;
}

export function mountNetworkViz(el: HTMLElement, currentGame: () => GameRef | null): void {
  el.innerHTML = `
    <div class="nv-controls">
      <label>Position <select class="nv-pos">
        ${SAMPLES.map((s) => `<option value="${s.id}">${s.label}</option>`).join('')}
        <option value="current">Your current game</option>
      </select></label>
      <div class="nv-tabs" role="tablist">
        <button data-view="instinct" class="active">Instinct</button>
        <button data-view="attention">Attention</button>
        <button data-view="search">After search</button>
      </div>
      <label class="nv-layer">Layer <input type="range" min="1" max="12" value="12"><output>12</output></label>
    </div>
    <div class="nv-body">
      <div class="nv-board"></div>
      <div class="nv-side">
        <div class="nv-caption"></div>
        <div class="nv-block"><div class="nv-h">Win / loss <span class="nv-who"></span></div><div class="nv-wdl"></div></div>
        <div class="nv-block"><div class="nv-h">Final point margin <span class="nv-margin-ev"></span></div><div class="nv-margin"></div></div>
        <div class="nv-block"><div class="nv-h">Top moves</div><table class="nv-moves"><thead><tr><th>Move</th><th>Instinct</th><th>After search</th></tr></thead><tbody></tbody></table></div>
      </div>
    </div>`;
  const q = <T extends Element>(s: string) => el.querySelector<T>(s)!;
  const posSelect = q<HTMLSelectElement>('.nv-pos');
  const layerInput = q<HTMLInputElement>('.nv-layer input');
  const layerOut = q<HTMLOutputElement>('.nv-layer output');
  const boardBox = q<HTMLDivElement>('.nv-board');
  const caption = q<HTMLDivElement>('.nv-caption');

  let view: 'instinct' | 'attention' | 'search' = 'instinct';
  let game: GameRef = SAMPLES[0]!.game;
  let pos = positionOf(game);
  let data: Inspect | null = null;
  let searched: Searched | null = null;
  let focus = -1; // attention source square
  const cache = new Map<string, Inspect>();
  const searchCache = new Map<string, Searched>();
  const key = (g: GameRef) => `${g.cho}|${g.han}|${g.moves.join(' ')}`;

  async function post<T>(path: string, body: object): Promise<T> {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = (await r.json()) as T & { error?: string };
    if (!r.ok) throw new Error(j.error ?? String(r.status));
    return j;
  }

  function destHeat(moves: [string, number][]): number[] {
    const heat = new Array<number>(90).fill(0);
    for (const [m, p] of moves) {
      const mv = parseUci(m);
      if (!mv.isPass) heat[mv.to]! += p;
    }
    return heat;
  }

  function drawBoard(): void {
    const w = 8 * CELL + 2 * M;
    const h = 9 * CELL + 2 * M;
    const parts: string[] = [`<rect width="${w}" height="${h}" rx="8" class="mb-wood"/>`];
    for (let c = 0; c < 9; c++) parts.push(`<line x1="${M + c * CELL}" y1="${M}" x2="${M + c * CELL}" y2="${M + 9 * CELL}" class="mb-line"/>`);
    for (let r = 0; r < 10; r++) parts.push(`<line x1="${M}" y1="${M + r * CELL}" x2="${M + 8 * CELL}" y2="${M + r * CELL}" class="mb-line"/>`);
    for (const top of [0, 7]) {
      parts.push(`<line x1="${M + 3 * CELL}" y1="${M + top * CELL}" x2="${M + 5 * CELL}" y2="${M + (top + 2) * CELL}" class="mb-line"/>`);
      parts.push(`<line x1="${M + 5 * CELL}" y1="${M + top * CELL}" x2="${M + 3 * CELL}" y2="${M + (top + 2) * CELL}" class="mb-line"/>`);
    }
    // Heat layer.
    let heat: number[] | null = null;
    let arrows: [string, number][] = [];
    if (data && view === 'instinct') {
      heat = destHeat(data.priors);
      arrows = data.priors.slice(0, 4);
    } else if (data && view === 'attention') {
      const row = data.attention[Number(layerInput.value) - 1]?.[focus];
      if (row) heat = row.map((v) => v / 255);
    } else if (searched && view === 'search') {
      const total = Math.max(1, searched.candidates.reduce((a, c) => a + c.visits, 0));
      const shares = searched.candidates.map((c): [string, number] => [c.move, c.visits / total]);
      heat = destHeat(shares);
      arrows = [...shares].sort((a, b) => b[1] - a[1]).slice(0, 4);
    }
    if (heat) {
      const max = Math.max(...heat, 1e-9);
      heat.forEach((v, sq) => {
        if (v <= 0.002 * max) return;
        const a = view === 'attention' ? v : Math.sqrt(v / max);
        parts.push(`<circle cx="${X(sq)}" cy="${Y(sq)}" r="${CELL * 0.5}" class="nv-heat" opacity="${(0.12 + 0.75 * a).toFixed(2)}"/>`);
      });
    }
    const ps = CELL * 0.9;
    for (let sq = 0; sq < 90; sq++) {
      const p = pieceAt(pos, sq);
      if (!p) continue;
      const file = `/assets/pieces/janggihb-ka-wooden/${p.side === 'cho' ? 'blue' : 'red'}_${PIECE_FILE[p.type]}.svg`;
      parts.push(`<image href="${file}" x="${X(sq) - ps / 2}" y="${Y(sq) - ps / 2}" width="${ps}" height="${ps}"/>`);
    }
    if (view === 'attention' && focus >= 0) parts.push(`<circle cx="${X(focus)}" cy="${Y(focus)}" r="${CELL * 0.5}" class="nv-focus"/>`);
    const top = arrows[0]?.[1] ?? 1;
    for (const [m, p] of arrows) {
      const mv = parseUci(m);
      if (mv.isPass) continue;
      const x1 = X(mv.from);
      const y1 = Y(mv.from);
      const x2 = X(mv.to);
      const y2 = Y(mv.to);
      const len = Math.hypot(x2 - x1, y2 - y1);
      // Stop short of the destination's centre so the head doesn't cover the piece there.
      const ex = x2 - ((x2 - x1) / len) * CELL * 0.28;
      const ey = y2 - ((y2 - y1) / len) * CELL * 0.28;
      parts.push(arrow(x1, y1, ex, ey, 'nv-arrow', 3 + 6 * (p / top)));
      parts.push(`<g class="nv-tag"><rect x="${ex - 19}" y="${ey - CELL * 0.62}" width="38" height="17" rx="8"/>` +
        `<text x="${ex}" y="${ey - CELL * 0.62 + 12.5}" text-anchor="middle">${Math.round(p * 100)}%</text></g>`);
    }
    // Click targets for choosing the attention source.
    for (let sq = 0; sq < 90; sq++) parts.push(`<circle cx="${X(sq)}" cy="${Y(sq)}" r="${CELL * 0.48}" class="nv-hit" data-sq="${sq}"/>`);
    boardBox.innerHTML = `<svg viewBox="0 0 ${w} ${h}" class="nv-svg">${parts.join('')}</svg>`;
    boardBox.querySelectorAll<SVGCircleElement>('.nv-hit').forEach((c) => {
      c.addEventListener('click', () => {
        focus = Number(c.dataset['sq']);
        if (view !== 'attention') setView('attention');
        else render();
      });
    });
  }

  function drawSide(): void {
    const who = q<HTMLSpanElement>('.nv-who');
    const wdl = q<HTMLDivElement>('.nv-wdl');
    const marginBox = q<HTMLDivElement>('.nv-margin');
    const ev = q<HTMLSpanElement>('.nv-margin-ev');
    const body = q<HTMLTableSectionElement>('.nv-moves tbody');
    if (!data) {
      who.textContent = '';
      wdl.innerHTML = marginBox.innerHTML = body.innerHTML = '';
      return;
    }
    const sideName = data.side === 'cho' ? 'Cho' : 'Han';
    who.textContent = `for ${sideName}, to move`;
    const [w, d, l] = data.wdl;
    wdl.innerHTML = `<div class="nv-wdl-bar"><span class="nv-w" style="width:${w * 100}%">${Math.round(w * 100)}% win</span>` +
      (d > 0.02 ? `<span class="nv-d" style="width:${d * 100}%"></span>` : '') +
      `<span class="nv-l" style="width:${l * 100}%">${Math.round(l * 100)}% loss</span></div>`;
    // Margin: 121 bins over -60..60 points, grouped by 4 for display.
    const groups: number[] = [];
    for (let i = 0; i < 120; i += 4) groups.push(data.margin.slice(i, i + 4).reduce((a, b) => a + b, 0));
    const gmax = Math.max(...groups);
    const expected = data.margin.reduce((a, p, i) => a + p * (-60 + i), 0);
    ev.textContent = `expected ${expected > 0 ? '+' : ''}${expected.toFixed(1)} for ${sideName}`;
    const bw = 300 / groups.length;
    marginBox.innerHTML = `<svg viewBox="0 0 300 70" class="nv-hist">${groups.map((g, i) =>
      `<rect x="${i * bw + 1}" y="${56 - (g / gmax) * 50}" width="${bw - 2}" height="${(g / gmax) * 50}" class="${i * 4 - 60 + 2 >= 0 ? 'nv-pos-bin' : 'nv-neg-bin'}"/>`).join('')}
      <line x1="150" x2="150" y1="0" y2="58" class="ch-zero"/>
      <text x="0" y="68" class="ch-tick">−60</text><text x="150" y="68" class="ch-tick" text-anchor="middle">0</text><text x="300" y="68" class="ch-tick" text-anchor="end">+60</text></svg>`;
    const total = searched ? Math.max(1, searched.candidates.reduce((a, c) => a + c.visits, 0)) : 1;
    const after = new Map(searched?.candidates.map((c) => [c.move, c.visits / total]) ?? []);
    const rows = new Set([...data.priors.slice(0, 5).map(([m]) => m), ...(searched ? searched.candidates.slice(0, 5).map((c) => c.move) : [])]);
    const priorOf = new Map(data.priors);
    body.innerHTML = [...rows].sort((a, b) => (after.get(b) ?? priorOf.get(b) ?? 0) - (after.get(a) ?? priorOf.get(a) ?? 0)).slice(0, 6).map((m) => {
      const p = priorOf.get(m) ?? 0;
      const a = after.get(m);
      return `<tr><td>${label(pos, m)}</td><td><span class="nv-pbar" style="width:${p * 70}px"></span>${Math.round(p * 100)}%</td>` +
        `<td>${a === undefined ? (searched ? '0%' : '—') : `<span class="nv-pbar nv-sbar" style="width:${a * 70}px"></span>${Math.round(a * 100)}%`}</td></tr>`;
    }).join('');
  }

  function render(): void {
    el.querySelectorAll<HTMLButtonElement>('.nv-tabs button').forEach((b) => b.classList.toggle('active', b.dataset['view'] === view));
    q<HTMLLabelElement>('.nv-layer').style.visibility = view === 'attention' ? 'visible' : 'hidden';
    layerOut.textContent = layerInput.value;
    caption.innerHTML = view === 'instinct'
      ? '<b>Instinct</b> - the network\'s move probabilities before any search. Shaded points are where it wants to move; arrows are its top choices.'
      : view === 'attention'
        ? `<b>Attention</b> - what the point ${focus >= 0 ? `<b>${sqName(focus)}</b>` : ''} looks at in layer ${layerInput.value} (all heads averaged). Click any point; early layers look nearby, later ones across the board.`
        : '<b>After search</b> - where 800 visits went. Compare with Instinct: search moves weight to moves that hold up when read ahead.';
    drawBoard();
    drawSide();
  }

  function setView(v: typeof view): void {
    view = v;
    render();
    if (v === 'search' && !searched) void loadSearch();
  }

  async function loadSearch(): Promise<void> {
    const k = key(game);
    if (searchCache.has(k)) {
      searched = searchCache.get(k)!;
      render();
      return;
    }
    caption.innerHTML = 'Searching 800 visits…';
    try {
      const r = await post<Searched>('/api/move', { ...game, visits: 800, temperature: 0 });
      searchCache.set(k, r);
      if (key(game) === k) {
        searched = r;
        render();
      }
    } catch (e) {
      caption.textContent = `Search unavailable (${String(e)})`;
    }
  }

  async function load(): Promise<void> {
    const k = key(game);
    pos = positionOf(game);
    searched = searchCache.get(k) ?? null;
    data = cache.get(k) ?? null;
    render();
    if (!data) {
      caption.innerHTML = 'Asking the network…';
      try {
        const r = await post<Inspect>('/api/inspect', game);
        cache.set(k, r);
        if (key(game) !== k) return;
        data = r;
      } catch (e) {
        caption.textContent = `Network unavailable (${String(e)}); is training/play_server.py running?`;
        return;
      }
    }
    if (focus < 0 || !pieceAt(pos, focus)) {
      const top = data.priors.find(([m]) => !parseUci(m).isPass);
      focus = top ? parseUci(top[0]).from : 40;
    }
    render();
    if (view === 'search') void loadSearch();
  }

  posSelect.addEventListener('change', () => {
    if (posSelect.value === 'current') {
      const g = currentGame();
      game = g && g.moves.length ? { ...g, moves: [...g.moves] } : SAMPLES[0]!.game;
    } else {
      game = SAMPLES.find((s) => s.id === posSelect.value)!.game;
    }
    focus = -1;
    void load();
  });
  el.querySelectorAll<HTMLButtonElement>('.nv-tabs button').forEach((b) =>
    b.addEventListener('click', () => setView(b.dataset['view'] as typeof view)));
  layerInput.addEventListener('input', render);
  void load();
}
