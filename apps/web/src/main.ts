import './style.css';
import {
  applyMove,
  createInitialPosition,
  generateLegalMoves,
  getGameResult,
  isCheck,
} from '@sinsan/rules';
import type { Formation, GameResult, Move, Position, Side } from '@sinsan/rules';
import { createBoardView } from '@sinsan/ui';
import { createAnalysisPanel, moveToUci } from './analysis-panel.ts';
import { createEngineSettings, PRESETS, presetSettings, requestFieldsFor, serverDown } from './engine-settings.ts';
import { fetchProgress, renderAbout, renderProgress, renderRules } from './pages.ts';
import { mountNetworkViz } from './network-viz.ts';
import { studentMove } from './student-client.ts';
import type { StudentRequest } from './student-worker.ts';
import type { FsfInfo, ServerAnalysis } from './analysis-panel.ts';
import type { BoardView } from '@sinsan/ui';

const FORMATIONS: Formation[] = ['masang-sangma', 'sangma-masang', 'masang-masang', 'sangma-sangma'];
// 마상 (masang) = horse-then-elephant, 상마 (sangma) = elephant-then-horse - the two orders each
// side can set its own left/right wing to at setup. Display labels are the real Hangul terms for
// these, not the internal English-romanized Formation identifiers used everywhere else in code.
const FORMATION_LABELS: Record<Formation, string> = {
  'masang-sangma': '마상-상마',
  'sangma-masang': '상마-마상',
  'masang-masang': '마상-마상',
  'sangma-sangma': '상마-상마',
};
const HUMAN_MOVE_MS = 180;
const AI_MOVE_MS = 230;
const SIDES: Side[] = ['cho', 'han'];

/** Who plays a side: "human", "sinsan:custom", "sinsan:<preset id>", or "fsf:<nnue|classical>:<nodes>". */
type PlayerId = string;

const FSF_LEVELS: [string, string][] = [
  ['fsf:nnue:1000', 'Fairy-Stockfish NNUE 1K nodes (0)'],
  ['fsf:nnue:4000', 'Fairy-Stockfish NNUE 4K (574)'],
  ['fsf:nnue:16000', 'Fairy-Stockfish NNUE 16K (1198)'],
  ['fsf:nnue:64000', 'Fairy-Stockfish NNUE 64K (1771)'],
  ['fsf:nnue:256000', 'Fairy-Stockfish NNUE 256K (2312)'],
  ['fsf:nnue:1000000', 'Fairy-Stockfish NNUE 1M nodes'],
  ['fsf:classical:16000', 'Fairy-Stockfish classical 16K (466)'],
];

const PLAYER_OPTIONS: [PlayerId, string][] = [
  ['human', 'You'],
  ['sinsan:custom', 'Sinsan - my settings'],
  ...PRESETS.map((p): [PlayerId, string] => [`sinsan:${p.id}`, `Sinsan ${p.label.split(' - ')[0]}`]),
  ...FSF_LEVELS,
];

function shortName(id: PlayerId): string {
  if (id === 'human') return 'You';
  if (id === 'sinsan:custom') return 'Sinsan';
  if (id.startsWith('sinsan:')) return `Sinsan ${PRESETS.find((p) => `sinsan:${p.id}` === id)?.label.split(' - ')[0] ?? ''}`;
  const [, kind, nodes] = id.split(':');
  const n = Number(nodes);
  return `FSF${kind === 'classical' ? ' classical' : ''} ${n >= 1e6 ? `${n / 1e6}M` : `${n / 1000}K`}`;
}

const app = document.querySelector<HTMLDivElement>('#app')!;
const playerSelect = (id: string, chosen: PlayerId) => `<select id="${id}">${PLAYER_OPTIONS.map(
  ([v, label]) => `<option value="${v}"${v === chosen ? ' selected' : ''}>${label}</option>`).join('')}</select>`;
app.innerHTML = `
  <header class="site-header">
    <a class="sinsan-title" href="#play">神算 Sinsan</a>
    <nav class="site-nav">
      <a href="#play" data-view="play">Play</a>
      <a href="#about" data-view="about">How it works</a>
      <a href="#progress" data-view="progress">Progress</a>
      <a href="#rules" data-view="rules">Rules</a>
    </nav>
  </header>
  <section class="view" id="view-play">
    <div class="sinsan-controls setup-row">
      <label class="player-pick cho-pick">Cho (초) ${playerSelect('player-cho', 'human')}</label>
      <label>formation <select id="setup-cho">${FORMATIONS.map((f) => `<option value="${f}">${FORMATION_LABELS[f]}</option>`).join('')}</select></label>
      <span class="vs">vs</span>
      <label class="player-pick han-pick">Han (한) ${playerSelect('player-han', 'sinsan:custom')}</label>
      <label>formation <select id="setup-han">${FORMATIONS.map((f) => `<option value="${f}">${FORMATION_LABELS[f]}</option>`).join('')}</select></label>
    </div>
    <div class="sinsan-layout">
      <div class="sinsan-play">
        <div class="sinsan-board-row">
          <div class="eval-bar" title="Winning chances of the side at the bottom"><div class="eval-bar-fill"></div><span class="eval-bar-num"></span></div>
          <div id="board"></div>
        </div>
        <div class="sinsan-controls">
          <button id="pass-btn">Pass</button>
          <button id="hint-btn" title="Ask Sinsan for the best move in your position">Hint</button>
          <button id="threat-btn" title="Show what your opponent could do if you passed">Threat</button>
          <button id="resign-btn">Resign</button>
          <button id="new-game-btn">New Game</button>
        </div>
        <div class="sinsan-controls bot-controls" hidden>
          <button id="pause-btn">Pause</button>
          <button id="step-btn" title="Play one move">Step</button>
          <label>Pace <select id="pace">
            <option value="150">Fast</option><option value="900" selected>Normal</option><option value="2500">Slow</option>
          </select></label>
        </div>
        <div class="sinsan-status" id="status"></div>
        <div class="sinsan-status" id="ai-info"></div>
        <div id="engine-settings"></div>
      </div>
      <aside id="analysis"></aside>
    </div>
  </section>
  <section class="view" id="view-about" hidden></section>
  <section class="view" id="view-progress" hidden></section>
  <section class="view" id="view-rules" hidden></section>
`;

const $ = <T extends Element>(sel: string) => document.querySelector<T>(sel)!;
const boardEl = $<HTMLDivElement>('#board');
const statusEl = $<HTMLDivElement>('#status');
const playerSelects: Record<Side, HTMLSelectElement> = { cho: $('#player-cho'), han: $('#player-han') };
const setupChoSelect = $<HTMLSelectElement>('#setup-cho');
const setupHanSelect = $<HTMLSelectElement>('#setup-han');
const passBtn = $<HTMLButtonElement>('#pass-btn');
const resignBtn = $<HTMLButtonElement>('#resign-btn');
const newGameBtn = $<HTMLButtonElement>('#new-game-btn');
const aiInfoEl = $<HTMLDivElement>('#ai-info');
const hintBtn = $<HTMLButtonElement>('#hint-btn');
const threatBtn = $<HTMLButtonElement>('#threat-btn');
const botControls = $<HTMLDivElement>('.bot-controls');
const pauseBtn = $<HTMLButtonElement>('#pause-btn');
const stepBtn = $<HTMLButtonElement>('#step-btn');
const paceSelect = $<HTMLSelectElement>('#pace');
const evalBarFill = $<HTMLDivElement>('.eval-bar-fill');
const evalBarNum = $<HTMLSpanElement>('.eval-bar-num');
const panel = createAnalysisPanel($<HTMLElement>('#analysis'), {
  onArrows: (arrows) => boardView?.setArrows(arrows),
  onEval: (win) => {
    // The bottom side's share fills from the bottom.
    evalBarFill.style.height = `${(win ?? 0.5) * 100}%`;
    evalBarNum.textContent = win === null ? '' : `${Math.round(win * 100)}`;
  },
});
const engineSettings = createEngineSettings($<HTMLElement>('#engine-settings'), () => updateStatus());

let position: Position;
let players: Record<Side, PlayerId> = { cho: 'human', han: 'sinsan:custom' };
let names: Record<Side, string> = { cho: 'You', han: 'Sinsan' };
let viewer: Side = 'cho';
let boardView: BoardView;
let gameOver = false;
let gameId = 0; // bumps on every new game so late replies from a previous game are dropped
let botTimer: ReturnType<typeof setTimeout> | undefined;
let botBusy = false;
let paused = false;
/** Moves of the current game in Fairy-Stockfish notation (files a-i, ranks 1-10 from Cho's side). */
let uciMoves: string[] = [];
let gameSetup = { cho: 'masang-sangma' as Formation, han: 'masang-sangma' as Formation };
let hintBusy = false;

const isHuman = (side: Side) => players[side] === 'human';
const humanSides = () => SIDES.filter(isHuman);
const botGame = () => humanSides().length === 0;

function resultText(result: GameResult): string {
  const winnerName = result.winner ? `${result.winner === 'cho' ? 'Cho' : 'Han'} (${names[result.winner]})` : null;
  // Bikjang, double pass, the move limit and repetition are decided by points under the default rules.
  const byPoints = (what: string): string => (winnerName ? `${what} - ${winnerName} wins on points.` : `${what} - draw.`);
  switch (result.kind) {
    case 'checkmate':
      return `Checkmate - ${winnerName} wins.`;
    case 'stalemate':
      return `No legal moves - ${winnerName} wins.`;
    case 'bikjang':
      return byPoints('Bikjang');
    case 'double-pass':
      return byPoints('Both sides passed');
    case 'move-limit':
      return byPoints('No-capture limit reached');
    case 'repetition':
      return byPoints('Repetition');
    case 'perpetual-check':
      return `Perpetual check - ${winnerName} wins.`;
    case 'timeout':
      return `Timeout - ${winnerName} wins.`;
    case 'resignation':
      return `Resignation - ${winnerName} wins.`;
  }
}

function updateStatus(): void {
  if (gameOver) return;
  const result = getGameResult(position);
  if (result) {
    gameOver = true;
    statusEl.textContent = resultText(result);
    for (const b of [passBtn, hintBtn, threatBtn, pauseBtn, stepBtn]) b.disabled = true;
    panel.gameOver(result);
    return;
  }
  const side = position.sideToMove;
  const checkNote = isCheck(position) ? ' - check!' : '';
  const thinking = botBusy ? ' (thinking…)' : paused && !isHuman(side) ? ' (paused)' : '';
  statusEl.textContent = `${side === 'cho' ? 'Cho' : 'Han'} (${names[side]}) to move${checkNote}${thinking}`;
  const humanTurn = isHuman(side);
  passBtn.disabled = !humanTurn || isCheck(position);
  hintBtn.disabled = !humanTurn || hintBusy;
  threatBtn.disabled = !humanTurn || hintBusy || isCheck(position);
  resignBtn.disabled = !humanTurn;
  pauseBtn.disabled = false;
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  stepBtn.disabled = !paused || botBusy || humanTurn;
}

function afterMove(move: Move, byBot: boolean): void {
  boardView.sync(position, move, byBot ? AI_MOVE_MS : HUMAN_MOVE_MS);
  updateStatus();
  scheduleBot();
}

async function post<T>(path: string, body: object): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...gameSetup, ...body }),
  });
  const reply = (await response.json()) as T & { error?: string };
  if (!response.ok) throw new Error(reply.error ?? `server error ${response.status}`);
  return reply;
}

/** Sinsan's search: the in-browser student when the chosen network is "browser", else the GPU server. */
function searchMove(body: { moves: string[] } & Record<string, unknown>): Promise<ServerAnalysis> {
  if (body['net'] === 'browser') return studentMove<ServerAnalysis>({ ...gameSetup, ...body } as StudentRequest);
  return post<ServerAnalysis>('/api/move', body);
}

function sinsanFields(id: PlayerId): Record<string, number | string> {
  const f = id === 'sinsan:custom' ? engineSettings.requestFields() : requestFieldsFor(presetSettings(id.slice('sinsan:'.length)));
  if (serverDown) f['net'] = 'browser'; // no GPU server: every Sinsan plays with the in-browser network
  return f;
}

/** Asks the side to move's bot for its move and records its analysis. */
async function botMove(current: Position, side: Side): Promise<Move> {
  const legal = generateLegalMoves(current);
  const id = players[side];
  const ply = uciMoves.length;
  if (id.startsWith('fsf:')) {
    const [, kind, nodes] = id.split(':');
    const info = await post<FsfInfo>('/api/fsf', { moves: uciMoves, nodes: Number(nodes), nnue: kind === 'nnue' });
    const move = legal.find((m) => moveToUci(m, current) === info.move);
    if (!move) throw new Error(`Fairy-Stockfish returned illegal move ${info.move}`);
    panel.noteMove(ply, current, move, info);
    aiInfoEl.textContent = `${names[side]}: depth ${info.depth}, ${info.nodes.toLocaleString()} nodes, ${info.ms.toFixed(0)}ms`;
    return move;
  }
  const reply = await searchMove({ moves: uciMoves, ...sinsanFields(id) });
  const move = legal.find((m) => moveToUci(m, current) === reply.move);
  if (!move) throw new Error(`server returned illegal move ${reply.move}`);
  panel.addAnalysis(ply, current, reply);
  aiInfoEl.textContent = `${names[side]}: ${reply.visits.toLocaleString()} visits, depth ${reply.depth}, ${reply.ms.toFixed(0)}ms`;
  return move;
}

function scheduleBot(delay?: number): void {
  clearTimeout(botTimer);
  if (gameOver || isHuman(position.sideToMove) || paused || botBusy) return;
  const wait = delay ?? (botGame() ? Number(paceSelect.value) : 300);
  botTimer = setTimeout(() => void runBot(), wait);
}

async function runBot(): Promise<void> {
  if (gameOver || isHuman(position.sideToMove) || botBusy) return;
  const id = gameId;
  const side = position.sideToMove;
  botBusy = true;
  updateStatus();
  panel.setWaiting(true);
  let move: Move;
  try {
    move = await botMove(position, side);
  } catch (error) {
    if (id === gameId) {
      aiInfoEl.textContent = `Engine unavailable (${String(error)}); is training/play_server.py running?`;
      paused = true;
    }
    return;
  } finally {
    if (id === gameId) {
      botBusy = false;
      panel.setWaiting(false);
      updateStatus();
    }
  }
  if (id !== gameId || gameOver) return; // a new game started (or someone resigned) while it was thinking
  uciMoves.push(moveToUci(move, position));
  position = applyMove(position, move);
  afterMove(move, true);
}

async function showHint(kind: 'hint' | 'threat'): Promise<void> {
  if (gameOver || !isHuman(position.sideToMove) || hintBusy) return;
  const pass = generateLegalMoves(position).find((m) => m.isPass);
  if (kind === 'threat' && !pass) return;
  hintBusy = true;
  updateStatus();
  panel.setWaiting(true, kind === 'hint' ? 'Searching your position…' : 'Searching what your opponent threatens…');
  const id = gameId;
  try {
    const fields = { ...engineSettings.requestFields(), temperature: 0 };
    if (kind === 'hint') {
      const reply = await searchMove({ moves: uciMoves, ...fields });
      if (id === gameId) panel.showHint('hint', position, reply);
    } else {
      const afterPass = applyMove(position, pass!);
      const reply = await searchMove({ moves: [...uciMoves, moveToUci(pass!, position)], ...fields });
      if (id === gameId) panel.showHint('threat', afterPass, reply);
    }
  } catch (error) {
    aiInfoEl.textContent = `Engine unavailable (${String(error)}); is training/play_server.py running?`;
  } finally {
    hintBusy = false;
    panel.setWaiting(false);
    updateStatus();
  }
}

function handleHumanMove(move: Move): void {
  if (gameOver || !isHuman(position.sideToMove)) return;
  panel.noteMove(uciMoves.length, position, move);
  uciMoves.push(moveToUci(move, position));
  position = applyMove(position, move);
  afterMove(move, false);
}

function newGame(): void {
  clearTimeout(botTimer);
  gameId++;
  gameOver = false;
  botBusy = false;
  hintBusy = false;
  paused = false;
  players = { cho: playerSelects.cho.value, han: playerSelects.han.value };
  names = { cho: shortName(players.cho), han: shortName(players.han) };
  if (names.cho === names.han) names = { cho: `${names.cho} (Cho)`, han: `${names.han} (Han)` };
  // A lone human sits at the bottom; otherwise Cho does.
  viewer = humanSides().length === 1 ? humanSides()[0]! : 'cho';
  botControls.hidden = !botGame();
  aiInfoEl.textContent = '';
  uciMoves = [];
  panel.reset(names, viewer, humanSides());
  gameSetup = { cho: setupChoSelect.value as Formation, han: setupHanSelect.value as Formation };
  position = createInitialPosition({ setupCho: gameSetup.cho, setupHan: gameSetup.han });
  boardEl.innerHTML = '';
  boardView = createBoardView(boardEl, position, {
    humanSide: viewer,
    humanSides: humanSides(),
    onHumanMove: handleHumanMove,
    animationDurationMs: HUMAN_MOVE_MS,
  });
  updateStatus();
  scheduleBot(botGame() ? 800 : 350);
}

passBtn.addEventListener('click', () => {
  if (gameOver || !isHuman(position.sideToMove)) return;
  const pass = generateLegalMoves(position).find((m) => m.isPass);
  if (pass) handleHumanMove(pass);
});

resignBtn.addEventListener('click', () => {
  if (gameOver || !isHuman(position.sideToMove)) return;
  clearTimeout(botTimer);
  gameOver = true;
  const winner: Side = position.sideToMove === 'cho' ? 'han' : 'cho';
  statusEl.textContent = resultText({ winner, kind: 'resignation' });
  for (const b of [passBtn, hintBtn, threatBtn, pauseBtn, stepBtn]) b.disabled = true;
  panel.gameOver({ winner, kind: 'resignation' });
});

pauseBtn.addEventListener('click', () => {
  paused = !paused;
  updateStatus();
  if (!paused) scheduleBot(100);
  else clearTimeout(botTimer);
});

stepBtn.addEventListener('click', () => {
  if (!paused || botBusy) return;
  void runBot();
});

newGameBtn.addEventListener('click', newGame);
hintBtn.addEventListener('click', () => void showHint('hint'));
threatBtn.addEventListener('click', () => void showHint('threat'));
playerSelects.cho.addEventListener('change', newGame);
playerSelects.han.addEventListener('change', newGame);

newGame();

/** Adds measured ladder ratings to the Sinsan preset options (FSF options carry theirs already). */
async function labelPresetRatings(): Promise<void> {
  try {
    const data = await fetchProgress<{ presets?: Record<string, { elo: number; bound: string; low?: number; high?: number }> }>();
    for (const select of Object.values(playerSelects)) {
      for (const opt of Array.from(select.options)) {
        const id = opt.value.startsWith('sinsan:') ? opt.value.slice('sinsan:'.length) : null;
        if (!id || id === 'custom') continue;
        const rating = data.presets?.[id];
        const base = opt.textContent!.replace(/ \(.*\)$/, '');
        // Ladder Elo as the 95% range of its match, or a bound when the match was one-sided.
        const text = !rating ? 'not rated yet'
          : rating.bound ? `${rating.bound}${rating.elo}`
          : rating.low !== undefined && rating.high !== undefined ? `${rating.low}~${rating.high}` : String(rating.elo);
        opt.textContent = `${base} (${text})`;
      }
    }
  } catch {
    // server unreachable: leave the plain labels
  }
}
void labelPresetRatings();

// Simple hash routing between the game and the info pages; the game keeps running in the background.
const VIEWS = ['play', 'about', 'progress', 'rules'] as const;
type View = (typeof VIEWS)[number];
const rendered = new Set<View>();

function showView(): void {
  const wanted = location.hash.slice(1) as View;
  const view: View = VIEWS.includes(wanted) ? wanted : 'play';
  for (const v of VIEWS) document.querySelector<HTMLElement>(`#view-${v}`)!.hidden = v !== view;
  document.querySelectorAll<HTMLAnchorElement>('.site-nav a').forEach((a) => a.classList.toggle('active', a.dataset['view'] === view));
  const el = document.querySelector<HTMLElement>(`#view-${view}`)!;
  if (view === 'about' && !rendered.has(view)) {
    renderAbout(el);
    mountNetworkViz(el.querySelector<HTMLElement>('#network-viz')!, () => ({ ...gameSetup, moves: [...uciMoves] }));
  }
  if (view === 'rules' && !rendered.has(view)) renderRules(el);
  if (view === 'progress') void renderProgress(el); // live data: refresh on every visit
  rendered.add(view);
  if (view !== 'play') window.scrollTo(0, 0);
}

window.addEventListener('hashchange', showView);
showView();
