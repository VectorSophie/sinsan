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
import { createEngineSettings } from './engine-settings.ts';
import { renderAbout, renderProgress, renderRules } from './pages.ts';
import type { ServerAnalysis } from './analysis-panel.ts';
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
const AI_THINK_DELAY_MS = 350;

const app = document.querySelector<HTMLDivElement>('#app')!;
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
    <div class="sinsan-controls">
      <label>Play as
        <select id="human-side">
          <option value="cho">Cho (초)</option>
          <option value="han">Han (한)</option>
        </select>
      </label>
      <label>Cho formation
        <select id="setup-cho">${FORMATIONS.map((f) => `<option value="${f}">${FORMATION_LABELS[f]}</option>`).join('')}</select>
      </label>
      <label>Han formation
        <select id="setup-han">${FORMATIONS.map((f) => `<option value="${f}">${FORMATION_LABELS[f]}</option>`).join('')}</select>
      </label>
    </div>
    <div class="sinsan-layout">
      <div class="sinsan-play">
        <div class="sinsan-board-row">
          <div class="eval-bar" title="Your winning chances as Sinsan sees them"><div class="eval-bar-fill"></div><span class="eval-bar-num"></span></div>
          <div id="board"></div>
        </div>
        <div class="sinsan-controls">
          <button id="pass-btn">Pass</button>
          <button id="hint-btn" title="Ask Sinsan for the best move in your position">Hint</button>
          <button id="threat-btn" title="Show what Sinsan would do if you passed">Threat</button>
          <button id="resign-btn">Resign</button>
          <button id="new-game-btn">New Game</button>
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

const boardEl = document.querySelector<HTMLDivElement>('#board')!;
const statusEl = document.querySelector<HTMLDivElement>('#status')!;
const humanSideSelect = document.querySelector<HTMLSelectElement>('#human-side')!;
const setupChoSelect = document.querySelector<HTMLSelectElement>('#setup-cho')!;
const setupHanSelect = document.querySelector<HTMLSelectElement>('#setup-han')!;
const passBtn = document.querySelector<HTMLButtonElement>('#pass-btn')!;
const resignBtn = document.querySelector<HTMLButtonElement>('#resign-btn')!;
const newGameBtn = document.querySelector<HTMLButtonElement>('#new-game-btn')!;
const aiInfoEl = document.querySelector<HTMLDivElement>('#ai-info')!;
const hintBtn = document.querySelector<HTMLButtonElement>('#hint-btn')!;
const threatBtn = document.querySelector<HTMLButtonElement>('#threat-btn')!;
const evalBarFill = document.querySelector<HTMLDivElement>('.eval-bar-fill')!;
const evalBarNum = document.querySelector<HTMLSpanElement>('.eval-bar-num')!;
const panel = createAnalysisPanel(document.querySelector<HTMLElement>('#analysis')!, {
  onArrows: (arrows) => boardView?.setArrows(arrows),
  onEval: (humanWin) => {
    // Your share fills from your side of the board (the bottom).
    evalBarFill.style.height = `${(humanWin ?? 0.5) * 100}%`;
    evalBarNum.textContent = humanWin === null ? '' : `${Math.round(humanWin * 100)}`;
  },
});
const engineSettings = createEngineSettings(document.querySelector<HTMLElement>('#engine-settings')!, () => updateStatus());

let position: Position;
let humanSide: Side = 'cho';
let boardView: BoardView;
let gameOver = false;
let aiTimer: ReturnType<typeof setTimeout> | undefined;
/** Moves of the current game in Fairy-Stockfish notation (files a-i, ranks 1-10 from Cho's side). */
let uciMoves: string[] = [];
let gameSetup = { cho: 'masang-sangma' as Formation, han: 'masang-sangma' as Formation };
let hintBusy = false;


function resultText(result: GameResult): string {
  const winnerName = result.winner === 'cho' ? 'Cho' : result.winner === 'han' ? 'Han' : null;
  // Bikjang, double pass, the move limit and repetition are decided by points under the default
  // rules, and drawn only under profiles without point counting.
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
    passBtn.disabled = true;
    hintBtn.disabled = true;
    threatBtn.disabled = true;
    panel.gameOver(result);
    return;
  }
  const turnName = position.sideToMove === 'cho' ? 'Cho' : 'Han';
  const checkNote = isCheck(position) ? ' - check!' : '';
  statusEl.textContent = `${turnName} to move${checkNote}`;
  passBtn.disabled = position.sideToMove !== humanSide || isCheck(position);
  const canAsk = position.sideToMove === humanSide && !hintBusy;
  hintBtn.disabled = !canAsk;
  threatBtn.disabled = !canAsk || isCheck(position);
}

function afterMove(move: Move): void {
  const isAiMove = move && position.sideToMove !== humanSide;
  boardView.sync(position, move, isAiMove ? AI_MOVE_MS : HUMAN_MOVE_MS);
  updateStatus();
  maybeScheduleAiMove();
}

async function serverSearch(moves: string[], fields: Record<string, number | string>): Promise<ServerAnalysis> {
  const response = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...gameSetup, moves, ...fields }),
  });
  const reply = (await response.json()) as ServerAnalysis & { error?: string };
  if (!response.ok) throw new Error(reply.error ?? `server error ${response.status}`);
  return reply;
}

async function pickServerMove(current: Position): Promise<Move> {
  const legalMoves = generateLegalMoves(current);
  const start = performance.now();
  const reply = await serverSearch(uciMoves, engineSettings.requestFields());
  const move = legalMoves.find((m) => moveToUci(m, current) === reply.move);
  if (!move) throw new Error(`server returned illegal move ${reply.move}`);
  const elapsedMs = performance.now() - start;
  panel.addAnalysis(uciMoves.length, current, reply);
  aiInfoEl.textContent = `Sinsan 2026 (${engineSettings.summary()}): ${elapsedMs.toFixed(0)}ms`;
  return move;
}

async function showHint(kind: 'hint' | 'threat'): Promise<void> {
  if (gameOver || position.sideToMove !== humanSide || hintBusy) return;
  const pass = generateLegalMoves(position).find((m) => m.isPass);
  if (kind === 'threat' && !pass) return;
  hintBusy = true;
  updateStatus();
  panel.setWaiting(true, kind === 'hint' ? 'Searching your position…' : 'Searching what Sinsan threatens…');
  try {
    const fields = { ...engineSettings.requestFields(), temperature: 0 };
    if (kind === 'hint') {
      panel.showHint('hint', position, await serverSearch(uciMoves, fields));
    } else {
      const afterPass = applyMove(position, pass!);
      panel.showHint('threat', afterPass, await serverSearch([...uciMoves, moveToUci(pass!, position)], fields));
    }
  } catch (error) {
    aiInfoEl.textContent = `Server engine unavailable (${String(error)}); is training/play_server.py running?`;
  } finally {
    hintBusy = false;
    panel.setWaiting(false);
    updateStatus();
  }
}

async function pickAiMove(current: Position): Promise<Move> {
  try {
    return await pickServerMove(current);
  } catch (error) {
    aiInfoEl.textContent = `Server engine unavailable (${String(error)}); is training/play_server.py running?`;
    throw error;
  }
}

function maybeScheduleAiMove(): void {
  if (gameOver || position.sideToMove === humanSide) return;
  aiTimer = setTimeout(async () => {
    panel.setWaiting(true);
    const choice = await pickAiMove(position).finally(() => panel.setWaiting(false));
    if (gameOver) return; // resigned/finished while the model was thinking
    uciMoves.push(moveToUci(choice, position));
    position = applyMove(position, choice);
    afterMove(choice);
  }, AI_THINK_DELAY_MS);
}

function handleHumanMove(move: Move): void {
  if (gameOver || position.sideToMove !== humanSide) return;
  panel.noteOpponentMove(uciMoves.length, position, move);
  uciMoves.push(moveToUci(move, position));
  position = applyMove(position, move);
  afterMove(move);
}

function newGame(): void {
  clearTimeout(aiTimer);
  gameOver = false;
  humanSide = humanSideSelect.value as Side;
  uciMoves = [];
  panel.reset(humanSide === 'cho' ? 'han' : 'cho');
  gameSetup = { cho: setupChoSelect.value as Formation, han: setupHanSelect.value as Formation };
  position = createInitialPosition({
    setupCho: setupChoSelect.value as Formation,
    setupHan: setupHanSelect.value as Formation,
  });
  boardEl.innerHTML = '';
  boardView = createBoardView(boardEl, position, {
    humanSide,
    onHumanMove: handleHumanMove,
    animationDurationMs: HUMAN_MOVE_MS,
  });
  updateStatus();
  maybeScheduleAiMove();
}

passBtn.addEventListener('click', () => {
  if (gameOver || position.sideToMove !== humanSide) return;
  const pass = generateLegalMoves(position).find((m) => m.isPass);
  if (!pass) return;
  handleHumanMove(pass);
});

resignBtn.addEventListener('click', () => {
  if (gameOver) return;
  clearTimeout(aiTimer);
  gameOver = true;
  const winner: Side = position.sideToMove === 'cho' ? 'han' : 'cho';
  statusEl.textContent = resultText({ winner, kind: 'resignation' });
  passBtn.disabled = true;
  hintBtn.disabled = true;
  threatBtn.disabled = true;
  panel.gameOver({ winner, kind: 'resignation' });
});

newGameBtn.addEventListener('click', newGame);
hintBtn.addEventListener('click', () => void showHint('hint'));
threatBtn.addEventListener('click', () => void showHint('threat'));
humanSideSelect.addEventListener('change', newGame);

newGame();

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
  if (view === 'about' && !rendered.has(view)) renderAbout(el);
  if (view === 'rules' && !rendered.has(view)) renderRules(el);
  if (view === 'progress') void renderProgress(el); // live data: refresh on every visit
  rendered.add(view);
  if (view !== 'play') window.scrollTo(0, 0);
}

window.addEventListener('hashchange', showView);
showView();
