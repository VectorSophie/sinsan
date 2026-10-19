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
import type { ServerAnalysis } from './analysis-panel.ts';
import type { BoardView } from '@sinsan/ui';
import { encodeMove } from '@sinsan/action-space';
import { createModelWorkerClient, positionToPlanes } from '@sinsan/model-runtime';
import type { ModelWorkerClient } from '@sinsan/model-runtime';
import { SearchTree } from '@sinsan/search';
import type { PositionEvaluation } from '@sinsan/search';

// Section 16's visit tiers. 16 stays responsive for a live demo; 64 ("Advanced") is real search
// depth for stronger play at the cost of being slow (each visit is a model call - expect roughly
// 4x a 16-visit move's latency, which was ~10.5s for the baseline model at time of writing per
// docs/BENCHMARK_PLAN.md, so 64 visits is closer to 40s/move - offered anyway since search depth
// is the single biggest lever on real playing strength this project has, and hiding it isn't
// honest just because it's slow. 128 ("Sinsan") is still not wired in - even slower, unverified
// whether it's worth the wait over 64.
const SEARCH_TIERS = [16, 64] as const;

const MODEL_VARIANTS = {
  smoke: { modelName: 'sinsan-smoke-v0', label: 'Smoke' },
  baseline: { modelName: 'sinsan-baseline-v0', label: 'Baseline' },
  v2: { modelName: 'sinsan-v2-56x7', label: 'V2' },
  v3: { modelName: 'sinsan-v3-56x7', label: 'V3' },
} as const;
type ModelVariant = keyof typeof MODEL_VARIANTS;

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
  <h1 class="sinsan-title">神算 Sinsan</h1>
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
    <label>AI
      <select id="ai-type">
        <option value="server" selected>Sinsan 2026 (GPU server - see settings)</option>
        <option value="random">Random</option>
        ${(Object.keys(MODEL_VARIANTS) as ModelVariant[])
          .map((v) => `<option value="model:${v}">Sinsan (${MODEL_VARIANTS[v].label}, policy only)</option>`)
          .join('')}
        ${SEARCH_TIERS.flatMap((visits) =>
          (Object.keys(MODEL_VARIANTS) as ModelVariant[]).map(
            (v) =>
              `<option value="search:${visits}:${v}">` +
              `Sinsan (${MODEL_VARIANTS[v].label}, ${visits}-visit search)</option>`,
          ),
        ).join('')}
      </select>
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
`;

const boardEl = document.querySelector<HTMLDivElement>('#board')!;
const statusEl = document.querySelector<HTMLDivElement>('#status')!;
const humanSideSelect = document.querySelector<HTMLSelectElement>('#human-side')!;
const setupChoSelect = document.querySelector<HTMLSelectElement>('#setup-cho')!;
const setupHanSelect = document.querySelector<HTMLSelectElement>('#setup-han')!;
const aiTypeSelect = document.querySelector<HTMLSelectElement>('#ai-type')!;
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
  const canAsk = aiTypeSelect.value === 'server' && position.sideToMove === humanSide && !hintBusy;
  hintBtn.disabled = !canAsk;
  threatBtn.disabled = !canAsk || isCheck(position);
}

function afterMove(move: Move): void {
  const isAiMove = move && position.sideToMove !== humanSide;
  boardView.sync(position, move, isAiMove ? AI_MOVE_MS : HUMAN_MOVE_MS);
  updateStatus();
  maybeScheduleAiMove();
}

const modelClientPromises = new Map<ModelVariant, Promise<ModelWorkerClient>>();

/** Lazily creates the Worker-hosted model client for a given variant on first use - fetches the
 * manifest+weights on the main thread (with SHA-256 verification), then hands the buffer to a
 * dedicated Worker. All tensor computation happens in that Worker, never here (Section 2). Each
 * variant gets its own client/Worker, cached separately, so switching in the AI dropdown doesn't
 * re-fetch a variant already loaded this session. */
function getModelClient(variant: ModelVariant): Promise<ModelWorkerClient> {
  let promise = modelClientPromises.get(variant);
  if (!promise) {
    const { modelName } = MODEL_VARIANTS[variant];
    promise = createModelWorkerClient(
      new URL('./model-worker.ts', import.meta.url),
      `/model/${modelName}.json`,
      `/model/${modelName}.bin`,
    );
    modelClientPromises.set(variant, promise);
  }
  return promise;
}

/** Greedy policy play (Section 16's "Beginner: policy sampling, no tree search" mode) - no MCTS,
 * just the chosen variant's raw policy head masked to legal moves. This is Phase 4's Web Worker
 * inference deliverable proven live, not a strength claim (docs/BENCHMARK_PLAN.md). */
async function pickModelMove(current: Position, variant: ModelVariant): Promise<Move> {
  const legalMoves = generateLegalMoves(current);
  const legalByAction = new Map<number, Move>();
  for (const move of legalMoves) {
    const actionId = encodeMove(move);
    if (actionId !== undefined) legalByAction.set(actionId, move);
  }

  const client = await getModelClient(variant);
  const start = performance.now();
  const { policyLogits, value } = await client.infer(positionToPlanes(current));
  const elapsedMs = performance.now() - start;

  let bestAction: number | undefined;
  let bestLogit = -Infinity;
  for (const actionId of legalByAction.keys()) {
    const logit = policyLogits[actionId]!;
    if (logit > bestLogit) {
      bestLogit = logit;
      bestAction = actionId;
    }
  }

  aiInfoEl.textContent =
    `Sinsan (${MODEL_VARIANTS[variant].label}): value=${value.toFixed(2)}, inference ${elapsedMs.toFixed(1)}ms`;
  return bestAction !== undefined ? legalByAction.get(bestAction)! : legalMoves[0]!;
}

function modelEvaluator(variant: ModelVariant): (current: Position) => Promise<PositionEvaluation> {
  return async (current) => {
    const client = await getModelClient(variant);
    return client.infer(positionToPlanes(current));
  };
}

const searchTrees = new Map<ModelVariant, SearchTree>();

/** PUCT/MCTS play (Section 16) - reuses the same tree across the game's moves via SearchTree
 * (Phase 4's search deliverable), not a fresh tree every call. Kept per-variant so switching the
 * AI dropdown mid-session doesn't hand one variant's tree to another variant's evaluator. */
async function pickSearchMove(current: Position, variant: ModelVariant, visits: number): Promise<Move> {
  let searchTree = searchTrees.get(variant);
  if (!searchTree) {
    searchTree = new SearchTree(modelEvaluator(variant));
    searchTrees.set(variant, searchTree);
  }
  const start = performance.now();
  const result = await searchTree.getMove(current, visits);
  const elapsedMs = performance.now() - start;
  aiInfoEl.textContent =
    `Sinsan (${MODEL_VARIANTS[variant].label}, ${visits}-visit search): value=${result.rootValue.toFixed(2)}, ` +
    `${result.visitsUsed} visits used, ${elapsedMs.toFixed(0)}ms`;
  return result.move;
}

function pickRandomMove(current: Position): Move {
  const legalMoves = generateLegalMoves(current);
  const nonPassMoves = legalMoves.filter((m) => !m.isPass);
  aiInfoEl.textContent = '';
  return nonPassMoves.length > 0 ? nonPassMoves[Math.floor(Math.random() * nonPassMoves.length)]! : legalMoves[0]!;
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
  const parts = aiTypeSelect.value.split(':');
  if (parts[0] === 'server') {
    try {
      return await pickServerMove(current);
    } catch (error) {
      aiInfoEl.textContent = `Server engine unavailable (${String(error)}); is training/play_server.py running?`;
      throw error;
    }
  }
  if (parts[0] === 'model') {
    return pickModelMove(current, parts[1] as ModelVariant);
  }
  if (parts[0] === 'search') {
    return pickSearchMove(current, parts[2] as ModelVariant, Number(parts[1]));
  }
  return pickRandomMove(current);
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
  searchTrees.clear(); // a new game is a different game line - don't reuse old trees
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
aiTypeSelect.addEventListener('change', () => updateStatus());
humanSideSelect.addEventListener('change', newGame);

newGame();
