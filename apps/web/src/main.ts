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
import type { BoardView } from '@sinsan/ui';
import { encodeMove } from '@sinsan/action-space';
import { createModelWorkerClient, positionToPlanes } from '@sinsan/model-runtime';
import type { ModelWorkerClient } from '@sinsan/model-runtime';
import { SearchTree } from '@sinsan/search';
import type { PositionEvaluation } from '@sinsan/search';

const SEARCH_VISITS = 16; // Section 16's "Intermediate: 16 visits" tier - kept small here for a
// responsive live demo; 64/128 ("Advanced"/"Sinsan") are supported by the same API but not wired
// into this UI given the smoke model's current ~60-100ms/inference cost (docs/BENCHMARK_PLAN.md).

const FORMATIONS: Formation[] = ['masang-sangma', 'sangma-masang', 'masang-masang', 'sangma-sangma'];
const HUMAN_MOVE_MS = 180;
const AI_MOVE_MS = 230;
const AI_THINK_DELAY_MS = 350;

const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <h1 class="sinsan-title">神算 Sinsan</h1>
  <div class="sinsan-controls">
    <label>Play as
      <select id="human-side">
        <option value="cho">Cho (블루)</option>
        <option value="han">Han (레드)</option>
      </select>
    </label>
    <label>Cho formation
      <select id="setup-cho">${FORMATIONS.map((f) => `<option value="${f}">${f}</option>`).join('')}</select>
    </label>
    <label>Han formation
      <select id="setup-han">${FORMATIONS.map((f) => `<option value="${f}">${f}</option>`).join('')}</select>
    </label>
    <label>AI
      <select id="ai-type">
        <option value="random">Random (smoke baseline)</option>
        <option value="model">Sinsan (policy only, no search)</option>
        <option value="search16">Sinsan (${SEARCH_VISITS}-visit search)</option>
      </select>
    </label>
  </div>
  <div id="board"></div>
  <div class="sinsan-controls">
    <button id="pass-btn">Pass</button>
    <button id="resign-btn">Resign</button>
    <button id="new-game-btn">New Game</button>
  </div>
  <div class="sinsan-status" id="status"></div>
  <div class="sinsan-status" id="ai-info"></div>
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

let position: Position;
let humanSide: Side = 'cho';
let boardView: BoardView;
let gameOver = false;
let aiTimer: ReturnType<typeof setTimeout> | undefined;

function resultText(result: GameResult): string {
  const winnerName = result.winner === 'cho' ? 'Cho' : result.winner === 'han' ? 'Han' : null;
  switch (result.kind) {
    case 'checkmate':
      return `Checkmate - ${winnerName} wins.`;
    case 'bikjang':
      return 'Bikjang - draw.';
    case 'repetition':
      return 'Repetition - draw.';
    case 'draw':
      return 'Draw (no-capture limit reached).';
    case 'material-adjudication':
      return `Material adjudication - ${winnerName} wins.`;
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
    return;
  }
  const turnName = position.sideToMove === 'cho' ? 'Cho' : 'Han';
  const checkNote = isCheck(position) ? ' - check!' : '';
  statusEl.textContent = `${turnName} to move${checkNote}`;
  passBtn.disabled = position.sideToMove !== humanSide || isCheck(position);
}

function afterMove(move: Move): void {
  const isAiMove = move && position.sideToMove !== humanSide;
  boardView.sync(position, move, isAiMove ? AI_MOVE_MS : HUMAN_MOVE_MS);
  updateStatus();
  maybeScheduleAiMove();
}

let modelClientPromise: Promise<ModelWorkerClient> | undefined;

/** Lazily creates the Worker-hosted model client on first use - fetches the manifest+weights on
 * the main thread (with SHA-256 verification), then hands the buffer to a dedicated Worker. All
 * tensor computation happens in that Worker, never here (Section 2). */
function getModelClient(): Promise<ModelWorkerClient> {
  if (!modelClientPromise) {
    modelClientPromise = createModelWorkerClient(
      new URL('./model-worker.ts', import.meta.url),
      '/model/sinsan-smoke-v0.json',
      '/model/sinsan-smoke-v0.bin',
    );
  }
  return modelClientPromise;
}

/** Greedy policy play (Section 16's "Beginner: policy sampling, no tree search" mode) - no MCTS,
 * just the smoke model's raw policy head masked to legal moves. This is Phase 4's Web Worker
 * inference deliverable proven live, not a strength claim (docs/BENCHMARK_PLAN.md). */
async function pickModelMove(current: Position): Promise<Move> {
  const legalMoves = generateLegalMoves(current);
  const legalByAction = new Map<number, Move>();
  for (const move of legalMoves) {
    const actionId = encodeMove(move);
    if (actionId !== undefined) legalByAction.set(actionId, move);
  }

  const client = await getModelClient();
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

  aiInfoEl.textContent = `Sinsan (smoke model): value=${value.toFixed(2)}, inference ${elapsedMs.toFixed(1)}ms`;
  return bestAction !== undefined ? legalByAction.get(bestAction)! : legalMoves[0]!;
}

async function modelEvaluator(current: Position): Promise<PositionEvaluation> {
  const client = await getModelClient();
  return client.infer(positionToPlanes(current));
}

let searchTree: SearchTree | undefined;

/** PUCT/MCTS play (Section 16) - reuses the same tree across the game's moves via SearchTree
 * (Phase 4's search deliverable), not a fresh tree every call. */
async function pickSearchMove(current: Position): Promise<Move> {
  if (!searchTree) searchTree = new SearchTree(modelEvaluator);
  const start = performance.now();
  const result = await searchTree.getMove(current, SEARCH_VISITS);
  const elapsedMs = performance.now() - start;
  aiInfoEl.textContent =
    `Sinsan (${SEARCH_VISITS}-visit search): value=${result.rootValue.toFixed(2)}, ` +
    `${result.visitsUsed} visits used, ${elapsedMs.toFixed(0)}ms`;
  return result.move;
}

function pickRandomMove(current: Position): Move {
  const legalMoves = generateLegalMoves(current);
  const nonPassMoves = legalMoves.filter((m) => !m.isPass);
  aiInfoEl.textContent = '';
  return nonPassMoves.length > 0 ? nonPassMoves[Math.floor(Math.random() * nonPassMoves.length)]! : legalMoves[0]!;
}

async function pickAiMove(current: Position): Promise<Move> {
  switch (aiTypeSelect.value) {
    case 'model':
      return pickModelMove(current);
    case 'search16':
      return pickSearchMove(current);
    default:
      return pickRandomMove(current);
  }
}

function maybeScheduleAiMove(): void {
  if (gameOver || position.sideToMove === humanSide) return;
  aiTimer = setTimeout(async () => {
    const choice = await pickAiMove(position);
    if (gameOver) return; // resigned/finished while the model was thinking
    position = applyMove(position, choice);
    afterMove(choice);
  }, AI_THINK_DELAY_MS);
}

function handleHumanMove(move: Move): void {
  if (gameOver || position.sideToMove !== humanSide) return;
  position = applyMove(position, move);
  afterMove(move);
}

function newGame(): void {
  clearTimeout(aiTimer);
  gameOver = false;
  searchTree = undefined; // a new game is a different game line - don't reuse the old tree
  humanSide = humanSideSelect.value as Side;
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
});

newGameBtn.addEventListener('click', newGame);
humanSideSelect.addEventListener('change', newGame);

newGame();
