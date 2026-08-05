import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import {
  applyMove,
  colOf,
  createInitialPosition,
  findGeneral,
  generateLegalMoves,
  getGameResult,
  positionKey,
  rowOf,
  BOARD_ROWS,
} from '@sinsan/rules';
import type { Formation, GameResult, Move, Position, RuleProfileId } from '@sinsan/rules';
import { encodeMove } from '@sinsan/action-space';
import { SinsanModel, positionToPlanes } from '@sinsan/model-runtime';
import type { ModelManifest } from '@sinsan/model-runtime';
import { SearchTree } from '@sinsan/search';

// Headless paired-match runner (docs/ARCHITECTURE.md's apps/arena, docs/BENCHMARK_PLAN.md
// Section 18.3's "paired fixed-opening matches"). Calls SinsanModel.infer() directly - no
// Worker/fetch boundary, since that requirement (Section 2) is specifically about the shipped
// browser artifact, and this tool never ships to the browser.
//
// KNOWN LIMITATION (docs/BENCHMARK_PLAN.md Phase 6 section): `policy` and `search` players are
// fully deterministic (no exploration noise), and there are only 16 possible (setupCho, setupHan)
// formation combos - so a --pairs count above ~16 starts producing exact-duplicate pairs (same
// opening drawn twice, byte-identical games), not new independent data. A 40-game run confirmed
// this directly: only 11 distinct outcome signatures across 20 pairs. Player specs now log their
// formation combo per pair so this is visible rather than silent. Not yet fixed - would need
// opening-move randomization or temperature-based exploration in the players themselves, not just
// more --pairs.

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const STOCKFISH_PATH = join(repoRoot, 'training', 'teacher', 'engine', 'Fairy-Stockfish', 'src', 'stockfish');
// Maps our RuleProfileId onto Fairy-Stockfish's own variant names (docs/RULES.md, confirmed
// identical during Phase 0/3 research) - same mapping tests/differential/generate-cases.ts uses.
const RULE_PROFILE_TO_VARIANT: Record<RuleProfileId, string> = {
  kja: 'janggi',
  traditional: 'janggitraditional',
  modern: 'janggimodern',
  casual: 'janggicasual',
};

function squareToUci(square: number): string {
  const file = String.fromCharCode('a'.charCodeAt(0) + colOf(square));
  const rank = BOARD_ROWS - rowOf(square);
  return `${file}${rank}`;
}

/** Pass is UCI-encoded as the mover's own general square, doubled - confirmed against pyffish
 * directly during Phase 1 (tests/differential/generate-cases.ts uses the same convention). */
function moveToUci(position: Position, move: Move): string {
  if (move.isPass) {
    const generalSquare = findGeneral(position, position.sideToMove);
    const sq = squareToUci(generalSquare);
    return sq + sq;
  }
  return squareToUci(move.from) + squareToUci(move.to);
}

function fullFen(position: Position): string {
  return `${positionKey(position)} - - ${position.noCapturePly} ${position.moveNumber}`;
}

/** Drives a Fairy-Stockfish subprocess over UCI (same protocol training/teacher/adapter.py uses
 * to label training data, ported to TS so apps/arena can use the real engine as an opponent, not
 * just Sinsan's own checkpoints - Section 18.3's "vs restricted Fairy-Stockfish" desideratum).
 *
 * HONESTY CAVEAT (see docs/BENCHMARK_PLAN.md): `UCI_Elo`/`UCI_LimitStrength` are Stockfish-family
 * mechanisms calibrated against CHESS self-play data. Whether a given UCI_Elo value corresponds
 * to the same real-world strength in the Janggi variant is not verified anywhere - there is no
 * known authoritative Janggi engine-strength-to-Elo (let alone Elo-to-Korean-dan) conversion this
 * project has access to. Results below are reported as "vs Fairy-Stockfish at UCI_Elo=N", not
 * translated into any dan-rank or absolute-strength claim. */
class StockfishEngine {
  private readonly proc: ReturnType<typeof spawn>;
  private lineBuffer = '';
  private readonly pendingLines: string[] = [];
  private readonly waiters: Array<() => void> = [];

  constructor() {
    this.proc = spawn(STOCKFISH_PATH, [], { stdio: ['pipe', 'pipe', 'pipe'] });
    this.proc.stdout!.on('data', (chunk: Buffer) => {
      this.lineBuffer += chunk.toString();
      let idx: number;
      while ((idx = this.lineBuffer.indexOf('\n')) !== -1) {
        this.pendingLines.push(this.lineBuffer.slice(0, idx).replace(/\r$/, ''));
        this.lineBuffer = this.lineBuffer.slice(idx + 1);
        this.waiters.shift()?.();
      }
    });
  }

  private send(command: string): void {
    this.proc.stdin!.write(command + '\n');
  }

  private async readUntil(marker: string): Promise<string[]> {
    const collected: string[] = [];
    for (;;) {
      while (this.pendingLines.length === 0) {
        await new Promise<void>((resolve) => this.waiters.push(resolve));
      }
      const line = this.pendingLines.shift()!;
      collected.push(line);
      if (line.includes(marker)) return collected;
    }
  }

  async init(variant: string, elo: number | undefined): Promise<void> {
    this.send('uci');
    await this.readUntil('uciok');
    this.send(`setoption name UCI_Variant value ${variant}`);
    this.send('setoption name MultiPV value 1');
    this.send('setoption name Threads value 1');
    if (elo !== undefined) {
      this.send('setoption name UCI_LimitStrength value true');
      this.send(`setoption name UCI_Elo value ${elo}`);
    }
    this.send('isready');
    await this.readUntil('readyok');
  }

  async bestMoveUci(fen: string, movetimeMs: number): Promise<string> {
    this.send(`position fen ${fen}`);
    this.send(`go movetime ${movetimeMs}`);
    const lines = await this.readUntil('bestmove');
    const bestLine = lines.find((l) => l.startsWith('bestmove'))!;
    return bestLine.split(' ')[1]!;
  }

  quit(): void {
    this.send('quit');
    this.proc.kill();
  }
}

const FORMATIONS: Formation[] = ['masang-sangma', 'sangma-masang', 'masang-masang', 'sangma-sangma'];
const MAX_PLIES = 400; // safety net only - real games should end via checkmate/bikjang/repetition/
// draw adjudication well before this; hitting it means one of those is broken and is reported as
// a distinct outcome below, never silently folded into win/loss/draw counts.

interface Player {
  readonly label: string;
  pickMove(position: Position): Promise<Move> | Move;
  readonly cleanup?: () => void;
}

const modelCache = new Map<string, SinsanModel>();

function loadModel(modelName: string): SinsanModel {
  let model = modelCache.get(modelName);
  if (!model) {
    const manifest: ModelManifest = JSON.parse(
      readFileSync(join(repoRoot, 'public', 'model', `${modelName}.json`), 'utf8'),
    );
    const weightsBuffer = new Uint8Array(
      readFileSync(join(repoRoot, 'public', 'model', `${modelName}.bin`)),
    ).buffer;
    model = new SinsanModel(manifest, weightsBuffer);
    modelCache.set(modelName, model);
  }
  return model;
}

function pickRandomMove(position: Position): Move {
  const legalMoves = generateLegalMoves(position);
  const nonPassMoves = legalMoves.filter((m) => !m.isPass);
  return nonPassMoves.length > 0 ? nonPassMoves[Math.floor(Math.random() * nonPassMoves.length)]! : legalMoves[0]!;
}

function pickPolicyMove(position: Position, model: SinsanModel): Move {
  const legalMoves = generateLegalMoves(position);
  const legalByAction = new Map<number, Move>();
  for (const move of legalMoves) {
    const actionId = encodeMove(move);
    if (actionId !== undefined) legalByAction.set(actionId, move);
  }
  const { policyLogits } = model.infer(positionToPlanes(position));
  let bestAction: number | undefined;
  let bestLogit = -Infinity;
  for (const actionId of legalByAction.keys()) {
    const logit = policyLogits[actionId]!;
    if (logit > bestLogit) {
      bestLogit = logit;
      bestAction = actionId;
    }
  }
  return bestAction !== undefined ? legalByAction.get(bestAction)! : legalMoves[0]!;
}

/** Player spec grammar: "random" | "policy:<modelName>" | "search:<modelName>:<visits>" |
 * "stockfish:<elo|full>[:<movetimeMs>]" (elo omitted/= "full" runs unrestricted Fairy-Stockfish;
 * movetimeMs defaults to 100). A fresh Player is created per game (not shared across games) so a
 * search player's tree - or a stockfish player's subprocess - never leaks state between
 * independent games. */
function createPlayer(spec: string, ruleProfile: RuleProfileId): Player {
  const parts = spec.split(':');
  const kind = parts[0];
  if (kind === 'random') {
    return { label: 'random', pickMove: pickRandomMove };
  }
  if (kind === 'stockfish') {
    const eloText = parts[1];
    const elo = eloText === undefined || eloText === 'full' ? undefined : Number(eloText);
    const movetimeMs = Number(parts[2] ?? 100);
    const engine = new StockfishEngine();
    const ready = engine.init(RULE_PROFILE_TO_VARIANT[ruleProfile], elo);
    return {
      label: elo !== undefined ? `stockfish:elo${elo}` : 'stockfish:full',
      pickMove: async (position) => {
        await ready;
        const uci = await engine.bestMoveUci(fullFen(position), movetimeMs);
        const legal = generateLegalMoves(position);
        const match = legal.find((m) => moveToUci(position, m) === uci);
        if (!match) {
          throw new Error(`stockfish returned "${uci}", not in our legal move list for ${fullFen(position)}`);
        }
        return match;
      },
      cleanup: () => engine.quit(),
    };
  }
  if (kind === 'policy') {
    const modelName = parts[1];
    if (!modelName) throw new Error(`policy player spec needs a model name: "${spec}"`);
    const model = loadModel(modelName);
    return { label: `policy:${modelName}`, pickMove: (position) => pickPolicyMove(position, model) };
  }
  if (kind === 'search') {
    const modelName = parts[1];
    const visits = Number(parts[2]);
    if (!modelName || !Number.isFinite(visits) || visits <= 0) {
      throw new Error(`search player spec needs a model name and visit count: "${spec}"`);
    }
    const model = loadModel(modelName);
    const tree = new SearchTree((position) => model.infer(positionToPlanes(position)));
    return {
      label: `search:${modelName}:${visits}`,
      pickMove: async (position) => (await tree.getMove(position, visits)).move,
    };
  }
  throw new Error(
    `unknown player spec: "${spec}" (expected random | policy:<model> | search:<model>:<visits> | stockfish:<elo|full>[:<movetimeMs>])`,
  );
}

interface GameOutcome {
  readonly result: GameResult | null; // null means MAX_PLIES was hit without a real result
  readonly plies: number;
}

/** Plays `randomPlies` uniform-random legal moves from the formation's starting position, shared
 * by both games in a pair (see the KNOWN LIMITATION note above) - this is what actually fixes the
 * redundancy, not the formation choice alone: 16 formation combos x many possible random
 * continuations gives far more effectively-distinct starting positions than 16 alone, while
 * keeping the players under test themselves deterministic (so a decisive result means the engine
 * actually won from that position, not that noise happened to break a tie). */
function randomOpeningPosition(
  setupCho: Formation,
  setupHan: Formation,
  ruleProfile: RuleProfileId,
  randomPlies: number,
): Position {
  let position = createInitialPosition({ ruleProfile, setupCho, setupHan });
  for (let i = 0; i < randomPlies && !getGameResult(position); i++) {
    const legal = generateLegalMoves(position);
    const nonPass = legal.filter((m) => !m.isPass);
    const pool = nonPass.length > 0 ? nonPass : legal;
    position = applyMove(position, pool[Math.floor(Math.random() * pool.length)]!);
  }
  return position;
}

function playGame(playerCho: Player, playerHan: Player, startPosition: Position): Promise<GameOutcome> {
  return (async () => {
    let position = startPosition;
    let plies = 0;
    let result = getGameResult(position);
    while (!result && plies < MAX_PLIES) {
      const mover = position.sideToMove === 'cho' ? playerCho : playerHan;
      const move = await mover.pickMove(position);
      position = applyMove(position, move);
      plies++;
      result = getGameResult(position);
    }
    return { result, plies };
  })();
}

interface Tally {
  wins: number;
  losses: number;
  draws: number;
  unresolved: number;
}

function main(): void {
  const args = process.argv.slice(2);
  const flags = new Map<string, string>();
  for (let i = 0; i < args.length; i += 2) flags.set(args[i]!.replace(/^--/, ''), args[i + 1]!);

  const specA = flags.get('a') ?? 'search:sinsan-baseline-v0:16';
  const specB = flags.get('b') ?? 'random';
  const pairs = Number(flags.get('pairs') ?? 10);
  const ruleProfile = (flags.get('rule-profile') ?? 'kja') as RuleProfileId;
  // Default 4, not 0: with deterministic players and only 16 formation combos, --pairs above ~16
  // starts producing exact-duplicate games without this (docs/BENCHMARK_PLAN.md Phase 6 section,
  // confirmed directly - a 40-game run had only 11 distinct outcomes across 20 pairs).
  const randomPlies = Number(flags.get('random-plies') ?? 4);

  console.log(`Arena: A="${specA}" vs B="${specB}", ${pairs} paired games (${pairs * 2} total), rule profile "${ruleProfile}"`);
  console.log(
    `Each pair plays a formation combo plus ${randomPlies} random opening plies (shared by both games in the pair) ` +
      'twice, colors reversed, to cancel first-move/formation bias.\n',
  );

  const tally: Tally = { wins: 0, losses: 0, draws: 0, unresolved: 0 };

  void (async () => {
    for (let i = 0; i < pairs; i++) {
      const setupCho = FORMATIONS[Math.floor(Math.random() * FORMATIONS.length)]!;
      const setupHan = FORMATIONS[Math.floor(Math.random() * FORMATIONS.length)]!;
      const openingPosition = randomOpeningPosition(setupCho, setupHan, ruleProfile, randomPlies);

      const playerA1 = createPlayer(specA, ruleProfile);
      const playerB1 = createPlayer(specB, ruleProfile);
      const game1 = await playGame(playerA1, playerB1, openingPosition);
      playerA1.cleanup?.();
      playerB1.cleanup?.();
      recordOutcome(tally, game1, 'cho'); // A played Cho

      const playerA2 = createPlayer(specA, ruleProfile);
      const playerB2 = createPlayer(specB, ruleProfile);
      const game2 = await playGame(playerB2, playerA2, openingPosition);
      playerA2.cleanup?.();
      playerB2.cleanup?.();
      recordOutcome(tally, game2, 'han'); // A played Han

      const played = (i + 1) * 2;
      console.log(
        `[${played}/${pairs * 2}] pair ${i + 1} (${setupCho} vs ${setupHan}): ` +
          `game1 ${describeOutcome(game1)} | game2 ${describeOutcome(game2)}`,
      );
    }

    const decisive = tally.wins + tally.losses;
    const total = tally.wins + tally.losses + tally.draws + tally.unresolved;
    console.log(`\n=== Result: A="${specA}" vs B="${specB}" ===`);
    console.log(`A wins: ${tally.wins}  B wins: ${tally.losses}  draws: ${tally.draws}  unresolved (hit ${MAX_PLIES}-ply cap): ${tally.unresolved}`);
    if (tally.unresolved > 0) {
      console.log(
        `WARNING: ${tally.unresolved} game(s) hit the ${MAX_PLIES}-ply safety cap without a rules-engine result - ` +
          'this means checkmate/bikjang/repetition/draw adjudication failed to end a real game and is worth investigating, not just noise.',
      );
    }
    if (decisive > 0) {
      const scoreFraction = (tally.wins + tally.draws * 0.5) / total;
      // Standard logistic Elo-difference-from-score-fraction formula - included as a familiar
      // unit, not a calibrated rating. With `pairs` games this small, treat the number as
      // directional only; docs/BENCHMARK_PLAN.md Section 18.3 still wants a real Elo pipeline
      // (many more games, proper confidence interval) before this is reported as a rating claim.
      const clamped = Math.min(0.99, Math.max(0.01, scoreFraction));
      const eloDiff = -400 * Math.log10(1 / clamped - 1);
      console.log(
        `A's score fraction: ${(scoreFraction * 100).toFixed(1)}% over ${total} games -> ` +
          `rough Elo diff estimate ${eloDiff >= 0 ? '+' : ''}${eloDiff.toFixed(0)} ` +
          `(small-sample estimate, not a calibrated rating - see Section 18.3 caveat above).`,
      );
    } else {
      console.log("All games drew or were unresolved - no decisive games to estimate a score fraction from.");
    }
  })();
}

function recordOutcome(tally: Tally, outcome: GameOutcome, aSide: 'cho' | 'han'): void {
  if (!outcome.result) {
    tally.unresolved++;
    return;
  }
  if (outcome.result.winner === null) {
    tally.draws++;
  } else if (outcome.result.winner === aSide) {
    tally.wins++;
  } else {
    tally.losses++;
  }
}

function describeOutcome(outcome: GameOutcome): string {
  if (!outcome.result) return `unresolved (${outcome.plies} plies, hit cap)`;
  const winner = outcome.result.winner ?? 'draw';
  return `${outcome.result.kind}, winner=${winner} (${outcome.plies} plies)`;
}

main();
