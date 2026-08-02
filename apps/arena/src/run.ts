import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  applyMove,
  createInitialPosition,
  generateLegalMoves,
  getGameResult,
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

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const FORMATIONS: Formation[] = ['masang-sangma', 'sangma-masang', 'masang-masang', 'sangma-sangma'];
const MAX_PLIES = 400; // safety net only - real games should end via checkmate/bikjang/repetition/
// draw adjudication well before this; hitting it means one of those is broken and is reported as
// a distinct outcome below, never silently folded into win/loss/draw counts.

interface Player {
  readonly label: string;
  pickMove(position: Position): Promise<Move> | Move;
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

/** Player spec grammar: "random" | "policy:<modelName>" | "search:<modelName>:<visits>". A fresh
 * Player is created per game (not shared across games) so a search player's tree never leaks
 * state between independent games. */
function createPlayer(spec: string): Player {
  const parts = spec.split(':');
  const kind = parts[0];
  if (kind === 'random') {
    return { label: 'random', pickMove: pickRandomMove };
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
  throw new Error(`unknown player spec: "${spec}" (expected random | policy:<model> | search:<model>:<visits>)`);
}

interface GameOutcome {
  readonly result: GameResult | null; // null means MAX_PLIES was hit without a real result
  readonly plies: number;
}

function playGame(
  playerCho: Player,
  playerHan: Player,
  setupCho: Formation,
  setupHan: Formation,
  ruleProfile: RuleProfileId,
): Promise<GameOutcome> {
  return (async () => {
    let position = createInitialPosition({ ruleProfile, setupCho, setupHan });
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

  console.log(`Arena: A="${specA}" vs B="${specB}", ${pairs} paired games (${pairs * 2} total), rule profile "${ruleProfile}"`);
  console.log('Each pair plays one fixed random opening (formation combo) twice, colors reversed, to cancel first-move/formation bias.\n');

  const tally: Tally = { wins: 0, losses: 0, draws: 0, unresolved: 0 };

  void (async () => {
    for (let i = 0; i < pairs; i++) {
      const setupCho = FORMATIONS[Math.floor(Math.random() * FORMATIONS.length)]!;
      const setupHan = FORMATIONS[Math.floor(Math.random() * FORMATIONS.length)]!;

      const playerA1 = createPlayer(specA);
      const playerB1 = createPlayer(specB);
      const game1 = await playGame(playerA1, playerB1, setupCho, setupHan, ruleProfile);
      recordOutcome(tally, game1, 'cho'); // A played Cho

      const playerA2 = createPlayer(specA);
      const playerB2 = createPlayer(specB);
      const game2 = await playGame(playerB2, playerA2, setupCho, setupHan, ruleProfile);
      recordOutcome(tally, game2, 'han'); // A played Han

      const played = (i + 1) * 2;
      console.log(
        `[${played}/${pairs * 2}] pair ${i + 1}: game1 ${describeOutcome(game1)} | game2 ${describeOutcome(game2)}`,
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
