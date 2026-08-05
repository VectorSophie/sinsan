/**
 * Position generator (Phase 4 / Section 23 item 11): plays self-play games with @sinsan/rules and
 * writes the positions reached to JSONL for the teacher adapter to label.
 *
 * Two move-selection sources, chosen with --player:
 *   --player random (default)         - uniform-random legal move, source="self_play_random".
 *   --player policy:<model-name>      - the named exported model's own policy head, temperature-
 *                                        sampled over legal actions for game diversity (Phase 6's
 *                                        on-policy self-play), source="self_play_policy:<name>".
 * Loads the model directly (SinsanModel, no Worker/fetch) the same way apps/arena does - this
 * script never ships to the browser, so that boundary doesn't apply here.
 *
 * Search-guided self-play was considered and rejected for data generation at this project's
 * current compute budget: a single 16-visit search call on the 48x6 baseline model measures
 * ~10.5s (docs/BENCHMARK_PLAN.md), which would make even a modest dataset take many hours to
 * generate. Policy-only inference (~0.3-0.9s/call) is roughly 15-30x faster and is what makes an
 * on-policy dataset actually tractable to generate at 10K+ position scale in a single session.
 *
 * Run with: node training/generate/self-play.ts [targetPositionCount] [outputFilename]
 *   [--player random|policy:<model-name>] [--temperature <t>] [--max-plies <n>]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_RULE_PROFILE_ID,
  applyMove,
  createInitialPosition,
  generateLegalMoves,
  getGameResult,
  positionKey,
} from '../../packages/rules/src/index.ts';
import type { Formation, Move, Position } from '../../packages/rules/src/index.ts';
import { encodeMove } from '../../packages/action-space/src/index.ts';
import { SinsanModel, positionToPlanes } from '../../packages/model-runtime/src/index.ts';
import type { ModelManifest } from '../../packages/model-runtime/src/index.ts';

const FORMATIONS: Formation[] = ['masang-sangma', 'sangma-masang', 'masang-masang', 'sangma-sangma'];
const TARGET_COUNT = Number(process.argv[2] ?? 512);
const OUTPUT_FILENAME = process.argv[3] ?? 'smoke-positions.jsonl';

const flags = new Map<string, string>();
{
  const rest = process.argv.slice(4);
  for (let i = 0; i < rest.length; i += 2) flags.set(rest[i]!.replace(/^--/, ''), rest[i + 1]!);
}
const PLAYER_SPEC = flags.get('player') ?? 'random';
const TEMPERATURE = Number(flags.get('temperature') ?? 1.0);
const MAX_PLIES_PER_GAME = Number(flags.get('max-plies') ?? 60);

let seed = 20260731;
function nextRandom(): number {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
}
function pick<T>(arr: readonly T[]): T {
  return arr[Math.floor(nextRandom() * arr.length)]!;
}

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

function loadModel(modelName: string): SinsanModel {
  const manifest: ModelManifest = JSON.parse(
    readFileSync(join(repoRoot, 'public', 'model', `${modelName}.json`), 'utf8'),
  );
  const weightsBuffer = new Uint8Array(readFileSync(join(repoRoot, 'public', 'model', `${modelName}.bin`))).buffer;
  return new SinsanModel(manifest, weightsBuffer);
}

/** Temperature-sampled draw from legal moves using the model's own policy logits - not argmax,
 * so games stay diverse (an all-greedy self-play policy collapses to a handful of repeated
 * lines, which is a weak training set regardless of how good the policy itself is). Uses the same
 * seeded RNG as the random player for reproducible runs. */
function samplePolicyMove(position: Position, legal: Move[], model: SinsanModel): Move {
  const legalByAction = new Map<number, Move>();
  for (const move of legal) {
    const actionId = encodeMove(move);
    if (actionId !== undefined) legalByAction.set(actionId, move);
  }
  const { policyLogits } = model.infer(positionToPlanes(position));
  const actionIds = [...legalByAction.keys()];
  let maxLogit = -Infinity;
  for (const id of actionIds) maxLogit = Math.max(maxLogit, policyLogits[id]!);
  const weights = actionIds.map((id) => Math.exp((policyLogits[id]! - maxLogit) / TEMPERATURE));
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  let threshold = nextRandom() * totalWeight;
  for (let i = 0; i < actionIds.length; i++) {
    threshold -= weights[i]!;
    if (threshold <= 0) return legalByAction.get(actionIds[i]!)!;
  }
  return legalByAction.get(actionIds[actionIds.length - 1]!)!;
}

const policyModelName = PLAYER_SPEC.startsWith('policy:') ? PLAYER_SPEC.slice('policy:'.length) : undefined;
const policyModel = policyModelName ? loadModel(policyModelName) : undefined;
const sourceLabel = policyModel ? `self_play_policy:${policyModelName}` : 'self_play_random';
if (policyModel) {
  console.log(`Using policy-guided self-play: model=${policyModelName}, temperature=${TEMPERATURE}, max-plies=${MAX_PLIES_PER_GAME}`);
}

function fullFen(position: Position): string {
  // Standard 6-field FEN the teacher engine expects: <board> <side> - - <halfmove> <fullmove>.
  // Janggi has no castling/en-passant, hence the fixed "- -".
  return `${positionKey(position)} - - ${position.noCapturePly} ${position.moveNumber}`;
}

interface Record_ {
  fen: string;
  game_id: number;
  ply: number;
  setup_cho: Formation;
  setup_han: Formation;
  rule_profile: string;
  source: string;
  // Computed here (not in Python) because @sinsan/rules is the legality authority - Python has
  // no independent Janggi rules engine, so re-deriving this would mean trusting a second
  // implementation instead of the one already tested against 45+7 unit/fixture cases.
  legal_actions: number[];
}

const records: Record_[] = [];
let gameId = 0;

while (records.length < TARGET_COUNT) {
  const setupCho = pick(FORMATIONS);
  const setupHan = pick(FORMATIONS);
  let position: Position = createInitialPosition({ setupCho, setupHan, ruleProfile: DEFAULT_RULE_PROFILE_ID });

  for (let ply = 0; ply < MAX_PLIES_PER_GAME && records.length < TARGET_COUNT; ply++) {
    // Stop on ANY terminal condition (checkmate, bikjang, repetition, material adjudication) -
    // not just "no legal moves". A position can be game-over via getGameResult() while still
    // having legal moves generated for it (e.g. bikjang), and continuing past that point produced
    // positions Fairy-Stockfish itself refuses to search normally (a real bug found during Phase 4:
    // the teacher returned a degenerate sentinel response for an already-drawn bikjang position,
    // which corrupted training - see training/datasets/ regeneration notes).
    if (getGameResult(position)) break;
    const legal = generateLegalMoves(position);

    records.push({
      fen: fullFen(position),
      game_id: gameId,
      ply,
      setup_cho: setupCho,
      setup_han: setupHan,
      rule_profile: position.ruleProfile,
      source: sourceLabel,
      legal_actions: legal.map((m) => encodeMove(m)!),
    });

    let nextMove: Move;
    if (policyModel) {
      nextMove = samplePolicyMove(position, legal, policyModel);
    } else {
      const nonPass = legal.filter((m) => !m.isPass);
      const pool = nonPass.length > 0 ? nonPass : legal;
      nextMove = pool[Math.floor(nextRandom() * pool.length)]!;
    }
    position = applyMove(position, nextMove);
  }
  gameId++;
  if (policyModel && gameId % 200 === 0) {
    console.log(`... ${records.length}/${TARGET_COUNT} positions from ${gameId} games`);
  }
}

const outPath = join(dirname(fileURLToPath(import.meta.url)), '..', 'datasets', OUTPUT_FILENAME);
writeFileSync(outPath, records.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`Wrote ${records.length} positions from ${gameId} games to ${outPath}`);
