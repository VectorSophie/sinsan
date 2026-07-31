/**
 * Smoke-scale position generator (Phase 4 / Section 23 item 11): plays random-legal-move games
 * with @sinsan/rules and writes the positions reached to JSONL for the teacher adapter to label.
 *
 * This is explicitly the "self_play_random" smoke source only - not the full Section 12.3 data
 * mixture (teacher self-play, on-policy rollout, tactical/endgame curation, licensed human games).
 * It exists to validate the generate -> label -> train -> export -> infer pipeline end to end, per
 * Phase 4's own stated purpose - not to produce a strength-representative dataset.
 *
 * Run with: node training/generate/self-play.ts [targetPositionCount]
 */
import { writeFileSync } from 'node:fs';
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
import type { Formation, Position } from '../../packages/rules/src/index.ts';
import { encodeMove } from '../../packages/action-space/src/index.ts';

const FORMATIONS: Formation[] = ['masang-sangma', 'sangma-masang', 'masang-masang', 'sangma-sangma'];
const TARGET_COUNT = Number(process.argv[2] ?? 512);
const MAX_PLIES_PER_GAME = 60;

let seed = 20260731;
function nextRandom(): number {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
}
function pick<T>(arr: readonly T[]): T {
  return arr[Math.floor(nextRandom() * arr.length)]!;
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
      source: 'self_play_random',
      legal_actions: legal.map((m) => encodeMove(m)!),
    });

    const nonPass = legal.filter((m) => !m.isPass);
    const pool = nonPass.length > 0 ? nonPass : legal;
    position = applyMove(position, pool[Math.floor(nextRandom() * pool.length)]!);
  }
  gameId++;
}

const outPath = join(dirname(fileURLToPath(import.meta.url)), '..', 'datasets', 'smoke-positions.jsonl');
writeFileSync(outPath, records.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`Wrote ${records.length} positions from ${gameId} games to ${outPath}`);
