/**
 * Differential-testing case generator (Section 8.4): plays random-legal-move games across all
 * four RuleProfiles with @sinsan/rules, and for each position reached, records the FEN plus the
 * full legal-move set encoded as UCI strings - ready for tests/differential/compare.py to check
 * against pyffish's own legal_moves() for the same position and variant.
 *
 * Run with: node tests/differential/generate-cases.ts [count]
 */
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  BOARD_ROWS,
  applyMove,
  colOf,
  createInitialPosition,
  findGeneral,
  generateLegalMoves,
  positionKey,
  rowOf,
} from '../../packages/rules/src/index.ts';
import type { Formation, Move, Position, RuleProfileId } from '../../packages/rules/src/index.ts';

const FORMATIONS: Formation[] = ['masang-sangma', 'sangma-masang', 'masang-masang', 'sangma-sangma'];
// Maps our RuleProfileId directly onto pyffish/Fairy-Stockfish's own variant names (confirmed
// identical during Phase 0/3 research - see docs/RULES.md).
const RULE_PROFILE_TO_VARIANT: Record<RuleProfileId, string> = {
  kja: 'janggi',
  traditional: 'janggitraditional',
  modern: 'janggimodern',
  casual: 'janggicasual',
};
const RULE_PROFILES: RuleProfileId[] = ['kja', 'traditional', 'modern', 'casual'];

const TARGET_COUNT = Number(process.argv[2] ?? 400);
const MAX_PLIES_PER_GAME = 50;

let seed = 987654321;
function nextRandom(): number {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
}
function pick<T>(arr: readonly T[]): T {
  return arr[Math.floor(nextRandom() * arr.length)]!;
}

function squareToUci(square: number): string {
  const file = String.fromCharCode('a'.charCodeAt(0) + colOf(square));
  const rank = BOARD_ROWS - rowOf(square);
  return `${file}${rank}`;
}

/** Pass is UCI-encoded as the mover's own general square, doubled - confirmed against pyffish
 * directly (not assumed): e.g. a general at row8,col3 ("d2" in UCI) produces "d2d2" for pass,
 * matching Fairy-Stockfish's own same-square-twice convention applied specifically to wherever
 * the general currently sits, not a fixed placeholder square. */
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

interface DifferentialCase {
  fen: string;
  variant: string;
  ruleProfile: RuleProfileId;
  gameId: number;
  ply: number;
  tsLegalMovesUci: string[];
}

const cases: DifferentialCase[] = [];
let gameId = 0;

while (cases.length < TARGET_COUNT) {
  const ruleProfile = pick(RULE_PROFILES);
  const setupCho = pick(FORMATIONS);
  const setupHan = pick(FORMATIONS);
  let position: Position = createInitialPosition({ setupCho, setupHan, ruleProfile });

  for (let ply = 0; ply < MAX_PLIES_PER_GAME && cases.length < TARGET_COUNT; ply++) {
    const legalMoves = generateLegalMoves(position);
    if (legalMoves.length === 0) break; // checkmate

    cases.push({
      fen: fullFen(position),
      variant: RULE_PROFILE_TO_VARIANT[ruleProfile],
      ruleProfile,
      gameId,
      ply,
      tsLegalMovesUci: legalMoves.map((m) => moveToUci(position, m)).sort(),
    });

    const nonPass = legalMoves.filter((m) => !m.isPass);
    const pool = nonPass.length > 0 ? nonPass : legalMoves;
    position = applyMove(position, pool[Math.floor(nextRandom() * pool.length)]!);
  }
  gameId++;
}

const outPath = join(dirname(fileURLToPath(import.meta.url)), 'cases.jsonl');
writeFileSync(outPath, cases.map((c) => JSON.stringify(c)).join('\n') + '\n');
console.log(`Wrote ${cases.length} differential-test cases from ${gameId} games to ${outPath}`);
