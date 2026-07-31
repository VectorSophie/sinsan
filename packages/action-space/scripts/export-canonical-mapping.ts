/**
 * Exports the canonical action-space definition and a golden encode/decode fixture so Python
 * (training/action_space.py) can load the exact same mapping TypeScript uses, rather than
 * maintaining a second hand-written implementation that could drift - see Section 9's
 * requirement that "action encoding is stable across Python and TypeScript."
 *
 * Run with: node packages/action-space/scripts/export-canonical-mapping.ts
 */
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  BOARD_COLS,
  BOARD_ROWS,
  applyMove,
  createInitialPosition,
  generateLegalMoves,
} from '../../rules/src/index.ts';
import type { Position } from '../../rules/src/index.ts';
import { ACTION_SPACE_SIZE, PASS_ACTION, TEMPLATES, TEMPLATE_COUNT, encodeMove } from '../src/index.ts';

const outDir = join(dirname(fileURLToPath(import.meta.url)), '..');

writeFileSync(
  join(outDir, 'canonical-mapping.json'),
  JSON.stringify(
    {
      boardCols: BOARD_COLS,
      boardRows: BOARD_ROWS,
      templateCount: TEMPLATE_COUNT,
      templates: TEMPLATES,
      actionSpaceSize: ACTION_SPACE_SIZE,
      passAction: PASS_ACTION,
    },
    null,
    2,
  ) + '\n',
);

// Golden fixture: (position, move, actionId) triples from real games, for Python to cross-check
// its own encode() against - the actual consistency guarantee, not just "trust the arithmetic."
let seed = 1234;
function nextRandom(): number {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
}

const fixture: Array<{ fen: string; from: number; to: number; isPass: boolean; actionId: number }> = [];
for (let game = 0; game < 4; game++) {
  let position: Position = createInitialPosition();
  for (let ply = 0; ply < 30; ply++) {
    const legal = generateLegalMoves(position);
    if (legal.length === 0) break;
    for (const move of legal) {
      const actionId = encodeMove(move);
      if (actionId === undefined) throw new Error(`encodeMove failed for ${JSON.stringify(move)}`);
      if (nextRandom() < 0.05) {
        // sample ~5% of moves at each position to keep the fixture a manageable size
        fixture.push({
          fen: boardFen(position),
          from: move.from,
          to: move.to,
          isPass: move.isPass,
          actionId,
        });
      }
    }
    const nonPass = legal.filter((m) => !m.isPass);
    const pool = nonPass.length > 0 ? nonPass : legal;
    position = applyMove(position, pool[Math.floor(nextRandom() * pool.length)]!);
  }
}

function boardFen(position: Position): string {
  // local copy of the board-part-of-serialization logic to avoid a circular dev-dependency;
  // matches packages/rules/src/position.ts's positionKey() board part exactly.
  return positionToBoardFen(position);
}

function positionToBoardFen(position: Position): string {
  const rows: string[] = [];
  for (let r = 0; r < BOARD_ROWS; r++) {
    let rowStr = '';
    let emptyRun = 0;
    for (let c = 0; c < BOARD_COLS; c++) {
      const piece = position.board[r * BOARD_COLS + c];
      if (piece) {
        if (emptyRun > 0) {
          rowStr += String(emptyRun);
          emptyRun = 0;
        }
        const letters: Record<string, string> = {
          chariot: 'r',
          horse: 'n',
          elephant: 'b',
          guard: 'a',
          general: 'k',
          cannon: 'c',
          soldier: 'p',
        };
        const letter = letters[piece.type]!;
        rowStr += piece.side === 'cho' ? letter.toUpperCase() : letter;
      } else {
        emptyRun++;
      }
    }
    if (emptyRun > 0) rowStr += String(emptyRun);
    rows.push(rowStr);
  }
  return rows.join('/') + (position.sideToMove === 'cho' ? ' w' : ' b');
}

writeFileSync(join(outDir, 'golden-fixture.json'), JSON.stringify(fixture, null, 2) + '\n');

console.log(`Wrote canonical-mapping.json (${TEMPLATE_COUNT} templates) and golden-fixture.json (${fixture.length} entries)`);
