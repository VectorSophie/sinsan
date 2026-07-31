import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { squareOf } from '../../packages/rules/src/index.ts';
import type { Piece, PieceType, Position, Side } from '../../packages/rules/src/index.ts';
import { NUM_INPUT_PLANES, positionToPlanes } from '../../packages/model-runtime/src/features.ts';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

interface ParityCase {
  fen: string;
  input_planes: number[];
}
const fixture: ParityCase[] = JSON.parse(
  readFileSync(join(repoRoot, 'packages', 'model-runtime', 'parity-fixture.json'), 'utf8'),
);

const LETTER_TO_TYPE: Record<string, PieceType> = {
  r: 'chariot',
  c: 'cannon',
  n: 'horse',
  b: 'elephant',
  a: 'guard',
  k: 'general',
  p: 'soldier',
};

/** Minimal standalone FEN-board parser for this test only - deliberately not importing
 * packages/rules' parsePosition(), which expects Sinsan's own extended serialization format, not
 * the plain 6-field FEN training/generate/self-play.ts emits for the teacher engine. */
function positionFromStandardFen(fen: string): Pick<Position, 'board' | 'sideToMove'> {
  const [boardPart, sidePart] = fen.split(' ');
  const board: (Piece | null)[] = new Array(90).fill(null);
  let row = 0;
  let col = 0;
  for (const ch of boardPart!) {
    if (ch === '/') {
      row++;
      col = 0;
    } else if (/[0-9]/.test(ch)) {
      col += Number(ch);
    } else {
      const type = LETTER_TO_TYPE[ch.toLowerCase()]!;
      const side: Side = ch === ch.toUpperCase() ? 'cho' : 'han';
      board[squareOf(row, col)] = { type, side };
      col++;
    }
  }
  return { board, sideToMove: sidePart === 'w' ? 'cho' : 'han' };
}

test('TypeScript positionToPlanes matches training/model/network.py fen_to_planes exactly', () => {
  assert.ok(fixture.length >= 5);
  for (const testCase of fixture) {
    const position = positionFromStandardFen(testCase.fen) as Position;
    const planes = positionToPlanes(position);
    assert.equal(planes.length, testCase.input_planes.length);
    assert.equal(planes.length, NUM_INPUT_PLANES * 90);
    for (let i = 0; i < planes.length; i++) {
      assert.equal(planes[i], testCase.input_planes[i], `plane value mismatch at index ${i} for ${testCase.fen}`);
    }
  }
});
