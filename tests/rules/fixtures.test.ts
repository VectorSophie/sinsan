import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parsePosition } from '../../packages/rules/src/position.ts';
import { generateLegalMoves, getGameResult } from '../../packages/rules/src/rules.ts';
import { pseudoLegalMovesFrom } from '../../packages/rules/src/moves.ts';
import { squareOf } from '../../packages/rules/src/board.ts';

const fixturesDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'fixtures', 'rules');

function loadFixture(name: string): any {
  return JSON.parse(readFileSync(join(fixturesDir, name), 'utf8'));
}

test('fixture: checkmate-column-block', () => {
  const fixture = loadFixture('checkmate-column-block.json');
  const position = parsePosition(fixture.position);
  assert.equal(generateLegalMoves(position).length, fixture.expectedLegalMoveCount);
  assert.deepEqual(getGameResult(position), fixture.expectedGameResult);
});

test('fixture: bikjang-bare-generals (kja and modern profiles diverge)', () => {
  const fixture = loadFixture('bikjang-bare-generals.json');
  const kjaPosition = parsePosition(fixture.position);
  assert.deepEqual(getGameResult(kjaPosition), fixture.expectedGameResult);

  const modernPosition = parsePosition(fixture.modernProfileVariant.position);
  assert.equal(getGameResult(modernPosition), fixture.modernProfileVariant.expectedGameResult);
});

test('fixture: cannon-cannot-screen-cannon', () => {
  const fixture = loadFixture('cannon-cannot-screen-cannon.json');
  const position = parsePosition(fixture.position);
  const cannonSquare = squareOf(fixture.cannonSquare.row, fixture.cannonSquare.col);
  const rightwardDestinations = pseudoLegalMovesFrom(position, cannonSquare)
    .map((m) => m.to)
    .filter((to) => to > cannonSquare && to % 9 > fixture.cannonSquare.col);
  assert.deepEqual(rightwardDestinations, fixture.expectedDestinationsOnRightwardRay);
});
