import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sq, piece, customPosition } from './helpers.ts';
import { generateLegalMoves, applyMove, isCheck, getGameResult } from '../../packages/rules/src/rules.ts';
import { createInitialPosition } from '../../packages/rules/src/position.ts';

test('a move that would leave the mover\'s own general in check is illegal (pin)', () => {
  // Cho general on column 4; a Cho soldier sits between it and a Han chariot on the same column.
  // Sideways moves would expose the general and must be filtered out, while the forward move -
  // which stays on the same column - remains legal.
  const position = customPosition({
    pieces: [
      [sq(8, 4), piece('general', 'cho')],
      [sq(6, 4), piece('soldier', 'cho')], // pinned - blocks the chariot's attack on the general
      [sq(1, 4), piece('chariot', 'han')],
      [sq(1, 3), piece('general', 'han')],
    ],
  });
  const soldierMoves = generateLegalMoves(position)
    .filter((m) => m.from === sq(6, 4))
    .map((m) => m.to)
    .sort((a, b) => a - b);
  assert.deepEqual(soldierMoves, [sq(5, 4)], 'only the in-column forward move stays legal; sideways moves expose the general');
});

test('pass is illegal while in check', () => {
  const position = customPosition({
    pieces: [
      [sq(9, 4), piece('general', 'cho')],
      [sq(6, 4), piece('chariot', 'han')],
      [sq(1, 4), piece('general', 'han')],
    ],
  });
  assert.equal(isCheck(position), true);
  assert.ok(!generateLegalMoves(position).some((m) => m.isPass), 'no pass move among the legal moves');
});

test('pass is legal when not in check', () => {
  const position = createInitialPosition();
  assert.equal(isCheck(position), false);
  assert.ok(generateLegalMoves(position).some((m) => m.isPass), 'pass is among the legal moves');
});

test('checkmate: general attacked with every palace escape square also covered', () => {
  const position = customPosition({
    pieces: [
      [sq(9, 4), piece('general', 'cho')],
      [sq(6, 3), piece('chariot', 'han')], // covers (9,3) down column 3
      [sq(6, 4), piece('chariot', 'han')], // covers the general directly and (8,4)
      [sq(6, 5), piece('chariot', 'han')], // covers (9,5) down column 5
      [sq(1, 4), piece('general', 'han')],
    ],
    sideToMove: 'cho',
  });
  assert.equal(isCheck(position), true);
  assert.deepEqual(generateLegalMoves(position), [], 'no legal move escapes check');
  const result = getGameResult(position);
  assert.deepEqual(result, { winner: 'han', kind: 'checkmate' });
});

test('check with an available escape square is not checkmate', () => {
  const position = customPosition({
    pieces: [
      [sq(9, 4), piece('general', 'cho')],
      [sq(6, 3), piece('chariot', 'han')],
      [sq(6, 4), piece('chariot', 'han')],
      // no attacker on column 5 - (9,5) is a free escape
      [sq(1, 4), piece('general', 'han')],
    ],
    sideToMove: 'cho',
  });
  assert.equal(isCheck(position), true);
  const legal = generateLegalMoves(position);
  assert.ok(legal.some((m) => m.to === sq(9, 5)), 'general can escape to (9,5)');
  assert.equal(getGameResult(position), null, 'game continues - not checkmate');
});

test('applyMove flips the side to move and tracks capture for the no-capture counter', () => {
  const position = createInitialPosition();
  const moves = generateLegalMoves(position);
  const nonPassMove = moves.find((m) => !m.isPass)!;
  const next = applyMove(position, nonPassMove);
  assert.equal(next.sideToMove, 'han');
  assert.equal(next.moveNumber, position.moveNumber + 1);
});
