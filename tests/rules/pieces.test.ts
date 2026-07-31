import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sq, piece, customPosition } from './helpers.ts';
import { pseudoLegalMovesFrom } from '../../packages/rules/src/moves.ts';

function destinations(position: ReturnType<typeof customPosition>, from: number): number[] {
  return pseudoLegalMovesFrom(position, from)
    .map((m) => m.to)
    .sort((a, b) => a - b);
}

test('chariot slides orthogonally and stops at the first piece, capturing enemies only', () => {
  const position = customPosition({
    pieces: [
      [sq(5, 4), piece('chariot', 'cho')],
      [sq(5, 6), piece('soldier', 'han')], // enemy blocker to the right
      [sq(5, 2), piece('soldier', 'cho')], // friendly blocker to the left
    ],
  });
  const dests = destinations(position, sq(5, 4));
  assert.ok(dests.includes(sq(5, 6)), 'can capture the enemy blocker');
  assert.ok(!dests.includes(sq(5, 7)), 'cannot slide past a capture');
  assert.ok(dests.includes(sq(5, 5))); // empty square before the enemy blocker
  assert.ok(dests.includes(sq(5, 3)), 'empty square before the friendly blocker is still reachable');
  assert.ok(!dests.includes(sq(5, 2)), 'cannot land on a friendly piece');
  assert.ok(!dests.includes(sq(5, 1)), 'cannot slide through a friendly piece');
  assert.ok(dests.includes(sq(0, 4)) || dests.includes(sq(9, 4)), 'slides vertically too');
});

test('chariot may use the palace diagonal, corner to opposite corner through an empty center', () => {
  const position = customPosition({
    pieces: [[sq(0, 3), piece('chariot', 'han')]], // Han palace top-left corner
  });
  const dests = destinations(position, sq(0, 3));
  assert.ok(dests.includes(sq(1, 4)), 'one diagonal step to palace center');
  assert.ok(dests.includes(sq(2, 5)), 'through the empty center to the opposite corner');
});

test('chariot cannot use a palace diagonal to jump a blocked center', () => {
  const position = customPosition({
    pieces: [
      [sq(0, 3), piece('chariot', 'han')],
      [sq(1, 4), piece('guard', 'han')], // own piece on the center blocks the diagonal
    ],
  });
  const dests = destinations(position, sq(0, 3));
  assert.ok(!dests.includes(sq(1, 4)), 'blocked by own piece');
  assert.ok(!dests.includes(sq(2, 5)), 'cannot jump past the blocker');
});

test('cannon must jump exactly one screen and cannot land without one', () => {
  const position = customPosition({
    pieces: [[sq(5, 4), piece('cannon', 'cho')]],
  });
  assert.deepEqual(destinations(position, sq(5, 4)), [], 'no screen anywhere on any ray - no moves');
});

test('cannon jumps a single non-cannon screen to capture the next piece beyond it', () => {
  const position = customPosition({
    pieces: [
      [sq(5, 4), piece('cannon', 'cho')],
      [sq(5, 6), piece('soldier', 'cho')], // screen
      [sq(5, 8), piece('soldier', 'han')], // capturable beyond the screen
    ],
  });
  const dests = destinations(position, sq(5, 4));
  assert.ok(dests.includes(sq(5, 8)), 'captures the enemy piece beyond the screen');
  assert.ok(!dests.includes(sq(5, 6)), 'cannot land on the screen square itself');
  assert.ok(!dests.includes(sq(5, 5)), 'cannot land before the screen');
});

test('cannons cannot capture cannons', () => {
  const position = customPosition({
    pieces: [
      [sq(5, 4), piece('cannon', 'cho')],
      [sq(5, 6), piece('soldier', 'cho')], // screen
      [sq(5, 8), piece('cannon', 'han')], // enemy cannon beyond the screen
    ],
  });
  const dests = destinations(position, sq(5, 4));
  assert.ok(!dests.includes(sq(5, 8)), 'cannot capture the enemy cannon');
});

test('cannons cannot use another cannon as their screen', () => {
  const position = customPosition({
    pieces: [
      [sq(5, 4), piece('cannon', 'cho')],
      [sq(5, 6), piece('cannon', 'han')], // illegal screen
      [sq(5, 8), piece('soldier', 'han')],
    ],
  });
  const dests = destinations(position, sq(5, 4));
  assert.deepEqual(dests, [], 'cannon screened by a cannon has no moves on this ray');
});

test('horse is blocked by a piece on its orthogonal leg', () => {
  const free = customPosition({ pieces: [[sq(5, 4), piece('horse', 'cho')]] });
  const freeDests = destinations(free, sq(5, 4));
  assert.ok(freeDests.includes(sq(3, 3)) && freeDests.includes(sq(3, 5)), 'unblocked horse has both upward jumps');

  const blocked = customPosition({
    pieces: [
      [sq(5, 4), piece('horse', 'cho')],
      [sq(4, 4), piece('soldier', 'han')], // blocks the upward leg
    ],
  });
  const blockedDests = destinations(blocked, sq(5, 4));
  assert.ok(!blockedDests.includes(sq(3, 3)) && !blockedDests.includes(sq(3, 5)), 'both upward jumps blocked by the leg');
});

test('elephant is blocked by a piece on either its leg or its first diagonal step', () => {
  const free = customPosition({ pieces: [[sq(5, 4), piece('elephant', 'cho')]] });
  assert.ok(destinations(free, sq(5, 4)).includes(sq(2, 2)), 'unblocked elephant reaches the far diagonal square');

  const legBlocked = customPosition({
    pieces: [
      [sq(5, 4), piece('elephant', 'cho')],
      [sq(4, 4), piece('soldier', 'han')], // leg square
    ],
  });
  assert.ok(!destinations(legBlocked, sq(5, 4)).includes(sq(2, 2)), 'blocked at the leg');

  const midBlocked = customPosition({
    pieces: [
      [sq(5, 4), piece('elephant', 'cho')],
      [sq(3, 3), piece('soldier', 'han')], // first diagonal step
    ],
  });
  assert.ok(!destinations(midBlocked, sq(5, 4)).includes(sq(2, 2)), 'blocked at the first diagonal step');
});

test('guard and general move one step within the palace, including its diagonals, and never leave it', () => {
  const position = customPosition({ pieces: [[sq(8, 4), piece('general', 'cho')]] }); // Cho palace center
  const dests = destinations(position, sq(8, 4));
  assert.deepEqual(
    dests.sort((a, b) => a - b),
    [sq(7, 3), sq(7, 4), sq(7, 5), sq(8, 3), sq(8, 5), sq(9, 3), sq(9, 4), sq(9, 5)].sort((a, b) => a - b),
  );
});

test('guard cannot step outside the palace even along the board edge', () => {
  const position = customPosition({ pieces: [[sq(9, 3), piece('guard', 'cho')]] }); // Cho palace corner
  const dests = destinations(position, sq(9, 3));
  assert.ok(!dests.includes(sq(9, 2)), 'orthogonal step outside the palace column range is illegal');
});

test('soldier moves forward and sideways but never backward', () => {
  const choSoldier = customPosition({ pieces: [[sq(5, 4), piece('soldier', 'cho')]] });
  const choDests = destinations(choSoldier, sq(5, 4));
  assert.ok(choDests.includes(sq(4, 4)), 'cho soldier moves toward row 0');
  assert.ok(choDests.includes(sq(5, 3)) && choDests.includes(sq(5, 5)), 'sideways moves');
  assert.ok(!choDests.includes(sq(6, 4)), 'no backward move');

  const hanSoldier = customPosition({ pieces: [[sq(5, 4), piece('soldier', 'han')]] });
  const hanDests = destinations(hanSoldier, sq(5, 4));
  assert.ok(hanDests.includes(sq(6, 4)), 'han soldier moves toward row 9');
  assert.ok(!hanDests.includes(sq(4, 4)), 'no backward move');
});

test('soldier gets a forward-diagonal move only on a palace diagonal line', () => {
  const position = customPosition({ pieces: [[sq(7, 3), piece('soldier', 'cho')]] }); // Cho palace corner, forward = toward row 0
  const dests = destinations(position, sq(7, 3));
  assert.ok(dests.includes(sq(8, 4)) === false, 'row 8 is backward for cho, should not be reachable diagonally');
});

test('soldier forward diagonal in the enemy palace (the realistic case of advance)', () => {
  const position = customPosition({ pieces: [[sq(2, 3), piece('soldier', 'cho')]] }); // deep in Han's palace corner
  const dests = destinations(position, sq(2, 3));
  assert.ok(dests.includes(sq(1, 4)), 'cho forward is decreasing row - diagonal into the palace center is legal');
});
