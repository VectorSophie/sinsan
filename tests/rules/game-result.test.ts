import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sq, piece, customPosition } from './helpers.ts';
import { generateLegalMoves, isBikjang, getMaterialScore, getGameResult } from '../../packages/rules/src/rules.ts';
import { positionKey } from '../../packages/rules/src/position.ts';

test('bikjang: both generals on the same open file with nothing between them', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
  });
  assert.equal(isBikjang(position), true);
});

test('no bikjang when a piece stands between the two generals', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(5, 4), piece('soldier', 'cho')],
      [sq(8, 4), piece('general', 'cho')],
    ],
  });
  assert.equal(isBikjang(position), false);
});

test('no bikjang when generals are not on the same file', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
  });
  assert.equal(isBikjang(position), false);
});

test('bikjang under the kja profile ends the game as a draw', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'kja',
  });
  assert.deepEqual(getGameResult(position), { winner: null, kind: 'bikjang' });
});

test('bikjang under the modern profile does not end the game', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'modern',
  });
  assert.equal(getGameResult(position), null);
});

// The following bikjang-move-legality tests document a real discrepancy found via differential
// testing against pyffish (Section 8.4, see docs/RULES.md and packages/rules/src/rules.ts's
// generateLegalMoves doc comment for the full story): bikjang must be *resolved* once it exists,
// but *creating* it from a clean position is not itself illegal.

test('an existing bikjang restricts non-pass moves to ones that resolve it (kja profile)', () => {
  // Both generals already share file e (col 4). Moving the Han general to (0,4) or (2,4) keeps
  // it on that same file (bikjang persists) and must be illegal; moving to (1,3)/(1,5) (off the
  // file) resolves it and must remain legal; pass must remain legal regardless.
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'kja',
    sideToMove: 'han',
  });
  assert.equal(isBikjang(position), true, 'precondition: bikjang already exists');

  const legalDestinations = generateLegalMoves(position)
    .filter((m) => m.from === sq(1, 4) && !m.isPass)
    .map((m) => m.to)
    .sort((a, b) => a - b);

  assert.ok(!legalDestinations.includes(sq(0, 4)), 'staying on the shared file (toward row0) must not resolve bikjang - illegal');
  assert.ok(!legalDestinations.includes(sq(2, 4)), 'staying on the shared file (toward row2) must not resolve bikjang - illegal');
  assert.ok(legalDestinations.includes(sq(1, 3)), 'moving off the shared file resolves bikjang - legal');
  assert.ok(legalDestinations.includes(sq(1, 5)), 'moving off the shared file resolves bikjang - legal');
  assert.ok(generateLegalMoves(position).some((m) => m.isPass), 'pass remains legal even though it does not resolve bikjang');
});

test('a move that newly creates bikjang from a clean position is legal (kja profile)', () => {
  // Han general starts off Cho's general's file entirely - no bikjang yet - so moving onto that
  // file is a normal move, not a bikjang violation, even though the resulting position has
  // bikjang. This is the asymmetry a first, over-broad fix attempt got wrong (see rules.ts).
  const position = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'kja',
    sideToMove: 'han',
  });
  assert.equal(isBikjang(position), false, 'precondition: no bikjang yet');

  const legal = generateLegalMoves(position);
  const intoBikjang = legal.find((m) => m.from === sq(1, 3) && m.to === sq(1, 4));
  assert.notEqual(intoBikjang, undefined, 'moving onto the shared file from a clean position must remain legal');
});

test('bikjang never restricts moves under the modern profile', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'modern',
    sideToMove: 'han',
  });
  const legalDestinations = generateLegalMoves(position)
    .filter((m) => m.from === sq(1, 4) && !m.isPass)
    .map((m) => m.to);
  assert.ok(legalDestinations.includes(sq(0, 4)), 'modern profile does not restrict bikjang-maintaining moves');
  assert.ok(legalDestinations.includes(sq(2, 4)));
});

test('material score is signed, positive favoring cho', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
      [sq(5, 0), piece('chariot', 'cho')], // +13
      [sq(4, 0), piece('soldier', 'han')], // -2
    ],
  });
  assert.equal(getMaterialScore(position), 11);
});

test('material-count adjudication triggers at the no-capture limit under a profile that enables it', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')], // off column 4 so this isn't also a bikjang position
      [sq(8, 4), piece('general', 'cho')],
      [sq(5, 0), piece('chariot', 'cho')],
    ],
    ruleProfile: 'kja', // materialCountingAdjudication: true, noCaptureMoveLimit: 200
    noCapturePly: 200,
  });
  const result = getGameResult(position);
  assert.deepEqual(result, { winner: 'cho', kind: 'material-adjudication' });
});

test('no-capture limit under a profile without material adjudication is a plain draw', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')], // off column 4 so this isn't also a bikjang position
      [sq(8, 4), piece('general', 'cho')],
      [sq(5, 0), piece('chariot', 'cho')],
    ],
    ruleProfile: 'traditional', // materialCountingAdjudication: false
    noCapturePly: 200,
  });
  assert.deepEqual(getGameResult(position), { winner: null, kind: 'draw' });
});

test('repetition beyond the profile limit ends the game as a draw', () => {
  const base = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')], // off column 4 so this isn't also a bikjang position
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'kja', // repetitionLimit: 3
  });
  const key = positionKey(base);
  // base already contributes one occurrence; pad history with 3 more occurrences (4 total > limit of 3)
  const position = { ...base, positionHistory: [key, key, key, key] };
  assert.deepEqual(getGameResult(position), { winner: null, kind: 'repetition' });
});

test('repetition within the profile limit does not end the game', () => {
  const base = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')], // off column 4 so this isn't also a bikjang position
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'kja',
  });
  const key = positionKey(base);
  const position = { ...base, positionHistory: [key, key] };
  assert.equal(getGameResult(position), null);
});

test('timeout is reported from the clock, independent of board state', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
  });
  assert.deepEqual(getGameResult(position, { choMsRemaining: 0, hanMsRemaining: 5000 }), {
    winner: 'han',
    kind: 'timeout',
  });
});
