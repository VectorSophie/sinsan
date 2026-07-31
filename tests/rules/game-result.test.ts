import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sq, piece, customPosition } from './helpers.ts';
import { isBikjang, getMaterialScore, getGameResult } from '../../packages/rules/src/rules.ts';
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
