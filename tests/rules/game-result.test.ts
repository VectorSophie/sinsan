import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sq, piece, customPosition } from './helpers.ts';
import { applyMove, generateLegalMoves, isBikjang, getMaterialScore, getGameResult } from '../../packages/rules/src/rules.ts';
import type { Move, Position } from '../../packages/rules/src/types.ts';

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

const pass = (position: Position): Move => {
  const general = position.board.findIndex((p) => p?.type === 'general' && p.side === position.sideToMove);
  return { from: general, to: general, isPass: true };
};

function play(position: Position, moves: ReadonlyArray<Move>): Position {
  return moves.reduce((p, m) => applyMove(p, m), position);
}

test('bikjang alone does not end the game; the opponent must answer it (kja profile)', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 4), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'kja',
  });
  assert.equal(getGameResult(position), null);
});

test('bikjang left standing for a second position is decided by points (kja profile)', () => {
  const start = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
      [sq(5, 0), piece('chariot', 'cho')], // Cho leads 13-0
    ],
    sideToMove: 'han',
    ruleProfile: 'kja',
  });
  const created = applyMove(start, { from: sq(1, 3), to: sq(1, 4), isPass: false }); // Han creates bikjang
  assert.equal(getGameResult(created), null);
  const ignored = applyMove(created, pass(created)); // Cho passes instead of breaking it
  assert.deepEqual(getGameResult(ignored), { winner: 'cho', kind: 'bikjang' });
});

test('two consecutive passes end the game on points, and Han wins ties thanks to compensation', () => {
  const start = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
      [sq(9, 3), piece('guard', 'cho')], // 3 points
      [sq(5, 0), piece('soldier', 'han')], // 2 points: Cho leads by 1, less than Han's 1.5
    ],
  });
  const afterOne = applyMove(start, pass(start));
  assert.equal(getGameResult(afterOne), null);
  const afterTwo = applyMove(afterOne, pass(afterOne));
  assert.deepEqual(getGameResult(afterTwo), { winner: 'han', kind: 'double-pass' });
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
    ruleProfile: 'kja', // materialCountingAdjudication: true, noCaptureMoveLimit: 100
    noCapturePly: 100,
  });
  const result = getGameResult(position);
  assert.deepEqual(result, { winner: 'cho', kind: 'move-limit' });
});

test('no-capture limit under a profile without material adjudication is a plain draw', () => {
  const position = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')], // off column 4 so this isn't also a bikjang position
      [sq(8, 4), piece('general', 'cho')],
      [sq(5, 0), piece('chariot', 'cho')],
    ],
    ruleProfile: 'traditional', // materialCountingAdjudication: false
    noCapturePly: 100,
  });
  assert.deepEqual(getGameResult(position), { winner: null, kind: 'move-limit' });
});

/** Both generals step back and forth (on different files, so no bikjang): every 4 plies the
 * start position recurs. */
function shuffle(start: Position, cycles: number): Position {
  const cho: Move[] = [{ from: sq(8, 4), to: sq(9, 4), isPass: false }, { from: sq(9, 4), to: sq(8, 4), isPass: false }];
  const han: Move[] = [{ from: sq(1, 3), to: sq(0, 3), isPass: false }, { from: sq(0, 3), to: sq(1, 3), isPass: false }];
  let p = start;
  for (let i = 0; i < cycles; i++) p = play(p, [cho[0]!, han[0]!, cho[1]!, han[1]!]);
  return p;
}

test('threefold repetition is decided by points (kja profile)', () => {
  const start = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
      [sq(5, 0), piece('chariot', 'cho')],
    ],
  });
  assert.equal(getGameResult(shuffle(start, 1)), null); // second occurrence
  assert.deepEqual(getGameResult(shuffle(start, 2)), { winner: 'cho', kind: 'repetition' }); // third
});

test('repetition within the profile limit does not end the game', () => {
  const start = customPosition({
    pieces: [
      [sq(1, 3), piece('general', 'han')],
      [sq(8, 4), piece('general', 'cho')],
    ],
    ruleProfile: 'modern', // fourfold
  });
  assert.equal(getGameResult(shuffle(start, 2)), null);
});

test('repeating while giving check on every move loses (perpetual check)', () => {
  // Han's general steps d10 <-> e10 and Cho's chariot follows it along rank 6, checking on every
  // move. The cycle repeats three times; Cho, the perpetual checker, loses.
  const start = customPosition({
    pieces: [
      [sq(0, 3), piece('general', 'han')],
      [sq(9, 5), piece('general', 'cho')],
      [sq(4, 3), piece('chariot', 'cho')],
    ],
    sideToMove: 'han',
  });
  const han: Move[] = [{ from: sq(0, 3), to: sq(0, 4), isPass: false }, { from: sq(0, 4), to: sq(0, 3), isPass: false }];
  const cho: Move[] = [{ from: sq(4, 3), to: sq(4, 4), isPass: false }, { from: sq(4, 4), to: sq(4, 3), isPass: false }];
  let p = start;
  for (let i = 0; i < 2; i++) {
    for (const m of [han[0]!, cho[0]!, han[1]!, cho[1]!]) {
      assert.ok(generateLegalMoves(p).some((l) => l.from === m.from && l.to === m.to), 'scripted move must be legal');
      p = applyMove(p, m);
    }
  }
  assert.deepEqual(getGameResult(p), { winner: 'han', kind: 'perpetual-check' });
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
