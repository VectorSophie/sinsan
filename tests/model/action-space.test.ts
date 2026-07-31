import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  applyMove,
  colOf,
  createInitialPosition,
  generateLegalMoves,
  rowOf,
  squareOf,
} from '../../packages/rules/src/index.ts';
import type { Piece, Position } from '../../packages/rules/src/index.ts';
import {
  ACTION_SPACE_SIZE,
  PASS_ACTION,
  TEMPLATE_COUNT,
  decodeAction,
  encodeMove,
  legalActionMask,
  reflectActionHorizontally,
} from '../../packages/action-space/src/index.ts';

test('action space size is exactly 90 origins x 60 templates + 1 pass', () => {
  assert.equal(TEMPLATE_COUNT, 60);
  assert.equal(ACTION_SPACE_SIZE, 90 * 60 + 1);
  assert.equal(PASS_ACTION, 5400);
});

test('pass has a dedicated action id regardless of its (unused) from/to fields', () => {
  assert.equal(encodeMove({ from: -1, to: -1, isPass: true }), PASS_ACTION);
  assert.equal(encodeMove({ from: 12, to: 34, isPass: true }), PASS_ACTION);
  assert.equal(decodeAction(PASS_ACTION), 'pass');
});

/** Plays a handful of pseudo-random games (deterministic seed) to gather a broad, realistic set
 * of legal-move lists covering every piece type's movement patterns, not just the opening. */
function collectLegalMoveLists(gameCount: number, plyCount: number): Array<ReturnType<typeof generateLegalMoves>> {
  let seed = 42;
  function nextRandom(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  const allLists: Array<ReturnType<typeof generateLegalMoves>> = [];
  for (let g = 0; g < gameCount; g++) {
    let position: Position = createInitialPosition();
    for (let ply = 0; ply < plyCount; ply++) {
      const legal = generateLegalMoves(position);
      allLists.push(legal);
      if (legal.length === 0) break;
      const nonPass = legal.filter((m) => !m.isPass);
      const pool = nonPass.length > 0 ? nonPass : legal;
      const move = pool[Math.floor(nextRandom() * pool.length)]!;
      position = applyMove(position, move);
    }
  }
  return allLists;
}

test('every legal move encodes to exactly one action, and decodes back to the same move', () => {
  const moveLists = collectLegalMoveLists(6, 40);
  let checked = 0;
  for (const moves of moveLists) {
    for (const move of moves) {
      const actionId = encodeMove(move);
      assert.notEqual(actionId, undefined, `move ${JSON.stringify(move)} did not encode to any action`);
      if (move.isPass) {
        assert.equal(actionId, PASS_ACTION);
        continue;
      }
      const decoded = decodeAction(actionId!);
      assert.notEqual(decoded, undefined, `action ${actionId} decoded to undefined (off-board?)`);
      assert.notEqual(decoded, 'pass');
      assert.deepEqual(decoded, { from: move.from, to: move.to });
      checked++;
    }
  }
  assert.ok(checked > 500, `expected broad coverage from random games, only checked ${checked} moves`);
});

test('no two distinct legal moves in the same position collide on the same action id', () => {
  const moveLists = collectLegalMoveLists(6, 40);
  for (const moves of moveLists) {
    const seen = new Map<number, string>();
    for (const move of moves) {
      const actionId = encodeMove(move)!;
      const key = `${move.from}->${move.to}${move.isPass ? '(pass)' : ''}`;
      const existing = seen.get(actionId);
      if (existing !== undefined && existing !== key) {
        assert.fail(`collision: "${existing}" and "${key}" both encode to action ${actionId}`);
      }
      seen.set(actionId, key);
    }
  }
});

test('legalActionMask marks exactly the legal actions and nothing else', () => {
  const position = createInitialPosition();
  const legal = generateLegalMoves(position);
  const mask = legalActionMask(legal);
  assert.equal(mask.length, ACTION_SPACE_SIZE);
  const expectedIds = new Set(legal.map((m) => encodeMove(m)!));
  for (let id = 0; id < ACTION_SPACE_SIZE; id++) {
    assert.equal(mask[id], expectedIds.has(id) ? 1 : 0, `mask mismatch at action ${id}`);
  }
});

function reflectPosition(position: Position): Position {
  const board: (Piece | null)[] = position.board.slice() as (Piece | null)[];
  const reflected: (Piece | null)[] = new Array(board.length).fill(null);
  for (let sq = 0; sq < board.length; sq++) {
    const piece = board[sq];
    if (!piece) continue;
    const mirroredSquare = squareOf(rowOf(sq), 8 - colOf(sq));
    reflected[mirroredSquare] = piece;
  }
  return { ...position, board: reflected };
}

test('horizontal reflection transforms actions consistently with a mirrored position', () => {
  let checked = 0;
  let seed = 7;
  function nextRandom(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }
  let position: Position = createInitialPosition();
  for (let ply = 0; ply < 30; ply++) {
    const legal = generateLegalMoves(position);
    if (legal.length === 0) break;
    const mirroredPosition = reflectPosition(position);
    const mirroredLegalDestinationsByOrigin = new Set(
      generateLegalMoves(mirroredPosition)
        .filter((m) => !m.isPass)
        .map((m) => encodeMove(m)!),
    );
    for (const move of legal) {
      if (move.isPass) continue;
      const actionId = encodeMove(move)!;
      const reflectedActionId = reflectActionHorizontally(actionId)!;
      assert.ok(
        mirroredLegalDestinationsByOrigin.has(reflectedActionId),
        `reflected action ${reflectedActionId} (from move ${JSON.stringify(move)}) is not legal in the mirrored position`,
      );
      checked++;
    }
    const nonPass = legal.filter((m) => !m.isPass);
    const pool = nonPass.length > 0 ? nonPass : legal;
    position = applyMove(position, pool[Math.floor(nextRandom() * pool.length)]!);
  }
  assert.ok(checked > 200, `expected broad reflection coverage, only checked ${checked} moves`);
});

test('reflecting twice returns the original action', () => {
  const position = createInitialPosition();
  for (const move of generateLegalMoves(position)) {
    if (move.isPass) continue;
    const actionId = encodeMove(move)!;
    const twice = reflectActionHorizontally(reflectActionHorizontally(actionId)!)!;
    assert.equal(twice, actionId);
  }
});
