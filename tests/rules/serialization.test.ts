import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createInitialPosition, serializePosition, parsePosition, positionKey } from '../../packages/rules/src/position.ts';
import { generateLegalMoves, applyMove } from '../../packages/rules/src/rules.ts';

test('serializePosition/parsePosition round-trips the initial position', () => {
  const position = createInitialPosition({ setupCho: 'sangma-masang', setupHan: 'masang-masang' });
  const serialized = serializePosition(position);
  const parsed = parsePosition(serialized);
  assert.deepEqual(parsed.board, position.board);
  assert.equal(parsed.sideToMove, position.sideToMove);
  assert.equal(parsed.ruleProfile, position.ruleProfile);
  assert.equal(parsed.setupCho, position.setupCho);
  assert.equal(parsed.setupHan, position.setupHan);
});

test('positionKey is stable for identical board+side and changes after a move', () => {
  const position = createInitialPosition();
  const keyBefore = positionKey(position);
  assert.equal(positionKey(position), keyBefore, 'deterministic for the same position');

  const move = generateLegalMoves(position).find((m) => !m.isPass)!;
  const next = applyMove(position, move);
  assert.notEqual(positionKey(next), keyBefore, 'key changes once a non-pass move is made');
});

test('round-trip preserves enough state to resume play identically', () => {
  const position = createInitialPosition();
  const move = generateLegalMoves(position).find((m) => !m.isPass)!;
  const afterMove = applyMove(position, move);
  const resumed = parsePosition(serializePosition(afterMove));
  assert.deepEqual(generateLegalMoves(resumed).length, generateLegalMoves(afterMove).length);
});
