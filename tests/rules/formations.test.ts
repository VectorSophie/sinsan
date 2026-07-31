import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createInitialPosition } from '../../packages/rules/src/position.ts';
import { squareOf } from '../../packages/rules/src/board.ts';
import type { Formation } from '../../packages/rules/src/types.ts';

function layoutAt(formation: Formation, row: number) {
  const position = createInitialPosition({ setupCho: formation, setupHan: formation });
  const cols = [0, 1, 2, 3, 5, 6, 7, 8].map((c) => position.board[squareOf(row, c)]?.type ?? null);
  return cols;
}

test('default formation is masang-sangma', () => {
  const position = createInitialPosition();
  assert.equal(position.setupCho, 'masang-sangma');
  assert.equal(position.setupHan, 'masang-sangma');
});

test('masang-sangma: horse next to chariot, elephant next to guard, both sides (mirror symmetric)', () => {
  const cols = layoutAt('masang-sangma', 9); // Cho back rank
  assert.deepEqual(cols, ['chariot', 'horse', 'elephant', 'guard', 'guard', 'elephant', 'horse', 'chariot']);
});

test('sangma-masang: elephant next to chariot, horse next to guard, both sides', () => {
  const cols = layoutAt('sangma-masang', 9);
  assert.deepEqual(cols, ['chariot', 'elephant', 'horse', 'guard', 'guard', 'horse', 'elephant', 'chariot']);
});

test('masang-masang: horse-elephant reading left to right on both pairs (not mirror symmetric)', () => {
  const cols = layoutAt('masang-masang', 9);
  assert.deepEqual(cols, ['chariot', 'horse', 'elephant', 'guard', 'guard', 'horse', 'elephant', 'chariot']);
});

test('sangma-sangma: elephant-horse reading left to right on both pairs', () => {
  const cols = layoutAt('sangma-sangma', 9);
  assert.deepEqual(cols, ['chariot', 'elephant', 'horse', 'guard', 'guard', 'elephant', 'horse', 'chariot']);
});

test('each side can choose its formation independently', () => {
  const position = createInitialPosition({ setupCho: 'masang-sangma', setupHan: 'sangma-sangma' });
  assert.equal(position.board[squareOf(9, 1)]?.type, 'horse'); // Cho: masang-sangma
  assert.equal(position.board[squareOf(0, 1)]?.type, 'elephant'); // Han: sangma-sangma
});

test('generals start at the palace center, not the back rank', () => {
  const position = createInitialPosition();
  assert.equal(position.board[squareOf(1, 4)]?.type, 'general');
  assert.equal(position.board[squareOf(1, 4)]?.side, 'han');
  assert.equal(position.board[squareOf(8, 4)]?.type, 'general');
  assert.equal(position.board[squareOf(8, 4)]?.side, 'cho');
});

test('cho moves first', () => {
  const position = createInitialPosition();
  assert.equal(position.sideToMove, 'cho');
});
