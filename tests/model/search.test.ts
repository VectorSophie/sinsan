import assert from 'node:assert/strict';
import { test } from 'node:test';
import { encodeMove, ACTION_SPACE_SIZE } from '../../packages/action-space/src/index.ts';
import { applyMove, createInitialPosition, generateLegalMoves } from '../../packages/rules/src/index.ts';
import type { Position } from '../../packages/rules/src/index.ts';
import { search, SearchTree } from '../../packages/search/src/index.ts';
import type { PositionEvaluation, PositionEvaluator } from '../../packages/search/src/index.ts';
import { customPosition, piece, sq } from '../rules/helpers.ts';

/** A deterministic, near-instant evaluator for testing search mechanics in isolation from any
 * real model - uniform-ish policy (tiny random jitter so PUCT ties break consistently) and a
 * fixed value. Real move *quality* is not what these tests check; that's what the trained model
 * and (eventually) arena evaluation are for. */
function uniformEvaluator(): PositionEvaluator {
  return (): PositionEvaluation => {
    const policyLogits = new Float32Array(ACTION_SPACE_SIZE);
    for (let i = 0; i < policyLogits.length; i++) policyLogits[i] = Math.sin(i) * 0.01;
    return { policyLogits, value: 0 };
  };
}

/** An evaluator whose policy overwhelmingly favors one specific action, so PUCT concentrates
 * nearly all visits there - makes tree-reuse testable deterministically instead of depending on
 * exactly how visits happen to distribute under a uniform prior. */
function spikedEvaluator(spikeActionId: number): PositionEvaluator {
  return (): PositionEvaluation => {
    const policyLogits = new Float32Array(ACTION_SPACE_SIZE).fill(-5);
    policyLogits[spikeActionId] = 10;
    return { policyLogits, value: 0 };
  };
}

test('search returns a legal move from the initial position', async () => {
  const position = createInitialPosition();
  const legalActionIds = new Set(generateLegalMoves(position).map((m) => encodeMove(m)!));
  const result = await search(position, uniformEvaluator(), { visits: 20 });
  const actionId = encodeMove(result.move);
  assert.notEqual(actionId, undefined);
  assert.ok(legalActionIds.has(actionId!), 'search must only ever return an actually-legal move');
  assert.equal(result.visitsUsed, 20);
});

test('search visit counts sum to at most visitsUsed and cover only legal actions', async () => {
  const position = createInitialPosition();
  const legalActionIds = new Set(generateLegalMoves(position).map((m) => encodeMove(m)!));
  const result = await search(position, uniformEvaluator(), { visits: 30 });
  let total = 0;
  for (const [actionId, count] of result.visitCounts) {
    assert.ok(legalActionIds.has(actionId), `visited action ${actionId} is not legal`);
    total += count;
  }
  assert.ok(total <= result.visitsUsed);
});

test('search throws a clear error when the root position is already game-over', async () => {
  // Same minimal forced-mate position as tests/rules/check.test.ts's checkmate fixture: Cho
  // general boxed in with all three palace escape squares covered by Han chariots.
  const checkmatePosition = customPosition({
    pieces: [
      [sq(9, 4), piece('general', 'cho')],
      [sq(6, 3), piece('chariot', 'han')],
      [sq(6, 4), piece('chariot', 'han')],
      [sq(6, 5), piece('chariot', 'han')],
      [sq(1, 4), piece('general', 'han')],
    ],
    sideToMove: 'cho',
  });
  await assert.rejects(() => search(checkmatePosition, uniformEvaluator(), { visits: 10 }), /already game-over/);
});

test('SearchTree reuses the subtree for the position it just returned a move for', async () => {
  const initial = createInitialPosition();
  const legalMoves = generateLegalMoves(initial).filter((m) => !m.isPass);
  const spikeMove = legalMoves[0]!;
  const spikeActionId = encodeMove(spikeMove)!;

  const tree = new SearchTree(spikedEvaluator(spikeActionId));
  const first = await tree.getMove(initial, 40);

  assert.equal(encodeMove(first.move), spikeActionId, 'the overwhelmingly-favored move should be selected');
  const carriedOverVisits = first.visitCounts.get(spikeActionId)!;
  assert.ok(carriedOverVisits > 10, `expected the spiked move to accumulate many of the 40 visits, got ${carriedOverVisits}`);

  const nextPosition: Position = applyMove(initial, first.move);
  const smallTarget = 5; // deliberately smaller than carriedOverVisits
  const second = await tree.getMove(nextPosition, smallTarget);

  // If the subtree were rebuilt from scratch, visitsUsed would be exactly smallTarget (freshly
  // achieved). Reuse means the reused node already started above that target, so runToVisitCount
  // does zero new simulations and simply reports the carried-over count.
  assert.equal(second.visitsUsed, carriedOverVisits, 'expected the carried-over visit count, not a fresh count capped at the small target');
});

test('SearchTree starts fresh when given a position it never explored', async () => {
  const initial = createInitialPosition();
  const legalMoves = generateLegalMoves(initial).filter((m) => !m.isPass);
  const tree = new SearchTree(uniformEvaluator());
  await tree.getMove(initial, 10);

  // A position two plies deep via a move the first search never explored as a real subtree root.
  const otherMove = legalMoves[legalMoves.length - 1]!;
  const unrelatedPosition = applyMove(initial, otherMove);
  const result = await tree.getMove(unrelatedPosition, 8);
  assert.equal(result.visitsUsed, 8, 'an unexplored position should search fresh up to the requested visit count');
});
