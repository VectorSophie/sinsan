import { encodeMove } from '@sinsan/action-space';
import { applyMove, generateLegalMoves, getGameResult, positionKey } from '@sinsan/rules';
import type { Move, Position, Side } from '@sinsan/rules';
import { MCTSNode } from './node.ts';

export interface PositionEvaluation {
  readonly policyLogits: Float32Array;
  readonly value: number;
}

/** Caller supplies this - keeps packages/search decoupled from any specific feature encoding or
 * model-runtime detail (docs/ARCHITECTURE.md: search depends on rules for legality and "the
 * model" for priors/value, but doesn't need to depend on *how* those are produced). */
export type PositionEvaluator = (position: Position) => Promise<PositionEvaluation> | PositionEvaluation;

export interface SearchOptions {
  readonly visits: number;
  /** PUCT exploration constant. 1.5 is a commonly-used AlphaZero-style default, not tuned or
   * calibrated for Janggi specifically - an explicit, disclosed placeholder (Section 14/16 don't
   * specify one; this needs real arena-strength tuning later, same caution as elsewhere in this
   * project about not presenting untested constants as settled). */
  readonly cPuct?: number;
}

export interface SearchResult {
  readonly move: Move;
  /** action id -> visit count, root children only - what the analysis screen (Section 17.2)
   * would display as candidate moves. */
  readonly visitCounts: ReadonlyMap<number, number>;
  readonly rootValue: number;
  readonly visitsUsed: number;
}

/** Terminal-position value from the perspective of `perspectiveSide` (the side to move at the
 * node being evaluated, i.e. whose turn it would be if the game weren't over) - checkmate is
 * always a loss for whoever has no moves, draws are 0, material adjudication depends on which
 * side actually won. */
function terminalValueFor(perspectiveSide: Side, position: Position): number | null {
  const result = getGameResult(position);
  if (!result) return null;
  if (result.winner === null) return 0;
  return result.winner === perspectiveSide ? 1 : -1;
}

function softmaxMasked(logits: Float32Array, actionIds: readonly number[]): Map<number, number> {
  let max = -Infinity;
  for (const id of actionIds) max = Math.max(max, logits[id]!);
  let sum = 0;
  const exps = new Map<number, number>();
  for (const id of actionIds) {
    const e = Math.exp(logits[id]! - max);
    exps.set(id, e);
    sum += e;
  }
  const probs = new Map<number, number>();
  for (const [id, e] of exps) probs.set(id, e / sum);
  return probs;
}

/** Expands a leaf: terminal check first (no NN call needed - Section 16's "terminal-result
 * handling" requirement), otherwise runs the evaluator and initializes children priors from the
 * legal-action-masked policy softmax. Returns the value to back up, from this leaf's own
 * to-move perspective. */
async function expand(node: MCTSNode, evaluate: PositionEvaluator): Promise<number> {
  const terminal = terminalValueFor(node.position.sideToMove, node.position);
  if (terminal !== null) {
    node.terminal = true;
    node.expanded = true;
    return terminal;
  }

  const legalMoves = generateLegalMoves(node.position);
  const actionIds: number[] = [];
  for (const move of legalMoves) {
    const actionId = encodeMove(move);
    if (actionId === undefined) continue; // shouldn't happen for a real legal move
    actionIds.push(actionId);
    node.movesByAction.set(actionId, move);
  }

  const { policyLogits, value } = await evaluate(node.position);
  const priors = softmaxMasked(policyLogits, actionIds);
  for (const [actionId, prior] of priors) node.priors.set(actionId, prior);
  node.expanded = true;
  return value;
}

function selectChildAction(node: MCTSNode, cPuct: number): number {
  const sqrtParentVisits = Math.sqrt(Math.max(1, node.visitCount));
  let bestAction = -1;
  let bestScore = -Infinity;
  for (const [actionId, prior] of node.priors) {
    const child = node.children.get(actionId);
    const childVisits = child?.visitCount ?? 0;
    // Q from `node`'s perspective is the negation of the child's own-perspective mean value -
    // standard negamax sign flip (docs/MODEL_DESIGN.md's value convention is always
    // current-player-perspective, and applyMove flips sideToMove every ply).
    const q = child ? -child.meanValue : 0;
    const u = (cPuct * prior * sqrtParentVisits) / (1 + childVisits);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      bestAction = actionId;
    }
  }
  return bestAction;
}

/** One MCTS simulation: select down to a leaf (creating child nodes lazily along the way),
 * expand it, and back up the value with alternating sign (negamax). */
async function simulate(root: MCTSNode, evaluate: PositionEvaluator, cPuct: number): Promise<void> {
  const path: MCTSNode[] = [root];
  let node = root;

  while (node.expanded && !node.terminal) {
    const actionId = selectChildAction(node, cPuct);
    let child = node.children.get(actionId);
    if (!child) {
      const move = node.movesByAction.get(actionId)!;
      child = new MCTSNode(applyMove(node.position, move), node, move);
      node.children.set(actionId, child);
    }
    node = child;
    path.push(node);
  }

  let value = node.terminal ? terminalValueFor(node.position.sideToMove, node.position)! : await expand(node, evaluate);

  for (let i = path.length - 1; i >= 0; i--) {
    const n = path[i]!;
    n.visitCount++;
    n.valueSum += value;
    value = -value;
  }
}

function buildResult(root: MCTSNode, visitsUsed: number): SearchResult {
  if (root.terminal) {
    throw new Error(
      'search: root position is already game-over (checkmate/bikjang/etc) - callers must check ' +
        'getGameResult() before searching for a move, same as apps/web already does via its own ' +
        'gameOver gating.',
    );
  }

  let bestAction = -1;
  let bestVisits = -1;
  const visitCounts = new Map<number, number>();
  for (const [actionId, child] of root.children) {
    visitCounts.set(actionId, child.visitCount);
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestAction = actionId;
    }
  }

  // No children were ever selected (visits=1, or every child collapsed to the same action) -
  // fall back to the single highest-prior legal move rather than leaving the caller with nothing.
  if (bestAction === -1) {
    for (const [actionId, prior] of root.priors) {
      if (bestAction === -1 || prior > (root.priors.get(bestAction) ?? -Infinity)) bestAction = actionId;
    }
  }

  return {
    move: root.movesByAction.get(bestAction)!,
    visitCounts,
    rootValue: root.meanValue,
    visitsUsed,
  };
}

async function runToVisitCount(root: MCTSNode, evaluate: PositionEvaluator, cPuct: number, visits: number): Promise<number> {
  if (!root.expanded) {
    const rootValue = await expand(root, evaluate);
    root.visitCount = 1;
    root.valueSum = rootValue;
  }
  let visitsUsed = root.visitCount;
  for (; visitsUsed < visits; visitsUsed++) {
    if (root.terminal) break; // nothing to search from an already-over position
    await simulate(root, evaluate, cPuct);
  }
  return visitsUsed;
}

/** Runs PUCT/MCTS from `rootPosition` for `options.visits` simulations and returns the move with
 * the most root-child visits (standard AlphaZero-style competitive selection - no temperature
 * sampling here; that's Beginner-tier raw-policy play in apps/web, a separate code path that
 * doesn't use this function at all). Builds a fresh tree every call - use `SearchTree` instead
 * when you can reuse the tree across a game's moves (Section 16's tree-reuse requirement). */
export async function search(rootPosition: Position, evaluate: PositionEvaluator, options: SearchOptions): Promise<SearchResult> {
  const root = new MCTSNode(rootPosition, null, undefined);
  const visitsUsed = await runToVisitCount(root, evaluate, options.cPuct ?? 1.5, options.visits);
  return buildResult(root, visitsUsed);
}

/**
 * Persists the search tree across moves (Section 16: "tree reuse"). Call `getMove` with each
 * position your game actually reaches, in order; if that position matches a child of the
 * previous root (i.e. the game advanced by a move this tree already explored, from either side),
 * that subtree - including its priors, visit counts, and any deeper exploration - is reused
 * instead of discarded. Positions that don't match any known child (a different game, an undo,
 * or a position this tree never explored) start a fresh subtree from scratch.
 *
 * Known limitation: matching reuses `positionKey` (board + side to move only), not the full
 * position (including move-order-dependent repetition history). In the normal call pattern -
 * always advancing via this tree's own returned move - the reused child is exactly the position
 * that move produces, so this is exact. A transposition (a different move order reaching the
 * same board+side) could in principle reuse a node whose repetition history doesn't match the
 * real game's, very slightly affecting repetition-adjudication accuracy deep in that subtree.
 * Not fixed here - full-history matching would close it but adds real cost for a narrow case.
 */
export class SearchTree {
  private root: MCTSNode | undefined;
  private readonly evaluate: PositionEvaluator;
  private readonly cPuct: number;

  constructor(evaluate: PositionEvaluator, cPuct: number = 1.5) {
    this.evaluate = evaluate;
    this.cPuct = cPuct;
  }

  async getMove(position: Position, visits: number): Promise<SearchResult> {
    this.root = this.reuseOrCreateRoot(position);
    const visitsUsed = await runToVisitCount(this.root, this.evaluate, this.cPuct, visits);
    return buildResult(this.root, visitsUsed);
  }

  /** Discards the tree - call when starting a new game. */
  reset(): void {
    this.root = undefined;
  }

  private reuseOrCreateRoot(position: Position): MCTSNode {
    if (this.root) {
      const targetKey = positionKey(position);
      for (const child of this.root.children.values()) {
        if (positionKey(child.position) === targetKey) return child;
      }
    }
    return new MCTSNode(position, null, undefined);
  }
}
