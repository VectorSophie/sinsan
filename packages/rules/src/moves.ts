import {
  BOARD_SIZE,
  ORTHOGONAL_DIRECTIONS,
  PALACE_DIAGONAL_RAYS,
  PALACE_OF,
  colOf,
  forwardRowDelta,
  inBounds,
  orthogonalRay,
  rowOf,
  squareOf,
} from './board.ts';
import { pieceAt } from './position.ts';
import type { Move, Piece, Position, Side, Square } from './types.ts';

export const PASS_MOVE: Move = { from: -1, to: -1, isPass: true };

function raysFor(square: Square, includeOrthogonal: boolean, includePalaceDiagonals: boolean): Square[][] {
  const rays: Square[][] = [];
  if (includeOrthogonal) {
    for (const [dr, dc] of ORTHOGONAL_DIRECTIONS) rays.push(orthogonalRay(square, dr, dc));
  }
  if (includePalaceDiagonals) {
    const diagonalRays = PALACE_DIAGONAL_RAYS.get(square);
    if (diagonalRays) rays.push(...diagonalRays);
  }
  return rays;
}

/** Slide along each ray until blocked: empty squares are moves, the first enemy piece hit is a
 * capturing move (and the ray stops there), the first friendly piece hit stops the ray with no move. */
function slideMoves(position: Position, from: Square, side: Side, rays: Square[][]): Move[] {
  const moves: Move[] = [];
  for (const ray of rays) {
    for (const to of ray) {
      const occupant = pieceAt(position, to);
      if (!occupant) {
        moves.push({ from, to, isPass: false });
        continue;
      }
      if (occupant.side !== side) moves.push({ from, to, isPass: false });
      break;
    }
  }
  return moves;
}

function chariotMoves(position: Position, from: Square, side: Side): Move[] {
  return slideMoves(position, from, side, raysFor(from, true, true));
}

/** Cannon: jumps over exactly one non-cannon piece (the "screen") per ray, of either side, then
 * may land on any empty square beyond it, or capture the first non-cannon enemy piece beyond it.
 * Cannons cannot capture cannons, and a cannon cannot itself be used as the screen. */
function cannonMoves(position: Position, from: Square, side: Side): Move[] {
  const moves: Move[] = [];
  for (const ray of raysFor(from, true, true)) {
    let screenIndex = -1;
    for (let i = 0; i < ray.length; i++) {
      if (pieceAt(position, ray[i]!)) {
        screenIndex = i;
        break;
      }
    }
    if (screenIndex === -1) continue; // no screen anywhere on this ray - cannon cannot move here
    const screen = pieceAt(position, ray[screenIndex]!)!;
    if (screen.type === 'cannon') continue; // illegal screen

    for (let i = screenIndex + 1; i < ray.length; i++) {
      const occupant = pieceAt(position, ray[i]!);
      if (!occupant) {
        moves.push({ from, to: ray[i]!, isPass: false });
        continue;
      }
      if (occupant.side !== side && occupant.type !== 'cannon') {
        moves.push({ from, to: ray[i]!, isPass: false });
      }
      break;
    }
  }
  return moves;
}

interface StepCandidate {
  readonly blockers: readonly Square[];
  readonly to: Square;
}

function pushIfInBounds(row: number, col: number): Square | null {
  return inBounds(row, col) ? squareOf(row, col) : null;
}

function horseCandidates(from: Square): StepCandidate[] {
  const r = rowOf(from);
  const c = colOf(from);
  const legs: ReadonlyArray<{ leg: readonly [number, number]; dests: ReadonlyArray<readonly [number, number]> }> = [
    { leg: [-1, 0], dests: [[-2, -1], [-2, 1]] },
    { leg: [1, 0], dests: [[2, -1], [2, 1]] },
    { leg: [0, -1], dests: [[-1, -2], [1, -2]] },
    { leg: [0, 1], dests: [[-1, 2], [1, 2]] },
  ];
  const candidates: StepCandidate[] = [];
  for (const { leg, dests } of legs) {
    const legSquare = pushIfInBounds(r + leg[0], c + leg[1]);
    if (legSquare === null) continue;
    for (const [dr, dc] of dests) {
      const to = pushIfInBounds(r + dr, c + dc);
      if (to === null) continue;
      candidates.push({ blockers: [legSquare], to });
    }
  }
  return candidates;
}

function elephantCandidates(from: Square): StepCandidate[] {
  const r = rowOf(from);
  const c = colOf(from);
  const legs: ReadonlyArray<{
    leg: readonly [number, number];
    diagonals: ReadonlyArray<readonly [number, number]>;
  }> = [
    { leg: [-1, 0], diagonals: [[-1, -1], [-1, 1]] },
    { leg: [1, 0], diagonals: [[1, -1], [1, 1]] },
    { leg: [0, -1], diagonals: [[-1, -1], [1, -1]] },
    { leg: [0, 1], diagonals: [[-1, 1], [1, 1]] },
  ];
  const candidates: StepCandidate[] = [];
  for (const { leg, diagonals } of legs) {
    const legSquare = pushIfInBounds(r + leg[0], c + leg[1]);
    if (legSquare === null) continue;
    for (const [dr, dc] of diagonals) {
      const midR = r + leg[0] + dr;
      const midC = c + leg[1] + dc;
      const midSquare = pushIfInBounds(midR, midC);
      if (midSquare === null) continue;
      const to = pushIfInBounds(midR + dr, midC + dc);
      if (to === null) continue;
      candidates.push({ blockers: [legSquare, midSquare], to });
    }
  }
  return candidates;
}

function stepMoves(position: Position, from: Square, side: Side, candidates: StepCandidate[]): Move[] {
  const moves: Move[] = [];
  for (const { blockers, to } of candidates) {
    if (blockers.some((sq) => pieceAt(position, sq) !== null)) continue;
    const occupant = pieceAt(position, to);
    if (occupant && occupant.side === side) continue;
    moves.push({ from, to, isPass: false });
  }
  return moves;
}

function horseMoves(position: Position, from: Square, side: Side): Move[] {
  return stepMoves(position, from, side, horseCandidates(from));
}

function elephantMoves(position: Position, from: Square, side: Side): Move[] {
  return stepMoves(position, from, side, elephantCandidates(from));
}

/** Guard and general move identically: one step, orthogonal or (on the palace's diagonal lines)
 * diagonal, and never leave their own palace. */
function palaceStepMoves(position: Position, from: Square, side: Side): Move[] {
  const moves: Move[] = [];
  const r = rowOf(from);
  const c = colOf(from);
  for (const [dr, dc] of ORTHOGONAL_DIRECTIONS) {
    const to = pushIfInBounds(r + dr, c + dc);
    if (to === null || PALACE_OF.get(to) !== side) continue;
    const occupant = pieceAt(position, to);
    if (occupant && occupant.side === side) continue;
    moves.push({ from, to, isPass: false });
  }
  const diagonalRays = PALACE_DIAGONAL_RAYS.get(from);
  if (diagonalRays) {
    for (const ray of diagonalRays) {
      const to = ray[0]!;
      if (PALACE_OF.get(to) !== side) continue;
      const occupant = pieceAt(position, to);
      if (occupant && occupant.side === side) continue;
      moves.push({ from, to, isPass: false });
    }
  }
  return moves;
}

/** Soldier: forward or sideways one step anywhere on the board, plus a forward-diagonal step
 * along a palace's diagonal lines when standing on one of that palace's five special squares. */
function soldierMoves(position: Position, from: Square, side: Side): Move[] {
  const moves: Move[] = [];
  const r = rowOf(from);
  const c = colOf(from);
  const forward = forwardRowDelta(side);
  const orthogonalCandidates: ReadonlyArray<readonly [number, number]> = [
    [forward, 0],
    [0, -1],
    [0, 1],
  ];
  for (const [dr, dc] of orthogonalCandidates) {
    const to = pushIfInBounds(r + dr, c + dc);
    if (to === null) continue;
    const occupant = pieceAt(position, to);
    if (occupant && occupant.side === side) continue;
    moves.push({ from, to, isPass: false });
  }
  const diagonalRays = PALACE_DIAGONAL_RAYS.get(from);
  if (diagonalRays) {
    for (const ray of diagonalRays) {
      const to = ray[0]!;
      if (Math.sign(rowOf(to) - r) !== forward) continue;
      const occupant = pieceAt(position, to);
      if (occupant && occupant.side === side) continue;
      moves.push({ from, to, isPass: false });
    }
  }
  return moves;
}

export function pseudoLegalMovesFrom(position: Position, from: Square): Move[] {
  const piece = pieceAt(position, from);
  if (!piece) return [];
  const { type, side } = piece;
  switch (type) {
    case 'chariot':
      return chariotMoves(position, from, side);
    case 'cannon':
      return cannonMoves(position, from, side);
    case 'horse':
      return horseMoves(position, from, side);
    case 'elephant':
      return elephantMoves(position, from, side);
    case 'guard':
    case 'general':
      return palaceStepMoves(position, from, side);
    case 'soldier':
      return soldierMoves(position, from, side);
  }
}

/** All pseudo-legal moves for `side` (no self-check filtering), plus the always-available pass. */
export function generatePseudoLegalMoves(position: Position, side: Side): Move[] {
  const moves: Move[] = [PASS_MOVE];
  for (let sq = 0; sq < BOARD_SIZE; sq++) {
    const piece: Piece | null = position.board[sq] ?? null;
    if (piece && piece.side === side) moves.push(...pseudoLegalMovesFrom(position, sq));
  }
  return moves;
}

export function isSquareAttackedBy(position: Position, square: Square, attacker: Side): boolean {
  for (let sq = 0; sq < BOARD_SIZE; sq++) {
    const piece = position.board[sq] ?? null;
    if (!piece || piece.side !== attacker) continue;
    for (const move of pseudoLegalMovesFrom(position, sq)) {
      if (move.to === square) return true;
    }
  }
  return false;
}
