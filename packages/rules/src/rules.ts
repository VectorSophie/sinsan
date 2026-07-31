import { BOARD_SIZE, colOf, otherSide, rowOf } from './board.ts';
import { generatePseudoLegalMoves, isSquareAttackedBy } from './moves.ts';
import { findGeneral, pieceAt, positionKey } from './position.ts';
import { ruleProfileFor } from './rule-profiles.ts';
import type { Clock, GameResult, Move, PieceType, Position, Side } from './types.ts';

/**
 * Commonly-cited Janggi material point values (chariot 13, cannon 7, horse 5, elephant/guard 3,
 * soldier 2). Cited widely in Janggi commentary/tooling but not independently verified against an
 * accessible official source during Phase 0 research - see docs/RESEARCH.md. General is excluded
 * (0) since its capture ends the game before material scoring is relevant.
 */
export const PIECE_VALUES: Record<PieceType, number> = {
  chariot: 13,
  cannon: 7,
  horse: 5,
  elephant: 3,
  guard: 3,
  soldier: 2,
  general: 0,
};

export function isSideInCheck(position: Position, side: Side): boolean {
  const generalSquare = findGeneral(position, side);
  return isSquareAttackedBy(position, generalSquare, otherSide(side));
}

export function isCheck(position: Position): boolean {
  return isSideInCheck(position, position.sideToMove);
}

export function applyMove(position: Position, move: Move): Position {
  const board = position.board.slice();
  let noCapturePly = position.noCapturePly + 1;
  if (!move.isPass) {
    const moving = board[move.from] ?? null;
    if (!moving) throw new Error(`applyMove: no piece at origin square ${move.from}`);
    const captured = board[move.to] ?? null;
    if (captured) noCapturePly = 0;
    board[move.to] = moving;
    board[move.from] = null;
  }
  const withoutHistory: Position = {
    board,
    sideToMove: otherSide(position.sideToMove),
    ruleProfile: position.ruleProfile,
    setupCho: position.setupCho,
    setupHan: position.setupHan,
    noCapturePly,
    moveNumber: position.moveNumber + 1,
    positionHistory: position.positionHistory,
  };
  return { ...withoutHistory, positionHistory: [...position.positionHistory, positionKey(withoutHistory)] };
}

/** Pseudo-legal moves, minus any that would leave the mover's own general in check. */
export function generateLegalMoves(position: Position): Move[] {
  const side = position.sideToMove;
  const legal: Move[] = [];
  for (const move of generatePseudoLegalMoves(position, side)) {
    if (!isSideInCheck(applyMove(position, move), side)) legal.push(move);
  }
  return legal;
}

export function getMaterialScore(position: Position): number {
  let score = 0;
  for (let sq = 0; sq < BOARD_SIZE; sq++) {
    const piece = pieceAt(position, sq);
    if (!piece) continue;
    const value = PIECE_VALUES[piece.type];
    score += piece.side === 'cho' ? value : -value;
  }
  return score;
}

/** Both generals share a column with no piece of either side between them. */
export function isBikjang(position: Position): boolean {
  const choSquare = findGeneral(position, 'cho');
  const hanSquare = findGeneral(position, 'han');
  if (colOf(choSquare) !== colOf(hanSquare)) return false;
  const col = colOf(choSquare);
  const [top, bottom] = rowOf(choSquare) < rowOf(hanSquare) ? [choSquare, hanSquare] : [hanSquare, choSquare];
  for (let r = rowOf(top) + 1; r < rowOf(bottom); r++) {
    if (pieceAt(position, r * 9 + col)) return false;
  }
  return true;
}

/**
 * Known simplification: repetition beyond the profile's limit is adjudicated as a draw. Real
 * Janggi additionally distinguishes perpetual check (a loss for the checking side) from plain
 * repetition (a draw) - Fairy-Stockfish's own `perpetualCheckIllegal` flag implements this by
 * making the repeating checking move itself illegal, rather than adjudicating after the fact.
 * That distinction is not yet implemented here; it needs differential testing against
 * Fairy-Stockfish (Phase 3) before being treated as correct - see docs/RULES.md.
 */
export function getGameResult(position: Position, clock?: Clock): GameResult | null {
  if (clock) {
    if (clock.choMsRemaining <= 0) return { winner: 'han', kind: 'timeout' };
    if (clock.hanMsRemaining <= 0) return { winner: 'cho', kind: 'timeout' };
  }

  if (generateLegalMoves(position).length === 0) {
    return { winner: otherSide(position.sideToMove), kind: 'checkmate' };
  }

  const profile = ruleProfileFor(position.ruleProfile);

  if (profile.bikjangEndsGame && isBikjang(position)) {
    return { winner: null, kind: 'bikjang' };
  }

  const currentKey = positionKey(position);
  const repetitionCount = position.positionHistory.filter((key) => key === currentKey).length;
  if (repetitionCount > profile.repetitionLimit) {
    return { winner: null, kind: 'repetition' };
  }

  if (position.noCapturePly >= profile.noCaptureMoveLimit) {
    if (!profile.materialCountingAdjudication) return { winner: null, kind: 'draw' };
    const adjustedScore = getMaterialScore(position) - profile.hanCompensationPoints;
    if (Math.abs(adjustedScore) < 1e-9) return { winner: null, kind: 'draw' };
    return { winner: adjustedScore > 0 ? 'cho' : 'han', kind: 'material-adjudication' };
  }

  return null;
}
