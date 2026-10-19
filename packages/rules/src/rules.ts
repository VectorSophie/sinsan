import { BOARD_SIZE, colOf, otherSide, rowOf } from './board.ts';
import { generatePseudoLegalMoves, isSquareAttackedBy } from './moves.ts';
import { findGeneral, pieceAt, positionKey } from './position.ts';
import { ruleProfileFor } from './rule-profiles.ts';
import type { Clock, GameResult, GameResultKind, Move, PieceType, PlyRecord, Position, Side } from './types.ts';

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

/** Board-only successor: no history bookkeeping. Used for legality tests, where only the
 * resulting board matters and computing check/bikjang flags per candidate would be wasted work. */
function playOnBoard(position: Position, move: Move): Position {
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
  return {
    board,
    sideToMove: otherSide(position.sideToMove),
    ruleProfile: position.ruleProfile,
    setupCho: position.setupCho,
    setupHan: position.setupHan,
    noCapturePly,
    moveNumber: position.moveNumber + 1,
    history: position.history,
  };
}

function plyRecordFor(position: Position, passed: boolean): PlyRecord {
  return { key: positionKey(position), passed, inCheck: isCheck(position), bikjang: isBikjang(position) };
}

export function applyMove(position: Position, move: Move): Position {
  const next = playOnBoard(position, move);
  const previous = position.history.slice();
  const last = previous[previous.length - 1];
  // Root records are created without flags (building a Position doesn't depend on this module).
  if (last && last.inCheck === undefined) previous[previous.length - 1] = plyRecordFor(position, last.passed);
  return { ...next, history: [...previous, plyRecordFor(next, move.isPass)] };
}

/**
 * Pseudo-legal moves, minus any that would leave the mover's own general in check, and - only
 * when bikjang already exists in the CURRENT position, under profiles where bikjangEndsGame is
 * true - minus any non-pass move that fails to resolve it (i.e. still leaves the two generals
 * facing on an open file after the move).
 *
 * Found via differential testing against pyffish (Section 8.4), not assumed, and refined twice
 * after two rounds of real counterexamples:
 *
 * 1. First finding: an earlier version of this engine generated moves pyffish rejected whenever
 *    bikjang was already on the board, always in a pattern where pyffish's few remaining legal
 *    moves either interposed a piece on the open file or were pass.
 * 2. That led to an over-broad first fix ("any non-pass move landing in bikjang is illegal"),
 *    which then produced the *opposite* mismatch: pyffish allowed a general move that newly
 *    created bikjang from a position that did *not* have bikjang before the move, which the
 *    over-broad version wrongly rejected.
 *
 * Net rule, matching both rounds of evidence: bikjang must be *resolved*, not *avoided* -
 * creating it from a clean position is fine, only failing to escape an existing one isn't.
 * `modern`/`casual` (bikjangEndsGame: false) never restrict on this basis, matching the existing
 * RuleProfile split, which is why this reuses that flag rather than adding a new one. See
 * docs/RULES.md and artifacts/rule-mismatches/ for the record.
 */
export function generateLegalMoves(position: Position): Move[] {
  const side = position.sideToMove;
  const profile = ruleProfileFor(position.ruleProfile);
  const mustResolveBikjang = profile.bikjangEndsGame && isBikjang(position);
  const legal: Move[] = [];
  for (const move of generatePseudoLegalMoves(position, side)) {
    // While bikjang stands, passing is always legal, even in check (FSF Position::legal).
    if (move.isPass && mustResolveBikjang) {
      legal.push(move);
      continue;
    }
    const next = playOnBoard(position, move);
    if (isSideInCheck(next, side)) continue;
    // Capturing the general only arises after a pass in check during bikjang, a position that is
    // already over (bikjang stood twice); skip the bikjang test there, as the Rust engine does.
    const capturesGeneral = position.board[move.to]?.type === 'general';
    if (!move.isPass && mustResolveBikjang && !capturesGeneral && isBikjang(next)) continue;
    legal.push(move);
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

/** Point-count winner (Fairy-Stockfish JANGGI_MATERIAL): Cho wins only if its point balance
 * exceeds Han's compensation, so with the standard 1.5 there is never a tie. */
function pointWinner(position: Position): Side {
  const profile = ruleProfileFor(position.ruleProfile);
  return getMaterialScore(position) - profile.hanCompensationPoints > 0 ? 'cho' : 'han';
}

function decided(position: Position, kind: GameResultKind): GameResult {
  const profile = ruleProfileFor(position.ruleProfile);
  return { winner: profile.materialCountingAdjudication ? pointWinner(position) : null, kind };
}

/**
 * Game result, following Fairy-Stockfish's janggi game-end logic exactly (the same rules as the
 * Rust engine in engine/, which is differentially tested against pyffish):
 *
 * 1. No legal moves: the side to move loses (checkmate, or stalemate, which FSF also scores as a loss).
 * 2. Bikjang in this and the previous position, or two consecutive passes: decided by points.
 * 3. noCaptureMoveLimit plies without a capture: decided by points.
 * 4. The position occurring repetitionLimit times within the no-capture window: if one side gave
 *    check on every move of the cycle, that side loses; otherwise decided by points.
 *
 * "Decided by points" becomes a draw under profiles without material counting.
 */
export function getGameResult(position: Position, clock?: Clock): GameResult | null {
  if (clock) {
    if (clock.choMsRemaining <= 0) return { winner: 'han', kind: 'timeout' };
    if (clock.hanMsRemaining <= 0) return { winner: 'cho', kind: 'timeout' };
  }

  const inCheck = isCheck(position);
  if (generateLegalMoves(position).length === 0) {
    return { winner: otherSide(position.sideToMove), kind: inCheck ? 'checkmate' : 'stalemate' };
  }

  const profile = ruleProfileFor(position.ruleProfile);
  const history = position.history;
  const last = history.length - 1;
  const current = history[last];
  const previous = history[last - 1];
  if (current && previous) {
    if (profile.bikjangEndsGame && isBikjang(position) && previous.bikjang) return decided(position, 'bikjang');
    if (current.passed && previous.passed) return decided(position, 'double-pass');
  }

  if (position.noCapturePly >= profile.noCaptureMoveLimit) return decided(position, 'move-limit');

  // Repetition: walk back two plies at a time within the no-capture window, tracking whether
  // either side has given check on every move since (perpetual check loses).
  const end = Math.min(position.noCapturePly, last);
  if (end >= 4 && current) {
    const at = (back: number): PlyRecord => history[last - back]!;
    let perpetualThem = inCheck && at(2).inCheck === true;
    let perpetualUs = at(1).inCheck === true && at(3).inCheck === true;
    let count = 0;
    for (let i = 4; i <= end; i += 2) {
      const record = at(i);
      perpetualThem &&= record.inCheck === true;
      if (record.key === current.key && ++count + 1 >= profile.repetitionLimit) {
        const us = position.sideToMove;
        if (perpetualThem && !perpetualUs) return { winner: us, kind: 'perpetual-check' };
        if (perpetualUs && !perpetualThem) return { winner: otherSide(us), kind: 'perpetual-check' };
        return decided(position, 'repetition');
      }
      if (i + 1 <= end) perpetualUs &&= at(i + 1).inCheck === true;
    }
  }
  return null;
}
