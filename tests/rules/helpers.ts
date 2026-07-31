import { BOARD_SIZE, squareOf } from '../../packages/rules/src/board.ts';
import { positionKey } from '../../packages/rules/src/position.ts';
import type { Piece, PieceType, Position, RuleProfileId, Side, Square } from '../../packages/rules/src/types.ts';

export function sq(row: number, col: number): Square {
  return squareOf(row, col);
}

export function piece(type: PieceType, side: Side): Piece {
  return { type, side };
}

interface CustomPositionOptions {
  readonly pieces: ReadonlyArray<readonly [Square, Piece]>;
  readonly sideToMove?: Side;
  readonly ruleProfile?: RuleProfileId;
  readonly noCapturePly?: number;
  readonly extraHistory?: ReadonlyArray<string>;
}

/** Builds a minimal custom Position for testing a single piece/rule in isolation, without going
 * through createInitialPosition's full starting layout. */
export function customPosition(options: CustomPositionOptions): Position {
  const board: (Piece | null)[] = new Array(BOARD_SIZE).fill(null);
  for (const [square, p] of options.pieces) board[square] = p;
  const base: Position = {
    board,
    sideToMove: options.sideToMove ?? 'cho',
    ruleProfile: options.ruleProfile ?? 'kja',
    setupCho: 'masang-sangma',
    setupHan: 'masang-sangma',
    noCapturePly: options.noCapturePly ?? 0,
    moveNumber: 1,
    positionHistory: [],
  };
  const history = [...(options.extraHistory ?? []), positionKey(base)];
  return { ...base, positionHistory: history };
}
