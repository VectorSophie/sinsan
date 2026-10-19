export type Side = 'cho' | 'han';

export type PieceType =
  | 'chariot'
  | 'cannon'
  | 'horse'
  | 'elephant'
  | 'guard'
  | 'general'
  | 'soldier';

export interface Piece {
  readonly type: PieceType;
  readonly side: Side;
}

/** 0-88, row-major: square = row * 9 + col. Row 0 is Han's back rank, row 9 is Cho's back rank
 * (matches Fairy-Stockfish's own Janggi FEN rank order, confirmed against its default StartFEN
 * during Phase 0 research - see docs/RULES.md). */
export type Square = number;

/**
 * Formation identifiers are our own stable English names (Section 8.2 warns against assuming
 * informal English names online are standardized). The horse/elephant column layout each name
 * produces was cross-checked against Fairy-Stockfish's default Janggi StartFEN - see
 * docs/RULES.md for the derivation and that cross-check.
 */
export type Formation = 'masang-sangma' | 'sangma-masang' | 'masang-masang' | 'sangma-sangma';

export type RuleProfileId = 'kja' | 'traditional' | 'modern' | 'casual';

export interface RuleProfile {
  readonly id: RuleProfileId;
  /** Bikjang standing in two consecutive positions ends the game (and while it stands, non-pass
   * moves must break it). */
  readonly bikjangEndsGame: boolean;
  /** Games ended by bikjang, double pass, the no-capture limit or plain repetition are decided by
   * points (Fairy-Stockfish JANGGI_MATERIAL) instead of drawn. */
  readonly materialCountingAdjudication: boolean;
  /** The game ends when the current position has occurred this many times. */
  readonly repetitionLimit: number;
  /** Plies without a capture after which the game ends. */
  readonly noCaptureMoveLimit: number;
  /** Points added to Han's total for point counting (compensation for moving second). */
  readonly hanCompensationPoints: number;
}

export interface Move {
  readonly from: Square;
  readonly to: Square;
  readonly isPass: boolean;
}

export interface Position {
  /** length-90 board, index = square. Immutable - applyMove returns a new Position. */
  readonly board: ReadonlyArray<Piece | null>;
  readonly sideToMove: Side;
  readonly ruleProfile: RuleProfileId;
  readonly setupCho: Formation;
  readonly setupHan: Formation;
  /** Plies since the last capture; drives the no-capture move limit. */
  readonly noCapturePly: number;
  readonly moveNumber: number;
  /** One record per position reached so far, oldest first; the last entry is this position. */
  readonly history: ReadonlyArray<PlyRecord>;
}

/** Per-position facts needed for Fairy-Stockfish-style game endings (repetition, perpetual
 * check, persistent bikjang, double pass). inCheck/bikjang are filled in lazily by applyMove for
 * the root position, so building a Position never needs the rules module. */
export interface PlyRecord {
  readonly key: string;
  /** The move that led to this position was a pass. */
  readonly passed: boolean;
  readonly inCheck?: boolean;
  readonly bikjang?: boolean;
}

export type GameResultKind =
  | 'checkmate'
  | 'stalemate'
  | 'resignation'
  | 'timeout'
  | 'bikjang'
  | 'double-pass'
  | 'move-limit'
  | 'repetition'
  | 'perpetual-check';

export interface GameResult {
  readonly winner: Side | null;
  readonly kind: GameResultKind;
}

export interface CreatePositionOptions {
  readonly ruleProfile?: RuleProfileId;
  readonly setupCho?: Formation;
  readonly setupHan?: Formation;
}

export interface Clock {
  readonly choMsRemaining: number;
  readonly hanMsRemaining: number;
}
