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
  /** Whether reaching a bikjang position (both generals facing on an open file) can end the game. */
  readonly bikjangEndsGame: boolean;
  /** What a bikjang resolves to, when bikjangEndsGame is true. */
  readonly bikjangResult: 'draw';
  readonly materialCountingAdjudication: boolean;
  /** Number of times a position may repeat before the repetition rule triggers. */
  readonly repetitionLimit: number;
  /** Plies without a capture before the no-capture limit triggers adjudication. */
  readonly noCaptureMoveLimit: number;
  /**
   * Points added to Han's material total before adjudication comparison, compensating Han for
   * moving second. Commonly-cited value pending verification against an accessible official
   * source (see docs/RESEARCH.md - kja.or.kr was inaccessible during Phase 0 research).
   */
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
  /** Serialized board+side keys for every position reached so far, oldest first; used for repetition detection. */
  readonly positionHistory: ReadonlyArray<string>;
}

export type GameResultKind =
  | 'checkmate'
  | 'resignation'
  | 'timeout'
  | 'bikjang'
  | 'repetition'
  | 'material-adjudication'
  | 'draw';

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
