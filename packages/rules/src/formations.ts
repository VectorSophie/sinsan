import type { Formation, PieceType } from './types.ts';

interface BackRankLayout {
  /** column 1 (next to the chariot) and column 2 (next to the guard), left of center */
  readonly left: readonly [PieceType, PieceType];
  /** column 6 (next to the guard) and column 7 (next to the chariot), right of center */
  readonly right: readonly [PieceType, PieceType];
}

/**
 * Each formation name is a left-token + right-token pair ("마상" = horse,elephant read in board
 * order; "상마" = elephant,horse), applied directly to each side's own left pair (cols 1-2) and
 * right pair (cols 6-7) - not mirrored. This was cross-checked, not assumed: Fairy-Stockfish's
 * default Janggi StartFEN row ("rnba1abnr") decodes to col1=horse, col2=elephant, col6=elephant,
 * col7=horse, which is exactly what masang-sangma produces below, and the spec's own stated
 * default is masang-sangma - see docs/RULES.md for the full derivation.
 */
const LAYOUTS: Record<Formation, BackRankLayout> = {
  'masang-sangma': { left: ['horse', 'elephant'], right: ['elephant', 'horse'] },
  'sangma-masang': { left: ['elephant', 'horse'], right: ['horse', 'elephant'] },
  'masang-masang': { left: ['horse', 'elephant'], right: ['horse', 'elephant'] },
  'sangma-sangma': { left: ['elephant', 'horse'], right: ['elephant', 'horse'] },
};

export function backRankLayout(formation: Formation): BackRankLayout {
  return LAYOUTS[formation];
}
