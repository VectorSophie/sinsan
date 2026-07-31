import { BOARD_COLS, BOARD_ROWS, BOARD_SIZE } from '@sinsan/rules';
import type { Position } from '@sinsan/rules';

/** Must match training/model/network.py's PIECE_TYPES order and fen_to_planes() exactly - the
 * "same canonical mapping across Python and TypeScript" requirement (Section 9) applies to the
 * input representation just as much as the action space. Cross-checked in
 * tests/model/model-runtime-parity.test.ts against real PyTorch-produced planes, not just by
 * matching this comment to the Python source by eye. */
const PIECE_TYPE_ORDER = ['chariot', 'cannon', 'horse', 'elephant', 'guard', 'general', 'soldier'] as const;
export const NUM_PIECE_PLANES = PIECE_TYPE_ORDER.length * 2; // 14
export const NUM_INPUT_PLANES = NUM_PIECE_PLANES + 1; // +1 side-to-move

/** Absolute (non-canonicalized) board encoding - the smoke-tier simplification documented in
 * docs/MODEL_DESIGN.md and training/model/network.py: fixed cho/han planes plus an explicit
 * side-to-move plane, rather than a perspective-relative "mine/theirs" canonicalization. */
export function positionToPlanes(position: Position): Float32Array {
  const planes = new Float32Array(NUM_INPUT_PLANES * BOARD_SIZE);
  for (let sq = 0; sq < BOARD_SIZE; sq++) {
    const piece = position.board[sq];
    if (!piece) continue;
    const typeIndex = PIECE_TYPE_ORDER.indexOf(piece.type);
    const sideOffset = piece.side === 'cho' ? 0 : PIECE_TYPE_ORDER.length;
    planes[(sideOffset + typeIndex) * BOARD_SIZE + sq] = 1.0;
  }
  if (position.sideToMove === 'cho') {
    planes.fill(1.0, NUM_PIECE_PLANES * BOARD_SIZE, NUM_INPUT_PLANES * BOARD_SIZE);
  }
  return planes;
}

export { BOARD_ROWS, BOARD_COLS };
