import { BOARD_COLS, BOARD_ROWS, BOARD_SIZE, colOf, rowOf, squareOf } from '@sinsan/rules';
import type { Move, Square } from '@sinsan/rules';
import { REFLECTED_TEMPLATE_INDEX, TEMPLATE_COUNT, TEMPLATES } from './templates.ts';

export { TEMPLATES, TEMPLATE_COUNT } from './templates.ts';
export type { Template } from './templates.ts';

/** 90 origins x 60 templates + 1 pass action (Section 9). */
export const ACTION_SPACE_SIZE = BOARD_SIZE * TEMPLATE_COUNT + 1; // 5401
export const PASS_ACTION = BOARD_SIZE * TEMPLATE_COUNT; // 5400

/** Encodes a move to its canonical action id. Returns undefined if no template explains the
 * move's (from -> to) delta - this should never happen for a move actually produced by
 * packages/rules (every piece's move generator only ever produces deltas covered by one of the
 * 60 templates; see docs/MODEL_DESIGN.md), so callers may treat undefined as a bug to investigate,
 * not a normal case to silently ignore. */
export function encodeMove(move: Move): number | undefined {
  if (move.isPass) return PASS_ACTION;
  const dr = rowOf(move.to) - rowOf(move.from);
  const dc = colOf(move.to) - colOf(move.from);
  const templateIndex = TEMPLATES.findIndex((t) => t.dr === dr && t.dc === dc);
  if (templateIndex === -1) return undefined;
  return move.from * TEMPLATE_COUNT + templateIndex;
}

export interface DecodedAction {
  readonly from: Square;
  readonly to: Square;
}

/** Decodes an action id back to a board move. Returns 'pass' for the pass action, or undefined
 * if the id is out of range or the template would step off the board from that origin (this is
 * the "legal-action mask must remove out-of-board moves" case from Section 9 - decode is a pure
 * geometric function and does not know about legality/occupancy, only board bounds). */
export function decodeAction(actionId: number): DecodedAction | 'pass' | undefined {
  if (actionId === PASS_ACTION) return 'pass';
  if (actionId < 0 || actionId >= PASS_ACTION) return undefined;
  const from = Math.floor(actionId / TEMPLATE_COUNT);
  const templateIndex = actionId % TEMPLATE_COUNT;
  const template = TEMPLATES[templateIndex]!;
  const row = rowOf(from) + template.dr;
  const col = colOf(from) + template.dc;
  if (row < 0 || row >= BOARD_ROWS || col < 0 || col >= BOARD_COLS) return undefined;
  return { from, to: squareOf(row, col) };
}

/** A 0/1 mask over the full action space, 1 for every action a legal move actually encodes to. */
export function legalActionMask(legalMoves: readonly Move[]): Uint8Array {
  const mask = new Uint8Array(ACTION_SPACE_SIZE);
  for (const move of legalMoves) {
    const actionId = encodeMove(move);
    if (actionId !== undefined) mask[actionId] = 1;
  }
  return mask;
}

/** Horizontal board reflection (column mirrored: col -> BOARD_COLS-1-col), used for symmetry
 * augmentation (Section 12.6). Pass reflects to itself. */
export function reflectActionHorizontally(actionId: number): number | undefined {
  if (actionId === PASS_ACTION) return PASS_ACTION;
  if (actionId < 0 || actionId >= PASS_ACTION) return undefined;
  const from = Math.floor(actionId / TEMPLATE_COUNT);
  const templateIndex = actionId % TEMPLATE_COUNT;
  const reflectedFrom = squareOf(rowOf(from), BOARD_COLS - 1 - colOf(from));
  const reflectedTemplate = REFLECTED_TEMPLATE_INDEX[templateIndex]!;
  return reflectedFrom * TEMPLATE_COUNT + reflectedTemplate;
}
