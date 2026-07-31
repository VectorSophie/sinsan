import type { Side, Square } from './types.ts';

export const BOARD_COLS = 9;
export const BOARD_ROWS = 10;
export const BOARD_SIZE = BOARD_COLS * BOARD_ROWS;

export function squareOf(row: number, col: number): Square {
  return row * BOARD_COLS + col;
}

export function rowOf(square: Square): number {
  return Math.floor(square / BOARD_COLS);
}

export function colOf(square: Square): number {
  return square % BOARD_COLS;
}

export function inBounds(row: number, col: number): boolean {
  return row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS;
}

/** Cho's back rank is row 9; Cho advances toward row 0. Han's back rank is row 0; Han advances toward row 9. */
export function forwardRowDelta(side: Side): -1 | 1 {
  return side === 'cho' ? -1 : 1;
}

export function otherSide(side: Side): Side {
  return side === 'cho' ? 'han' : 'cho';
}

export function palaceRowOffset(side: Side): number {
  return side === 'cho' ? 7 : 0;
}

export const PALACE_OF = new Map<Square, Side>();

for (const side of ['han', 'cho'] as const) {
  const rowOffset = palaceRowOffset(side);
  for (let r = rowOffset; r < rowOffset + 3; r++) {
    for (let c = 3; c <= 5; c++) {
      PALACE_OF.set(squareOf(r, c), side);
    }
  }
}

export function isPalaceSquare(square: Square, side?: Side): boolean {
  const owner = PALACE_OF.get(square);
  if (owner === undefined) return false;
  return side === undefined || owner === side;
}

/**
 * Palace diagonal lines. Each palace has two 3-square diagonal lines through its center
 * (corner - center - opposite corner). Sliding pieces (chariot, cannon) may travel along these
 * lines only when starting from one of the five special squares (4 corners + center); guard/
 * general/soldier take at most the first step of a ray. Built once from the two corner-center-
 * corner lines per palace rather than hand-listing every square, to keep the one non-obvious
 * piece of geometry in this file verifiable by inspection instead of by transcription.
 */
export const PALACE_DIAGONAL_RAYS = new Map<Square, Square[][]>();

for (const side of ['han', 'cho'] as const) {
  const rowOffset = palaceRowOffset(side);
  const topLeft = squareOf(rowOffset, 3);
  const topRight = squareOf(rowOffset, 5);
  const bottomLeft = squareOf(rowOffset + 2, 3);
  const bottomRight = squareOf(rowOffset + 2, 5);
  const center = squareOf(rowOffset + 1, 4);
  const lines: Square[][] = [
    [topLeft, center, bottomRight],
    [topRight, center, bottomLeft],
  ];
  for (const line of lines) {
    for (let i = 0; i < line.length; i++) {
      const sq = line[i]!;
      const rays: Square[][] = [];
      if (i > 0) rays.push([...line.slice(0, i)].reverse());
      if (i < line.length - 1) rays.push(line.slice(i + 1));
      const existing = PALACE_DIAGONAL_RAYS.get(sq) ?? [];
      PALACE_DIAGONAL_RAYS.set(sq, [...existing, ...rays]);
    }
  }
}

export const ORTHOGONAL_DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

/** A ray of squares extending from `square` in direction (dr, dc) to the edge of the board. */
export function orthogonalRay(square: Square, dr: number, dc: number): Square[] {
  const ray: Square[] = [];
  let r = rowOf(square) + dr;
  let c = colOf(square) + dc;
  while (inBounds(r, c)) {
    ray.push(squareOf(r, c));
    r += dr;
    c += dc;
  }
  return ray;
}
