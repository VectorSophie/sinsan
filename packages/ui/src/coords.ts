import { colOf, rowOf, squareOf } from '@sinsan/rules';
import type { Square } from '@sinsan/rules';
import { files, ranks } from 'chessgroundx/types.js';
import type { Key } from 'chessgroundx/types.js';

/**
 * chessgroundx ranks run bottom-to-top as '1'..'9',':'  (':' is the 10th rank - its Key type
 * uses one ASCII character per rank, so ranks beyond '9' continue from ':' onward). Our own row
 * 9 (Cho's back rank) is the board's bottom row, so it maps to chessground rank '1'; our row 0
 * (Han's back rank) maps to the 10th character, ':'.
 */
export function squareToKey(square: Square): Key {
  const file = files[colOf(square)];
  const rank = ranks[9 - rowOf(square)];
  if (file === undefined || rank === undefined) throw new Error(`squareToKey: square out of range: ${square}`);
  return `${file}${rank}` as Key;
}

export function keyToSquare(key: Key): Square {
  const file = key[0]!;
  const rank = key.slice(1);
  const col = files.indexOf(file as (typeof files)[number]);
  const rankIndex = ranks.indexOf(rank as (typeof ranks)[number]);
  if (col === -1 || rankIndex === -1) throw new Error(`keyToSquare: invalid key: ${key}`);
  return squareOf(9 - rankIndex, col);
}
