import { BOARD_COLS, BOARD_ROWS, BOARD_SIZE, squareOf, palaceRowOffset } from './board.ts';
import { backRankLayout } from './formations.ts';
import type { CreatePositionOptions, Piece, PieceType, Position, Side, Square } from './types.ts';
import { DEFAULT_RULE_PROFILE_ID } from './rule-profiles.ts';

/** Xiangqi/Janggi-style FEN letters, matching Fairy-Stockfish's own Janggi FEN convention
 * (confirmed during Phase 0 research) so board state can be cross-referenced during differential
 * testing. Lowercase = Han, uppercase = Cho. */
const PIECE_LETTERS: Record<PieceType, string> = {
  chariot: 'r',
  horse: 'n',
  elephant: 'b',
  guard: 'a',
  general: 'k',
  cannon: 'c',
  soldier: 'p',
};

const LETTER_TO_PIECE_TYPE = new Map<string, PieceType>(
  Object.entries(PIECE_LETTERS).map(([type, letter]) => [letter, type as PieceType]),
);

function pieceLetter(piece: Piece): string {
  const letter = PIECE_LETTERS[piece.type];
  return piece.side === 'cho' ? letter.toUpperCase() : letter;
}

export function createInitialPosition(options: CreatePositionOptions = {}): Position {
  const ruleProfile = options.ruleProfile ?? DEFAULT_RULE_PROFILE_ID;
  const setupCho = options.setupCho ?? 'masang-sangma';
  const setupHan = options.setupHan ?? 'masang-sangma';

  const board: (Piece | null)[] = new Array(BOARD_SIZE).fill(null);

  function place(side: Side, row: number, col: number, type: PieceType): void {
    board[squareOf(row, col)] = { type, side };
  }

  function placeBackRank(side: Side, formationName: Parameters<typeof backRankLayout>[0], row: number): void {
    const { left, right } = backRankLayout(formationName);
    place(side, row, 0, 'chariot');
    place(side, row, 1, left[0]);
    place(side, row, 2, left[1]);
    place(side, row, 3, 'guard');
    place(side, row, 5, 'guard');
    place(side, row, 6, right[0]);
    place(side, row, 7, right[1]);
    place(side, row, 8, 'chariot');
  }

  // Han: back rank row 0, general/palace-center row 1, cannons row 2, soldiers row 3.
  placeBackRank('han', setupHan, 0);
  place('han', 1, 4, 'general');
  place('han', 2, 1, 'cannon');
  place('han', 2, 7, 'cannon');
  for (const col of [0, 2, 4, 6, 8]) place('han', 3, col, 'soldier');

  // Cho: back rank row 9, general/palace-center row 8, cannons row 7, soldiers row 6.
  placeBackRank('cho', setupCho, 9);
  place('cho', 8, 4, 'general');
  place('cho', 7, 1, 'cannon');
  place('cho', 7, 7, 'cannon');
  for (const col of [0, 2, 4, 6, 8]) place('cho', 6, col, 'soldier');

  const position: Position = {
    board,
    sideToMove: 'cho',
    ruleProfile,
    setupCho,
    setupHan,
    noCapturePly: 0,
    moveNumber: 1,
    positionHistory: [],
  };
  return { ...position, positionHistory: [positionKey(position)] };
}

/** Board + side-to-move only - the portion relevant to repetition detection. Deliberately
 * excludes noCapturePly/moveNumber/ruleProfile/setup, which don't affect whether a position has
 * recurred. */
export function positionKey(position: Position): string {
  const rows: string[] = [];
  for (let r = 0; r < BOARD_ROWS; r++) {
    let rowStr = '';
    let emptyRun = 0;
    for (let c = 0; c < BOARD_COLS; c++) {
      const piece = position.board[squareOf(r, c)];
      if (piece) {
        if (emptyRun > 0) {
          rowStr += String(emptyRun);
          emptyRun = 0;
        }
        rowStr += pieceLetter(piece);
      } else {
        emptyRun++;
      }
    }
    if (emptyRun > 0) rowStr += String(emptyRun);
    rows.push(rowStr);
  }
  const side = position.sideToMove === 'cho' ? 'w' : 'b';
  return `${rows.join('/')} ${side}`;
}

export function serializePosition(position: Position): string {
  return [
    positionKey(position),
    position.ruleProfile,
    position.setupCho,
    position.setupHan,
    position.noCapturePly,
    position.moveNumber,
  ].join(' ');
}

export function parsePosition(serialized: string): Position {
  const parts = serialized.split(' ');
  const boardPart = parts[0];
  const sidePart = parts[1];
  const ruleProfile = parts[2] as Position['ruleProfile'];
  const setupCho = parts[3] as Position['setupCho'];
  const setupHan = parts[4] as Position['setupHan'];
  const noCapturePly = Number(parts[5]);
  const moveNumber = Number(parts[6]);
  if (boardPart === undefined || sidePart === undefined) {
    throw new Error(`parsePosition: malformed serialized position: ${serialized}`);
  }

  const board: (Piece | null)[] = new Array(BOARD_SIZE).fill(null);
  const rows = boardPart.split('/');
  for (let r = 0; r < rows.length; r++) {
    let c = 0;
    for (const ch of rows[r]!) {
      if (/[0-9]/.test(ch)) {
        c += Number(ch);
        continue;
      }
      const type = LETTER_TO_PIECE_TYPE.get(ch.toLowerCase());
      if (!type) throw new Error(`parsePosition: unknown piece letter '${ch}'`);
      const side: Side = ch === ch.toUpperCase() ? 'cho' : 'han';
      board[squareOf(r, c)] = { type, side };
      c++;
    }
  }

  const position: Position = {
    board,
    sideToMove: sidePart === 'w' ? 'cho' : 'han',
    ruleProfile,
    setupCho,
    setupHan,
    noCapturePly,
    moveNumber,
    positionHistory: [],
  };
  return { ...position, positionHistory: [positionKey(position)] };
}

export function pieceAt(position: Position, square: Square): Piece | null {
  return position.board[square] ?? null;
}

export function findGeneral(position: Position, side: Side): Square {
  for (let sq = 0; sq < BOARD_SIZE; sq++) {
    const piece = position.board[sq];
    if (piece && piece.side === side && piece.type === 'general') return sq;
  }
  throw new Error(`findGeneral: no general found for ${side}`);
}

export { palaceRowOffset };
