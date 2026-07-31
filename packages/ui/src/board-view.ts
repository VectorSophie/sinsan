import { Chessground } from 'chessgroundx';
import type { Config } from 'chessgroundx/config.js';
import type { Dests, Key } from 'chessgroundx/types.js';
import { Notation } from 'chessgroundx/types.js';
import { generateLegalMoves, isCheck, positionKey } from '@sinsan/rules';
import type { Move, Position, Side } from '@sinsan/rules';
import { squareToKey, keyToSquare } from './coords.ts';

export interface BoardViewOptions {
  readonly humanSide: Side;
  readonly onHumanMove: (move: Move) => void;
  readonly animationDurationMs?: number;
}

export interface BoardView {
  sync(position: Position, lastMove?: Move, animationDurationMs?: number): void;
  destroy(): void;
}

/** positionKey() returns "board fen" + " " + side-to-move; chessgroundx's FEN reader uses the
 * same row order, piece letters, and uppercase-is-first-mover convention as our own serialization
 * (both confirmed by reading chessgroundx's fen.ts/util.ts during Phase 2 research), so the board
 * part can be handed to it directly with no manual piece-map conversion. */
function boardFen(position: Position): string {
  return positionKey(position).split(' ')[0]!;
}

function legalDests(position: Position): Dests {
  const dests: Dests = new Map();
  for (const move of generateLegalMoves(position)) {
    if (move.isPass) continue;
    const from = squareToKey(move.from);
    const list = dests.get(from) ?? [];
    list.push(squareToKey(move.to));
    dests.set(from, list);
  }
  return dests;
}

function colorOf(side: Side): 'white' | 'black' {
  return side === 'cho' ? 'white' : 'black';
}

export function createBoardView(container: HTMLElement, position: Position, options: BoardViewOptions): BoardView {
  container.classList.add('sinsan-board');

  const config: Config = {
    fen: boardFen(position),
    orientation: colorOf(options.humanSide),
    turnColor: colorOf(position.sideToMove),
    dimensions: { width: 9, height: 10 },
    notation: Notation.JANGGI,
    // chessgroundx's bundled coordinate-label CSS positions ranks/files for an 8x8 board only;
    // rendering them correctly for 9x10 needs additional CSS this vertical slice skips for now.
    coordinates: false,
    animation: { enabled: true, duration: options.animationDurationMs ?? 180 },
    highlight: { lastMove: true, check: true },
    movable: {
      free: false,
      color: colorOf(options.humanSide),
      dests: legalDests(position),
      showDests: true,
      events: {
        after: (orig, dest) => {
          options.onHumanMove({ from: keyToSquare(orig as Key), to: keyToSquare(dest), isPass: false });
        },
      },
    },
    draggable: { enabled: true, showGhost: true },
    selectable: { enabled: true },
  };

  const api = Chessground(container, config);

  function sync(next: Position, lastMove?: Move, animationDurationMs?: number): void {
    api.set({
      fen: boardFen(next),
      turnColor: colorOf(next.sideToMove),
      check: isCheck(next) ? colorOf(next.sideToMove) : false,
      lastMove: lastMove && !lastMove.isPass ? [squareToKey(lastMove.from), squareToKey(lastMove.to)] : undefined,
      animation: animationDurationMs !== undefined ? { enabled: true, duration: animationDurationMs } : undefined,
      movable: {
        color: colorOf(options.humanSide),
        dests: legalDests(next),
      },
    });
  }

  return {
    sync,
    destroy: () => api.destroy(),
  };
}
