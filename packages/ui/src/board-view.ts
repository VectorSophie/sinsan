import { Chessground } from 'chessgroundx';
import type { Config } from 'chessgroundx/config.js';
import type { DrawShape } from 'chessgroundx/draw.js';
import type { Dests, Key } from 'chessgroundx/types.js';
import { Notation } from 'chessgroundx/types.js';
import { generateLegalMoves, isCheck, positionKey } from '@sinsan/rules';
import type { Move, Position, Side, Square } from '@sinsan/rules';
import { squareToKey, keyToSquare } from './coords.ts';

export interface BoardViewOptions {
  /** The side shown at the bottom of the board. */
  readonly humanSide: Side;
  /** Which sides a person moves for (defaults to just humanSide). Empty for bot-vs-bot games. */
  readonly humanSides?: readonly Side[];
  readonly onHumanMove: (move: Move) => void;
  readonly animationDurationMs?: number;
}

/** An arrow drawn over the board (analysis display only - never affects the game). */
export interface BoardArrow {
  readonly from: Square;
  readonly to: Square;
  /** chessground brush name: green, red, blue, yellow, paleBlue, paleGreen, paleRed, paleGrey. */
  readonly brush: string;
  readonly lineWidth?: number;
  /** A small badge (e.g. a win chance) drawn on the destination square. */
  readonly label?: { readonly text: string; readonly color: string };
}

export interface BoardView {
  /** Shows `position`; with `readOnly` nobody can move (e.g. while inspecting an earlier position). */
  sync(position: Position, lastMove?: Move, animationDurationMs?: number, readOnly?: boolean): void;
  setArrows(arrows: readonly BoardArrow[]): void;
  /** Which side is drawn at the bottom. */
  setOrientation(side: Side): void;
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

/** Arrow brushes by rank for each side's options: shades of the side's colour, darkest for the most
 * searched move, so the ranking reads at a glance. */
export const RANK_BRUSHES = 5;
const RANK_SHADES: Record<'cho' | 'han', string[]> = {
  cho: ['#0b3a8c', '#1f6fe0', '#4f97f5', '#86bbf9', '#b9d8fc'],
  han: ['#8a130b', '#d4301f', '#f0604e', '#f69486', '#fac2ba'],
};
function rankBrushes(): Record<string, { key: string; color: string; opacity: number; lineWidth: number }> {
  const out: Record<string, { key: string; color: string; opacity: number; lineWidth: number }> = {};
  for (const side of ['cho', 'han'] as const) {
    RANK_SHADES[side].forEach((color, i) => {
      out[`${side}${i}`] = { key: `${side[0]}r${i}`, color, opacity: 0.92, lineWidth: 10 };
    });
  }
  return out;
}

/** Badge drawn in the destination square's 100x100 box; `slot` stacks badges that share a square. */
function badgeSvg(text: string, color: string, slot: number): string {
  const y = 64 - slot * 30;
  return `<rect x='14' y='${y}' width='72' height='28' rx='14' fill='${color}' stroke='#fff' stroke-width='2.5' opacity='0.95'/>` +
    `<text x='50' y='${y + 20.5}' font-family='system-ui, sans-serif' font-size='19' font-weight='700' fill='#fff' text-anchor='middle'>${text}</text>`;
}

export function createBoardView(container: HTMLElement, position: Position, options: BoardViewOptions): BoardView {
  container.classList.add('sinsan-board');

  const humanSides = options.humanSides ?? [options.humanSide];
  // Only the side to move can be dragged, and only when a person plays it.
  const movableColor = (p: Position): 'white' | 'black' | undefined =>
    humanSides.includes(p.sideToMove) ? colorOf(p.sideToMove) : undefined;

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
      color: movableColor(position),
      dests: legalDests(position),
      showDests: true,
      events: {
        after: (orig, dest) => {
          options.onHumanMove({ from: keyToSquare(orig as Key), to: keyToSquare(dest), isPass: false });
        },
      },
    },
    draggable: { enabled: true, showGhost: true },
    drawable: { brushes: rankBrushes() } as Config['drawable'],
    selectable: { enabled: true },
  };

  const api = Chessground(container, config);

  function sync(next: Position, lastMove?: Move, animationDurationMs?: number, readOnly = false): void {
    api.set({
      fen: boardFen(next),
      turnColor: colorOf(next.sideToMove),
      check: isCheck(next) ? colorOf(next.sideToMove) : false,
      lastMove: lastMove && !lastMove.isPass ? [squareToKey(lastMove.from), squareToKey(lastMove.to)] : undefined,
      animation: animationDurationMs !== undefined ? { enabled: true, duration: animationDurationMs } : undefined,
      movable: {
        color: readOnly ? undefined : movableColor(next),
        dests: readOnly ? new Map() : legalDests(next),
      },
    });
  }

  function setArrows(arrows: readonly BoardArrow[]): void {
    const badges = new Map<Square, number>();
    const shapes: DrawShape[] = arrows.map((a) => ({
      orig: squareToKey(a.from),
      dest: squareToKey(a.to),
      brush: a.brush,
      modifiers: a.lineWidth !== undefined ? { lineWidth: a.lineWidth } : undefined,
    }));
    // Badges after the arrows so they are drawn on top.
    for (const a of arrows) {
      if (!a.label) continue;
      const slot = badges.get(a.to) ?? 0;
      badges.set(a.to, slot + 1);
      shapes.push({ orig: squareToKey(a.to), customSvg: badgeSvg(a.label.text, a.label.color, slot) });
    }
    api.setAutoShapes(shapes);
  }

  return {
    sync,
    setArrows,
    setOrientation: (side) => api.set({ orientation: colorOf(side) }),
    destroy: () => api.destroy(),
  };
}
