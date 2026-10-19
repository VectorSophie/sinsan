/**
 * Replays engine-games.jsonl (from dump_engine_games.py) through packages/rules and checks the
 * legal-move count and game result after every ply against the Rust engine, which is itself
 * verified against pyffish. Keeps the web app's rules identical to the training/teacher rules.
 *
 * Run: node tests/differential/check-ts-results.ts
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { applyMove, createInitialPosition, generateLegalMoves, getGameResult } from '../../packages/rules/src/index.ts';
import type { Formation, GameResultKind, Move } from '../../packages/rules/src/index.ts';

const KIND_TO_ENGINE: Record<GameResultKind, string> = {
  checkmate: 'checkmate',
  stalemate: 'stalemate',
  bikjang: 'bikjang',
  'double-pass': 'double-pass',
  'move-limit': 'move-limit',
  repetition: 'repetition',
  'perpetual-check': 'perpetual-check',
  resignation: 'resignation',
  timeout: 'timeout',
};

function parseSquare(text: string): number {
  const col = text.charCodeAt(0) - 'a'.charCodeAt(0);
  const row = 10 - Number(text.slice(1));
  return row * 9 + col;
}

function parseUci(uci: string): Move {
  const split = [...uci].findIndex((ch, i) => i > 0 && /[a-i]/.test(ch));
  const from = parseSquare(uci.slice(0, split));
  const to = parseSquare(uci.slice(split));
  return { from, to, isPass: from === to };
}

interface EngineGame {
  cho: Formation;
  han: Formation;
  moves: string[];
  plies: { legal: number; result: [string, string] | null }[];
}

const path = join(dirname(fileURLToPath(import.meta.url)), 'engine-games.jsonl');
const games: EngineGame[] = readFileSync(path, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
let positions = 0;
let failures = 0;
for (const [gi, game] of games.entries()) {
  let position = createInitialPosition({ setupCho: game.cho, setupHan: game.han, ruleProfile: 'kja' });
  for (const [ply, expected] of game.plies.entries()) {
    positions++;
    let legal: number;
    try {
      legal = generateLegalMoves(position).length;
    } catch (err) {
      console.log(`ERROR game ${gi} ply ${ply}: ${(err as Error).message}`);
      failures++;
      break;
    }
    const result = getGameResult(position);
    const ours = result ? [result.winner === 'cho' ? 'w' : 'b', KIND_TO_ENGINE[result.kind]] : null;
    if (legal !== expected.legal || JSON.stringify(ours) !== JSON.stringify(expected.result)) {
      failures++;
      if (failures <= 10) {
        console.log(`MISMATCH game ${gi} ply ${ply}: legal ${legal} vs ${expected.legal}, result ${JSON.stringify(ours)} vs ${JSON.stringify(expected.result)}`);
      }
      break;
    }
    const uci = game.moves[ply];
    if (uci !== undefined) position = applyMove(position, parseUci(uci));
  }
}
console.log(`${games.length} games, ${positions} positions checked, ${failures} mismatching games`);
process.exit(failures ? 1 : 0);
