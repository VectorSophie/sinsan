"""Sinsan as a UCI engine (Janggi, Fairy-Stockfish conventions: UCI_Variant janggi, pass = same square twice).

Usable from any UCI-capable Janggi GUI, or from a bot client (e.g. for pychess.org). Searches with
the batched PUCT engine (one tree, several leaves per GPU batch), reusing the tree between moves.

Options: Net (checkpoint path), Device (cuda:0), Visits (0 = time-managed, else fixed visits per
move), Leaves (leaves per batch), Threads/Hash are accepted and ignored for GUI compatibility.

Run: cd training && uv run python uci_engine.py --net runs/<run>/ema-XXXXXXX.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch

import sinsan_engine as se
from eval.net_arena import NetEvaluator

STARTPOS = "rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1"
CP_SCALE = 105.0  # fitted on teacher games: P(win) = sigmoid(cp / 105)
CPUCT = 1.2  # play-time exploration (docs/REBUILD_LOG.md); self-play keeps the engine default


def value_to_cp(v: float) -> int:
    p = min(max((v + 1) / 2, 1e-4), 1 - 1e-4)
    return int(round(CP_SCALE * math.log(p / (1 - p))))


class Engine:
    def __init__(self, net: str, device: str, visits: int, leaves: int) -> None:
        self.net_path, self.device, self.visits, self.leaves = net, device, visits, leaves
        self.evaluate: NetEvaluator | None = None
        self.batch = se.SearchBatch(cpuct=CPUCT)
        self.root: tuple[str, list[str]] | None = None

    def ensure_loaded(self) -> None:
        if self.evaluate is None:
            self.evaluate = NetEvaluator(self.net_path, torch.device(self.device))

    def set_position(self, fen: str, moves: list[str]) -> None:
        # Reuse the tree when the new position extends the previous one by a move or two.
        if self.root and self.root[0] == fen and moves[: len(self.root[1])] == self.root[1] and len(moves) - len(self.root[1]) <= 2 \
                and self.batch.has_slot(0):
            for m in moves[len(self.root[1]):]:
                self.batch.advance(0, m)
        else:
            self.batch.set_root_fen(0, fen, moves)
        self.root = (fen, list(moves))

    def search(self, budget_s: float | None, visits: int | None) -> str:
        self.ensure_loaded()
        start = time.time()
        last_info = start
        while True:
            d = self.batch.collect(self.leaves, [0])
            if d["count"]:
                self.batch.apply(*self.evaluate(d))
            n = self.batch.root_visits(0)
            elapsed = time.time() - start
            if time.time() - last_info > 1.0:
                self.info(n, elapsed)
                last_info = time.time()
            if not self.batch.root_expanded(0):
                if not d["count"]:
                    break  # terminal root
                continue
            if visits is not None and n >= visits:
                break
            if budget_s is not None and elapsed >= budget_s:
                break
            # Stop early when the best move cannot change within the remaining budget.
            if budget_s is not None and visits is None and elapsed > 0.1:
                children = sorted(self.batch.root_children(0), key=lambda c: -c[1])
                if len(children) > 1:
                    rate = n / elapsed
                    remaining = rate * (budget_s - elapsed)
                    if children[0][1] - children[1][1] > remaining:
                        break
        self.info(self.batch.root_visits(0), time.time() - start)
        best = self.batch.best_move(0)
        if best is None:
            legal = se.Game(self.root[0]).legal_moves() if self.root else []
            best = legal[0] if legal else "0000"
        return best

    def info(self, nodes: int, elapsed: float) -> None:
        pv = self.batch.principal_variation(0, 12)
        cp = value_to_cp(self.batch.root_value(0))
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        print(f"info depth {len(pv)} score cp {cp} nodes {nodes} nps {nps} time {int(elapsed * 1000)} pv {' '.join(pv)}", flush=True)


def time_budget(tokens: list[str], white_to_move: bool) -> tuple[float | None, int | None]:
    args = {tokens[i]: tokens[i + 1] for i in range(len(tokens) - 1) if tokens[i] in
            ("wtime", "btime", "winc", "binc", "movetime", "nodes", "movestogo")}
    if "nodes" in args:
        return None, int(args["nodes"])
    if "movetime" in args:
        return max(0.01, int(args["movetime"]) / 1000 - 0.05), None
    key, inc = ("wtime", "winc") if white_to_move else ("btime", "binc")
    if key in args:
        remaining = int(args[key]) / 1000
        increment = int(args.get(inc, 0)) / 1000
        moves_to_go = int(args.get("movestogo", 30))
        return max(0.05, min(remaining * 0.5, remaining / moves_to_go + 0.8 * increment) - 0.05), None
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visits", type=int, default=0)
    parser.add_argument("--leaves", type=int, default=16)
    args = parser.parse_args()
    engine = Engine(args.net, args.device, args.visits, args.leaves)
    fen, moves = STARTPOS, []
    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            print("id name Sinsan\nid author Sinsan project", flush=True)
            print("option name UCI_Variant type combo default janggi var janggi", flush=True)
            print(f"option name Visits type spin default {args.visits} min 0 max 1000000", flush=True)
            print("option name Threads type spin default 1 min 1 max 512", flush=True)
            print("option name Hash type spin default 16 min 1 max 65536", flush=True)
            print("uciok", flush=True)
        elif cmd == "setoption" and "name" in tokens and "value" in tokens:
            name = " ".join(tokens[tokens.index("name") + 1: tokens.index("value")])
            value = " ".join(tokens[tokens.index("value") + 1:])
            if name == "Visits":
                engine.visits = int(value)
        elif cmd == "isready":
            engine.ensure_loaded()
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            engine.root = None
        elif cmd == "position":
            if tokens[1] == "startpos":
                fen, rest = STARTPOS, tokens[2:]
            else:
                end = tokens.index("moves") if "moves" in tokens else len(tokens)
                fen, rest = " ".join(tokens[2:end]), tokens[end:]
            moves = rest[1:] if rest and rest[0] == "moves" else []
            engine.set_position(fen, moves)
        elif cmd == "go":
            if engine.root is None:
                engine.set_position(fen, moves)
            budget, visits = time_budget(tokens[1:], se.Game(fen).side_to_move == "w" if not moves else
                                         (len(moves) % 2 == 0) == (fen.split()[1] == "w"))
            if visits is None and budget is None:
                visits = engine.visits or 800
            elif engine.visits and budget is None:
                visits = engine.visits
            print(f"bestmove {engine.search(budget, visits)}", flush=True)
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
