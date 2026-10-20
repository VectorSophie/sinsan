"""Rating method 2: move quality on human positions (Regan-style intrinsic rating, simplified).

Data: the PyChess Janggi game archive published by the PyChess maintainer
(github.com/gbtami/pychess-variants-games, release "pychess-janggi-games"; credit: PyChess,
pychess.org). Only players with an established rating at game time are used, at most --cap games
each, so no single player dominates.

1. Every position before one of those players' moves is analysed by Fairy-Stockfish NNUE at
   --nodes: the best move's score, and the score after the move actually played. The move's cost is
   the win chance it gave away, with P(win) = sigmoid(cp / 105) (the scale fitted on teacher games).
2. Sinsan plays the same positions (a network at --visits) and its moves are scored the same way.
3. Per player, mean win-chance loss is regressed on rating (rating ~ a + b ln loss); Sinsan's mean
   loss is read off the fit. A bootstrap over players gives the interval.

  uv run python -m eval.human_rating --net runs/student-tf192x6/ema-0030000.pt --visits 400
Results go to eval/results/human-rating-*.json; the per-position analysis is cached.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sinsan_engine as se
import torch

from eval.net_arena import NetEvaluator
from eval.uci import fairy_stockfish

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "datasets" / "pychess-games" / "pychess-janggi-games.json"
CACHE = ROOT / "datasets" / "pychess-games" / "analysis"
BOTS = {"Fairy-Stockfish", "Random-Mover"}
CP_SCALE = 105.0

# PyChess stores each move as two characters, one per square; ranks are 0-based (compress.py).
_M2C = dict(zip([a + b for a, b in itertools.product("abcdefghij", "0123456789")], range(34, 256)))
_C2M = {v: k for k, v in _M2C.items()}


def decode(m: str) -> str:
    sq = lambda c: (lambda s: s[0] + str(int(s[1:]) + 1))(_C2M[ord(c)])  # noqa: E731
    return sq(m[0]) + sq(m[1])


def win(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / CP_SCALE))


def positions(cap: int, skip_plies: int) -> list[dict]:
    """Positions before moves by players with an established rating."""
    per_player: dict[str, int] = {}
    out = []
    with open(ARCHIVE) as f:
        for line in f:
            d = json.loads(line)
            if len(d["m"]) < 20 or d["s"] == 0 or not d.get("if"):
                continue
            sides = []
            for side in (0, 1):
                user, elo = d["us"][side], str(d.get(f"p{side}", {}).get("e", ""))
                if user in BOTS or user.startswith("Anon") or not elo.isdigit() or per_player.get(user, 0) >= cap:
                    continue
                per_player[user] = per_player.get(user, 0) + 1
                sides.append((side, user, int(elo)))
            if not sides:
                continue
            try:
                moves = [decode(m) for m in d["m"]]
            except KeyError:
                continue
            for side, user, elo in sides:
                for ply in range(skip_plies + (skip_plies + side) % 2, len(moves), 2):
                    out.append({"game": d["_id"], "fen": d["if"], "moves": moves[:ply], "played": moves[ply],
                                "player": user, "rating": elo, "vs_ai": bool(BOTS & set(d["us"]))})
    return out


class Fsf:
    """Thread-local Fairy-Stockfish engines scoring (fen, moves) for the side to move."""

    def __init__(self, nodes: int) -> None:
        self.nodes, self.local = nodes, threading.local()

    def score(self, fen: str, moves: list[str]) -> tuple[float, str]:
        eng = getattr(self.local, "e", None)
        if eng is None:
            eng = self.local.e = fairy_stockfish(nnue=True, hash_mb=64)
        info = eng.search(fen, moves, f"nodes {self.nodes}")
        if info.score_mate is not None:
            cp = 10000.0 if info.score_mate > 0 else -10000.0
        else:
            cp = float(info.score_cp or 0)
        return cp, info.bestmove


def move_loss(fsf: Fsf, fen: str, moves: list[str], move: str, before: tuple[float, str] | None = None) -> tuple[float, float, str]:
    """(win chance before, win chance lost by `move`, engine best) for the side to move."""
    cp_before, best = before or fsf.score(fen, moves)
    if move == best:
        return win(cp_before), 0.0, best
    game = se.Game(fen)
    for m in moves + [move]:
        game.push(m)
    if game.result() is not None:  # the move ended the game: it won, or it lost
        res = game.result()[0]
        mover = "w" if len(moves) % 2 == 0 else "b"
        after = 1.0 if res == mover else 0.0
    else:
        cp_after, _ = fsf.score(fen, moves + [move])
        after = 1.0 - win(cp_after)
    return win(cp_before), max(0.0, win(cp_before) - after), best


def sinsan_moves(pos: list[dict], net: str, visits: int, device: str) -> list[str]:
    """Sinsan's move in each position, searched in large batches."""
    ev = NetEvaluator(net, torch.device(device))
    out: list[str] = [""] * len(pos)
    batch_size = 256
    for start in range(0, len(pos), batch_size):
        chunk = pos[start:start + batch_size]
        b = se.SearchBatch()
        for i, p in enumerate(chunk):
            b.set_root_fen(i, p["fen"], p["moves"])
        active = set(range(len(chunk)))
        while active:
            d = b.collect(8, sorted(active))
            if d["count"]:
                b.apply(*ev(d))
            for i in list(active):
                if b.root_expanded(i) and b.root_visits(i) >= visits:
                    out[start + i] = b.best_move(i)
                    active.discard(i)
                elif not d["count"] and b.root_expanded(i):
                    out[start + i] = b.best_move(i) or ""
                    active.discard(i)
        print(f"  sinsan: {min(start + batch_size, len(pos))}/{len(pos)}", flush=True)
    return out


def fit(players: dict[str, tuple[float, float, int]]) -> tuple[float, float]:
    """rating = a + b ln(mean loss), weighted by moves."""
    x = np.array([math.log(max(v[0], 1e-4)) for v in players.values()])
    y = np.array([v[1] for v in players.values()])
    w = np.array([v[2] for v in players.values()], float)
    b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
    return a, b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True)
    p.add_argument("--visits", type=int, default=400)
    p.add_argument("--nodes", type=int, default=100_000, help="Fairy-Stockfish nodes per analysed position")
    p.add_argument("--cap", type=int, default=40, help="games per player")
    p.add_argument("--skip-plies", type=int, default=4, help="ignore the first plies (formation openings)")
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--name", default="student400")
    args = p.parse_args()

    pos = positions(args.cap, args.skip_plies)
    n_players = len({q["player"] for q in pos})
    print(f"{len(pos)} positions from {n_players} established players", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    human_file = CACHE / f"human-n{args.nodes}-cap{args.cap}.jsonl"
    fsf = Fsf(args.nodes)
    if human_file.exists():
        human = [json.loads(line) for line in human_file.read_text().splitlines()]
    else:
        def analyse(q: dict) -> dict:
            w0, loss, best = move_loss(fsf, q["fen"], q["moves"], q["played"])
            return {**q, "win_before": w0, "loss": loss, "best": best}
        with ThreadPoolExecutor(args.threads) as ex:
            human = list(ex.map(analyse, pos))
        human_file.write_text("".join(json.dumps(h) + "\n" for h in human))
    print("human moves analysed", flush=True)

    # Sinsan on the same positions; its move scored with the cached "before" analysis.
    mine = sinsan_moves(human, args.net, args.visits, args.device)

    def analyse_mine(i: int) -> float:
        h = human[i]
        cp_before = CP_SCALE * math.log(h["win_before"] / max(1e-9, 1 - h["win_before"])) if 0 < h["win_before"] < 1 else (10000.0 if h["win_before"] >= 1 else -10000.0)
        return move_loss(fsf, h["fen"], h["moves"], mine[i] or h["best"], (cp_before, h["best"]))[1]

    with ThreadPoolExecutor(args.threads) as ex:
        sinsan_loss = list(ex.map(analyse_mine, range(len(human))))

    # Per player: mean loss, rating (mean of game-time ratings), moves.
    per: dict[str, list] = {}
    for h in human:
        per.setdefault(h["player"], []).append(h)
    players = {u: (float(np.mean([h["loss"] for h in hs])), float(np.mean([h["rating"] for h in hs])), len(hs))
               for u, hs in per.items() if len(hs) >= 60}
    a, b = fit(players)
    s_loss = float(np.mean(sinsan_loss))
    est = a + b * math.log(s_loss)
    rng = random.Random(0)
    names = list(players)
    boots = []
    for _ in range(2000):
        sample = {f"{k}#{i}": players[k] for i, k in enumerate(rng.choice(names) for _ in names)}
        try:
            ba, bb = fit(sample)
            boots.append(ba + bb * math.log(s_loss))
        except (np.linalg.LinAlgError, ValueError):
            pass
    lo, hi = np.percentile(boots, [5, 95]) if boots else (float("nan"), float("nan"))
    match = float(np.mean([m == h["best"] for m, h in zip(mine, human)]))
    report = {
        "method": "move quality on human positions", "net": args.net, "visits": args.visits, "fsf_nodes": args.nodes,
        "positions": len(human), "players": {u: {"mean_loss": v[0], "rating": v[1], "moves": v[2]} for u, v in players.items()},
        "fit": {"a": a, "b": b}, "sinsan_mean_loss": s_loss, "sinsan_top_match": match,
        "human_mean_loss": float(np.mean([h["loss"] for h in human])),
        "estimate": est, "interval90": [float(lo), float(hi)],
        "source": "PyChess Janggi game archive (github.com/gbtami/pychess-variants-games), pychess.org",
    }
    out = ROOT / "eval" / "results" / f"human-rating-{args.name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for u, v in sorted(players.items(), key=lambda t: t[1][1]):
        print(f"  {v[1]:6.0f}  loss {100 * v[0]:5.2f}%  ({v[2]} moves)")
    print(f"Sinsan ({args.name}): mean loss {100 * s_loss:.2f}%, matches FSF's move {100 * match:.0f}%")
    print(f"=> pychess-scale estimate {est:.0f} (90% {lo:.0f} ~ {hi:.0f}); fit rating = {a:.0f} + {b:.0f} ln(loss)")


if __name__ == "__main__":
    main()
