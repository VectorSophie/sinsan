"""Builds a balanced opening suite for matches.

For each of the 16 (Cho, Han) formation pairs, plays 4-8 plies by sampling among Fairy-Stockfish
NNUE MultiPV candidates within a small margin of the best move, then keeps the position only if a
deeper search rates it roughly level. Every opening is played twice in a match, colors swapped.

Run: cd training && uv run python -m eval.openings --per-pair 32 --workers 64
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from multiprocessing import Pool
from pathlib import Path

import sinsan_engine as se

from eval.uci import fairy_stockfish

FORMATIONS = ["masang-sangma", "sangma-masang", "masang-masang", "sangma-sangma"]
OUT = Path(__file__).parent / "openings.jsonl"

_engine = None


def _init() -> None:
    global _engine
    _engine = fairy_stockfish(nnue=True, multipv=4, hash_mb=32)


def make_opening(task: tuple[str, str, int]) -> dict | None:
    cho, han, seed = task
    rng = random.Random(seed)
    game = se.Game(cho=cho, han=han)
    root = game.root_fen()
    _engine.send("setoption name MultiPV value 4")
    for _ in range(rng.randint(4, 8)):
        info = _engine.search(root, game.moves(), "nodes 60000")
        best = info.multipv[0][1]
        if best is None:
            return None
        candidates = [m for m, cp, mate in info.multipv if cp is not None and best - cp <= 40]
        move = rng.choice(candidates)
        if move not in game.legal_moves():
            return None
        game.push(move)
        if game.result() is not None:
            return None
    _engine.send("setoption name MultiPV value 1")
    final = _engine.search(root, game.moves(), "nodes 400000")
    if final.score_cp is None or abs(final.score_cp) > 80:
        return None
    return {"cho": cho, "han": han, "moves": game.moves(), "fen": game.fen(), "eval_cp": final.score_cp}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-pair", type=int, default=32)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    pairs = list(itertools.product(FORMATIONS, FORMATIONS))
    tasks = [(c, h, seed) for c, h in pairs for seed in range(args.per_pair * 3)]  # oversample, then trim
    kept: dict[tuple[str, str], list[dict]] = {p: [] for p in pairs}
    seen = set()
    with Pool(args.workers, initializer=_init) as pool:
        for op in pool.imap_unordered(make_opening, tasks, chunksize=2):
            if op is None or op["fen"] in seen:
                continue
            key = (op["cho"], op["han"])
            if len(kept[key]) < args.per_pair:
                seen.add(op["fen"])
                kept[key].append(op)
    openings = [op for p in pairs for op in kept[p]]
    with OUT.open("w") as f:
        for i, op in enumerate(openings):
            f.write(json.dumps({"id": i, **op}) + "\n")
    short = {f"{c}/{h}": len(v) for (c, h), v in kept.items() if len(v) < args.per_pair}
    print(f"wrote {len(openings)} openings to {OUT}" + (f"; short pairs: {short}" if short else ""))


if __name__ == "__main__":
    main()
