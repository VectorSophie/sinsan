"""Dumps randomized edge-case games played by sinsan_engine (itself verified against pyffish by
compare_engine.py) with the engine's legal-move count and game result after every ply, for
check-ts-results.ts to replay through packages/rules.

Run: cd training && uv run python ../tests/differential/dump_engine_games.py --games 1000
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sinsan_engine as se  # noqa: E402
from compare_engine import FORMATIONS, MAX_PLIES, choose_move, random_style  # noqa: E402

OUT = Path(__file__).parent / "engine-games.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()
    with OUT.open("w") as out:
        for seed in range(args.games):
            rng = random.Random(10_000 + seed)
            cho, han = rng.choice(FORMATIONS), rng.choice(FORMATIONS)
            game = se.Game(cho=cho, han=han)
            styles = {"w": random_style(rng), "b": random_style(rng)}
            plies = []
            for _ in range(MAX_PLIES):
                legal = game.legal_moves()
                result = game.result()
                plies.append({"legal": len(legal), "result": result})
                if result is not None:
                    break
                game.push(choose_move(game, legal, styles[game.side_to_move], rng))
            out.write(json.dumps({"cho": cho, "han": han, "moves": game.moves(), "plies": plies}) + "\n")
    print(f"wrote {args.games} games to {OUT}")


if __name__ == "__main__":
    main()
