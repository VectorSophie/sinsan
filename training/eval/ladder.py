"""Rates a set of players on one Elo scale from paired matches between neighbours.

Default ladder: Fairy-Stockfish NNUE at doubling node budgets (the main scale), classical
evaluation at a few budgets, random play, and any extra players given with --add (e.g. a Sinsan
checkpoint), each matched against its nearest ladder rungs. Ratings come from a joint
Bradley-Terry fit with Fairy-Stockfish NNUE at 1,000 nodes anchored at 0.

Run: cd training && uv run python -m eval.ladder --pairs 200 [--add "oldnet:baseline-v3-56x7:56x7:16=fsf:nnue:64,fsf:nnue:128"]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.match import run_match
from eval.rating import bradley_terry, pentanomial_elo

RESULTS = Path(__file__).parent / "results"
NNUE_NODES = [64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
ANCHOR = "fsf:nnue:1000"


def default_matchups() -> list[tuple[str, str]]:
    nnue = [f"fsf:nnue:{n}" for n in NNUE_NODES]
    pairs = list(zip(nnue[1:], nnue[:-1]))  # each doubling
    pairs += list(zip(nnue[2:], nnue[:-2]))  # each 4x, for connectivity
    pairs += [("fsf:classical:1000", "fsf:nnue:256"), ("fsf:classical:1000", "fsf:nnue:512"),
              ("fsf:classical:16000", "fsf:nnue:2000"), ("fsf:classical:16000", "fsf:nnue:4000"),
              ("fsf:classical:128000", "fsf:nnue:8000"), ("fsf:classical:128000", "fsf:nnue:16000"),
              ("fsf:nnue:64", "random")]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--add", action="append", default=[], help="SPEC=OPP1,OPP2: extra player and its opponents")
    parser.add_argument("--name", default="ladder")
    parser.add_argument("--skip-default", action="store_true", help="only play --add matchups, reusing saved results")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    games_file = RESULTS / f"{args.name}-games.jsonl"
    matchups = [] if args.skip_default else default_matchups()
    for extra in args.add:
        spec, opponents = extra.split("=")
        matchups += [(spec, opp) for opp in opponents.split(",")]

    for a, b in matchups:
        t = time.time()
        records = run_match(a, b, args.pairs, args.workers, out=games_file, quiet=True)
        elo, lo, hi = pentanomial_elo([r["a_score"] for r in records])
        print(f"{a:>28} vs {b:<22} {elo:+6.0f} [{lo:+.0f}, {hi:+.0f}]  ({time.time() - t:.0f}s)", flush=True)

    # Fit on every game recorded so far for this ladder name.
    results = []
    for line in games_file.read_text().splitlines():
        rec = json.loads(line)
        for game, a_side in zip(rec["games"], ("w", "b")):
            score = 1.0 if game["winner"] == a_side else (0.5 if game["winner"] is None else 0.0)
            results.append((rec["a"], rec["b"], score))
    ratings = bradley_terry(results, anchor=ANCHOR)
    print(f"\nRatings ({len(results)} games, {ANCHOR} = 0):")
    for player, (elo, err) in sorted(ratings.items(), key=lambda kv: kv[1][0]):
        print(f"  {player:>32}  {elo:+7.0f} ± {1.96 * err:.0f}")
    (RESULTS / f"{args.name}-ratings.json").write_text(json.dumps(ratings, indent=1))


if __name__ == "__main__":
    main()
