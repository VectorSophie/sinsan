"""Parallel paired match between two players on the opening suite.

Each opening is played twice with colors swapped. The referee is sinsan_engine (Fairy-Stockfish
janggi rules), so every game ends by checkmate or a points decision; there are no draws under
janggi rules, but the scoring code still handles them.

Run: cd training && uv run python -m eval.match --a fsf:nnue:4000 --b fsf:nnue:1000 --pairs 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import sinsan_engine as se

from eval.players import make_player
from eval.rating import pentanomial_elo, sprt_bounds, sprt_llr

OPENINGS = Path(__file__).parent / "openings.jsonl"
MAX_PLIES = 600

_players: dict[str, object] = {}


def _player(spec: str):
    if spec not in _players:
        _players[spec] = make_player(spec, seed=os.getpid())
    return _players[spec]


def play_game(opening: dict, cho_spec: str, han_spec: str) -> dict:
    game = se.Game(cho=opening["cho"], han=opening["han"])
    for m in opening["moves"]:
        game.push(m)
    players = {"w": _player(cho_spec), "b": _player(han_spec)}
    for p in players.values():
        p.new_game()
    result = game.result()
    while result is None and game.plies < MAX_PLIES:
        game.push(players[game.side_to_move].choose(game))
        result = game.result()
    winner, reason = result if result else (None, "max-plies")
    return {"winner": winner, "reason": reason, "plies": game.plies, "moves": game.moves()}


def play_pair(task: tuple[int, dict, str, str]) -> dict:
    pair_id, opening, a, b = task
    g1 = play_game(opening, a, b)  # A plays Cho
    g2 = play_game(opening, b, a)  # A plays Han
    score = lambda g, a_side: 1.0 if g["winner"] == a_side else (0.5 if g["winner"] is None else 0.0)  # noqa: E731
    return {"pair": pair_id, "opening": opening["id"], "games": [g1, g2], "a_score": score(g1, "w") + score(g2, "b")}


def run_match(a: str, b: str, pairs: int, workers: int, out: Path | None = None, sprt: tuple[float, float] | None = None,
              quiet: bool = False) -> list[dict]:
    """Plays the match and returns the per-pair records (see play_pair)."""
    openings = [json.loads(l) for l in OPENINGS.read_text().splitlines() if l.strip()]
    tasks = [(i, openings[i % len(openings)], a, b) for i in range(pairs)]
    pair_scores: list[float] = []
    records: list[dict] = []
    reasons: dict[str, int] = {}
    start = time.time()
    lower, upper = sprt_bounds()
    sink = out.open("a") if out else None
    with Pool(workers) as pool:
        for rec in pool.imap_unordered(play_pair, tasks):
            pair_scores.append(rec["a_score"])
            records.append(rec)
            for g in rec["games"]:
                reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
            if sink:
                sink.write(json.dumps({"a": a, "b": b, **rec}) + "\n")
                sink.flush()
            if sprt:
                llr = sprt_llr(pair_scores, *sprt)
                if llr <= lower or llr >= upper:
                    if not quiet:
                        print(f"SPRT {'H1 accepted' if llr >= upper else 'H0 accepted'} after {len(pair_scores)} pairs (llr {llr:.2f})")
                    pool.terminate()
                    break
    if sink:
        sink.close()
    if not quiet:
        elo, lo, hi = pentanomial_elo(pair_scores)
        n = len(pair_scores)
        points = sum(pair_scores)
        print(f"{a} vs {b}: {points:.1f}/{2 * n} ({100 * points / (2 * n):.1f}%), Elo {elo:+.0f} [{lo:+.0f}, {hi:+.0f}], "
              f"{n} pairs in {time.time() - start:.0f}s; endings {dict(sorted(reasons.items()))}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(120, os.cpu_count() or 8))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sprt", type=float, nargs=2, default=None, metavar=("ELO0", "ELO1"))
    args = parser.parse_args()
    run_match(args.a, args.b, args.pairs, args.workers, args.out, tuple(args.sprt) if args.sprt else None)


if __name__ == "__main__":
    main()
