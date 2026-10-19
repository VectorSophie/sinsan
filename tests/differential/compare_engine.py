"""Differential test: sinsan_engine (Rust) vs pyffish (Fairy-Stockfish `janggi`).

Plays randomized games biased toward rule edge cases (passes, bikjang, back-and-forth repetition,
checks) and compares, at every ply: legal moves, FEN, check status, and each game-end function
(no-legal-moves result, immediate end, optional end). Unlike the older compare.py, this checks game
results as well as legality, since the old rules engine was legality-correct but ended games
differently from the teacher.

Mismatches are written as JSON repros to artifacts/rule-mismatches/engine/.

Run: cd training && uv run python ../tests/differential/compare_engine.py --games 2000 --workers 120
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import pyffish
import sinsan_engine as se

VARIANT = "janggi"
FORMATIONS = ["masang-sangma", "sangma-masang", "masang-masang", "sangma-sangma"]
REPO_ROOT = Path(__file__).resolve().parents[2]
MISMATCH_DIR = REPO_ROOT / "artifacts" / "rule-mismatches" / "engine"
MAX_PLIES = 400


def ffish_end(kind: str, fen: str, moves: list[str]) -> str | None:
    """pyffish game-end calls -> winner 'w'/'b' (or 'draw'), from the side-to-move-relative score."""
    side = pyffish.get_fen(VARIANT, fen, moves).split()[1]
    if kind == "none":
        value = pyffish.game_result(VARIANT, fen, moves)
        ended = True
    elif kind == "immediate":
        ended, value = pyffish.is_immediate_game_end(VARIANT, fen, moves)
    else:
        ended, value = pyffish.is_optional_game_end(VARIANT, fen, moves)
    if not ended:
        return None
    if value == 0:
        return "draw"
    return side if value > 0 else ("b" if side == "w" else "w")


def is_pass(m: str) -> bool:
    return m[: len(m) // 2] == m[len(m) // 2 :]


def random_style(rng: random.Random) -> dict:
    """Per-side move-choice biases, randomized per game so every ending type gets exercised:
    passes (double-pass, bikjang), undo moves (repetition), checks (perpetual check), and
    capture avoidance (the 100-ply rule)."""
    return {
        "pass": rng.choice([0.0, 0.0, 0.02, 0.15]),
        "undo": rng.choice([0.0, 0.1, 0.5, 0.9]),
        "check": rng.choice([0.0, 0.3, 0.9]),
        "quiet": rng.choice([False, False, True]),
    }


def choose_move(game: se.Game, legal: list[str], style: dict, rng: random.Random) -> str:
    history = game.moves()
    if rng.random() < style["undo"] and len(history) >= 2:
        prev = history[-2]
        split = next(i for i in range(2, len(prev)) if prev[i].isalpha())
        reverse = prev[split:] + prev[:split]
        if reverse in legal and not is_pass(prev):
            return reverse
    passes = [m for m in legal if is_pass(m)]
    if passes and rng.random() < style["pass"]:
        return passes[0]
    pool = [m for m in legal if not is_pass(m)] or legal
    if rng.random() < style["check"]:
        checking = []
        for m in rng.sample(pool, min(len(pool), 16)):
            game.push(m)
            if game.in_check():
                checking.append(m)
            game.pop()
        if checking:
            return rng.choice(checking)
    if style["quiet"]:
        before = game.point_balance()
        quiet = []
        for m in rng.sample(pool, min(len(pool), 16)):
            game.push(m)
            if game.point_balance() == before:
                quiet.append(m)
            game.pop()
        if quiet:
            return rng.choice(quiet)
    return rng.choice(pool)


def run_game(seed: int) -> tuple[int, list[dict], dict]:
    rng = random.Random(seed)
    game = se.Game(cho=rng.choice(FORMATIONS), han=rng.choice(FORMATIONS))
    root = game.fen()
    styles = {"w": random_style(rng), "b": random_style(rng)}
    mismatches: list[dict] = []
    stats = {"positions": 0, "endings": {}}

    def report(kind: str, ours, theirs) -> None:
        mismatches.append({"seed": seed, "kind": kind, "root": root, "moves": game.moves(), "ours": ours, "pyffish": theirs})

    for _ in range(MAX_PLIES):
        moves = game.moves()
        stats["positions"] += 1
        # Immediate ends first: pyffish returns no legal moves once one has fired.
        ours_imm = game.immediate_end()
        theirs_imm = ffish_end("immediate", root, moves)
        if (ours_imm[0] if ours_imm else None) != theirs_imm:
            report("immediate", ours_imm, theirs_imm)
            break
        if ours_imm:
            stats["endings"][ours_imm[1]] = 1
            break
        ours_legal = sorted(game.legal_moves())
        theirs_legal = sorted(pyffish.legal_moves(VARIANT, root, moves))
        if ours_legal != theirs_legal:
            report("legal", sorted(set(ours_legal) - set(theirs_legal)), sorted(set(theirs_legal) - set(ours_legal)))
            break
        if game.fen() != pyffish.get_fen(VARIANT, root, moves):
            report("fen", game.fen(), pyffish.get_fen(VARIANT, root, moves))
            break
        theirs_check = pyffish.gives_check(VARIANT, root, moves)
        if game.in_check() != theirs_check:
            report("check", game.in_check(), theirs_check)
            break

        ours_opt = game.optional_end()
        theirs_opt = ffish_end("optional", root, moves)
        if (ours_opt[0] if ours_opt else None) != theirs_opt:
            report("optional", ours_opt, theirs_opt)
            break
        if not ours_legal:
            theirs_none = ffish_end("none", root, moves)
            res = game.result()
            if res is None or res[0] != theirs_none:
                report("no-moves", res, theirs_none)
            stats["endings"][res[1] if res else "?"] = 1
            break

        result = game.result()
        if result is not None:
            stats["endings"][result[1]] = 1
            break
        game.push(choose_move(game, ours_legal, styles[game.side_to_move], rng))
    return seed, mismatches, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    MISMATCH_DIR.mkdir(parents=True, exist_ok=True)
    total_positions = 0
    endings: dict[str, int] = {}
    failures = 0
    seeds = range(args.seed, args.seed + args.games)
    with Pool(args.workers) as pool:
        for seed, mismatches, stats in pool.imap_unordered(run_game, seeds, chunksize=4):
            total_positions += stats["positions"]
            for k in stats["endings"]:
                endings[k] = endings.get(k, 0) + 1
            for m in mismatches:
                failures += 1
                path = MISMATCH_DIR / f"{m['kind']}-{seed}.json"
                path.write_text(json.dumps(m, indent=1))
                if failures <= 10:
                    print(f"MISMATCH {m['kind']} seed={seed} ply={len(m['moves'])}: ours={m['ours']} pyffish={m['pyffish']}")
    print(f"{args.games} games, {total_positions} positions compared, {failures} mismatching games")
    print("game endings seen:", dict(sorted(endings.items())))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
