"""Teacher data (P2): Fairy-Stockfish NNUE plays full Janggi games and labels every position.

Each position gets one MultiPV search (default 8 lines, 100K nodes, about depth 10). The search
labels the position (every candidate's score, kept raw so the policy temperature is a training-
time choice) and also chooses the move played:

- opening (first 8-20 plies): sample candidates within 150 cp of the best, weighted by
  exp((cp - best) / 60), so the 16 formation pairs branch into many different games;
- afterwards: the best move, except 5% of the time a candidate within 60 cp (weight T = 30 cp).

Games are played to the referee's end (sinsan_engine, i.e. FSF janggi rules), so middlegames and
endgames are covered, and each game stores its result and final point balance for value targets.

Output: training/datasets/teacher/<run>/w<worker>.jsonl.gz, one game per line. Restartable: each
worker resumes counting from what is already on disk.

Run: cd training && uv run python -m generate.teacher_games --run nnue100k --positions 50000000
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import time
from multiprocessing import Process, Queue
from queue import Empty
from pathlib import Path

import sinsan_engine as se

from eval.uci import fairy_stockfish

FORMATIONS = ["masang-sangma", "sangma-masang", "masang-masang", "sangma-sangma"]
OUT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "teacher"
MAX_PLIES = 600
MATE_CP = 30000


def cp_equivalent(cp: int | None, mate: int | None) -> int:
    if cp is not None:
        return cp
    assert mate is not None
    return (MATE_CP - abs(mate)) * (1 if mate > 0 else -1)


def sample(cands: list[tuple[str, int]], window: int, temperature: float, rng: random.Random) -> str:
    """cands must be sorted best first."""
    best = cands[0][1]
    pool = [(m, s) for m, s in cands if best - s <= window]
    weights = [math.exp((s - best) / temperature) for _, s in pool]
    return rng.choices([m for m, _ in pool], weights)[0]


def play_game(engine, rng: random.Random, nodes: int) -> dict:
    cho, han = rng.choice(FORMATIONS), rng.choice(FORMATIONS)
    game = se.Game(cho=cho, han=han)
    root = game.root_fen()
    engine.new_game()
    opening_plies = rng.randint(8, 20)
    labels = []
    result = game.result()
    while result is None and game.plies < MAX_PLIES:
        info = engine.search(root, game.moves(), f"nodes {nodes}")
        legal = set(game.legal_moves())
        # MultiPV lines can come from different iterations, so sort: best first is relied on downstream.
        cands = sorted(((m, cp_equivalent(cp, mate)) for m, cp, mate in info.multipv if m in legal),
                       key=lambda c: -c[1])
        if not cands:
            # FSF sees the game as over but the referee does not (should not happen); stop cleanly.
            break
        labels.append({"c": [[m, s] for m, s in cands], "d": info.depth})
        if game.plies < opening_plies:
            move = sample(cands, 150, 60.0, rng)
        elif rng.random() < 0.05:
            move = sample(cands, 60, 30.0, rng)
        else:
            move = cands[0][0]
        game.push(move)
        result = game.result()
    return {
        "cho": cho,
        "han": han,
        "moves": game.moves(),
        "labels": labels,
        "result": list(result) if result else None,
        "points": game.point_balance(),
        "nodes": nodes,
    }


def worker(wid: int, run_dir: Path, quota: int, nodes: int, multipv: int, seed: int, progress: Queue) -> None:
    path = run_dir / f"w{wid:03d}.jsonl.gz"
    done = 0
    games_done = 0
    if path.exists():
        try:
            with gzip.open(path, "rt") as f:
                for line in f:
                    done += len(json.loads(line)["labels"])
                    games_done += 1
        except (EOFError, OSError, json.JSONDecodeError):
            # A torn final record from a killed run: keep the intact prefix.
            good = []
            with gzip.open(path, "rt") as f:
                try:
                    for line in f:
                        json.loads(line)
                        good.append(line)
                except (EOFError, OSError, json.JSONDecodeError):
                    pass
            with gzip.open(path, "wt") as f:
                f.writelines(good)
            done = sum(len(json.loads(l)["labels"]) for l in good)
            games_done = len(good)
    rng = random.Random(seed * 1_000_003 + wid * 7919 + games_done)
    engine = fairy_stockfish(nnue=True, multipv=multipv, hash_mb=64)
    while done < quota:
        try:
            record = play_game(engine, rng, nodes)
        except Exception as exc:  # one bad game must not stall the whole run
            print(f"worker {wid}: game failed ({exc!r}); restarting engine", flush=True)
            engine.close()
            engine = fairy_stockfish(nnue=True, multipv=multipv, hash_mb=64)
            continue
        with gzip.open(path, "at") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        done += len(record["labels"])
        progress.put(len(record["labels"]))
    engine.close()
    progress.put(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="nnue100k")
    parser.add_argument("--positions", type=int, default=50_000_000)
    parser.add_argument("--workers", type=int, default=min(120, os.cpu_count() or 8))
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run_dir = OUT_ROOT / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))
    quota = math.ceil(args.positions / args.workers)
    progress: Queue = Queue()
    procs = [Process(target=worker, args=(w, run_dir, quota, args.nodes, args.multipv, args.seed, progress))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    finished, positions, games, start, last = 0, 0, 0, time.time(), 0.0
    while finished < len(procs):
        try:
            item = progress.get(timeout=60)
        except Empty:
            if not any(p.is_alive() for p in procs):
                break
            continue
        if item is None:
            finished += 1
            continue
        positions += item
        games += 1
        if time.time() - last > 60:
            last = time.time()
            rate = positions / (last - start)
            print(f"{positions:,} new positions in {games:,} games ({rate:,.0f}/s, "
                  f"{(args.positions - positions) / max(rate, 1) / 3600:.1f} h left at this rate)", flush=True)
    for p in procs:
        p.join()
    print(f"done: {positions:,} new positions in {games:,} games")


if __name__ == "__main__":
    main()
