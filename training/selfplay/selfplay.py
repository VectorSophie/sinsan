"""Self-play data generation (P5): Gumbel search with playout-cap randomization.

One process per GPU plays many games at once. Every move runs a Gumbel sequential-halving search
from the (reused) tree: with probability --p-full a full search (--full visits over --m-full
candidates) whose improved policy becomes a training target, otherwise a fast search (--fast
visits) that only advances the game (KataGo's playout-cap randomization). The first plies sample
from the improved policy for opening variety; afterwards the Gumbel choice is played.

Finished games are written as shards (same layout as data/shards.py, plus pol_probs) to
datasets/selfplay/<run>/<chunk>/, with only full-search positions recorded.

Run: cd training && uv run python -m selfplay.selfplay --net runs/x/ema-0100000.pt --run sp1 --games 1024 --positions 2000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sinsan_engine as se
import torch

from eval.net_arena import NetEvaluator

FORMATIONS = ["masang-sangma", "sangma-masang", "masang-masang", "sangma-sangma"]
OUT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "selfplay"
POLICY_K = 64
MAX_PLIES = 500


class ChunkWriter:
    """Accumulates finished games and writes them as shard directories of ~chunk positions."""

    def __init__(self, out: Path, chunk: int) -> None:
        self.out, self.chunk = out, chunk
        self.rows: dict[str, list[np.ndarray]] = {}
        self.count = 0
        self.index = len(list(out.glob("c[0-9]*")))
        self.games = 0

    def add_game(self, cho: str, han: str, moves: list[str], records: dict[int, tuple[list[str], list[float]]],
                 result: tuple[str, str] | None, points: int) -> int:
        if result is None or not records:
            return 0
        plies = sorted(records)
        cands = [records[p][0] if p in records else [] for p in range(plies[-1] + 1)]
        enc = se.encode_game(cho, han, moves, cands)
        p_all, k = len(cands), enc["k"]
        rows = np.array(plies)
        boards = np.frombuffer(enc["boards"], np.uint8).reshape(p_all, 4, 90)[rows]
        scalars = np.frombuffer(enc["scalars"], np.float32).reshape(p_all, -1)[rows]
        legal = np.frombuffer(enc["legal"], np.uint8).reshape(p_all, -1)[rows]
        stm = np.frombuffer(enc["stm"], np.uint8)[rows]
        actions = np.full((len(rows), POLICY_K), -1, np.int16)
        actions[:, :k] = np.frombuffer(enc["cand_actions"], np.int16).reshape(p_all, k)[rows]
        probs = np.zeros((len(rows), POLICY_K), np.float16)
        for i, p in enumerate(plies):
            pr = records[p][1]
            probs[i, : len(pr)] = pr
        winner = {"w": 0, "b": 1}[result[0]]
        cho_margin = points - 1.5
        cols = {
            "boards": boards, "scalars": scalars, "legal": legal, "cand_actions": actions, "pol_probs": probs,
            "cand_scores": np.zeros((len(rows), 8), np.int16), "depth": np.zeros(len(rows), np.uint8),
            "result": np.where(stm == winner, 1, -1).astype(np.int8),
            "margin": np.where(stm == 0, cho_margin, -cho_margin).astype(np.float32),
            "plies_left": (len(moves) - rows).astype(np.int16),
            "game": np.full(len(rows), self.games, np.int32), "ply": rows.astype(np.int16),
        }
        self.games += 1
        for name, arr in cols.items():
            self.rows.setdefault(name, []).append(arr)
        self.count += len(rows)
        if self.count >= self.chunk:
            self.flush()
        return len(rows)

    def flush(self) -> None:
        if not self.count:
            return
        d = self.out / f"c{self.index:05d}"
        tmp = d.with_suffix(".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        for name, parts in self.rows.items():
            np.save(tmp / f"{name}.npy", np.concatenate(parts))
        tmp.rename(d)
        self.index += 1
        self.rows, self.count = {}, 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--positions", type=int, default=1_000_000, help="recorded (full-search) positions to produce")
    p.add_argument("--games", type=int, default=1024, help="concurrent games")
    p.add_argument("--full", type=int, default=200)
    p.add_argument("--fast", type=int, default=32)
    p.add_argument("--m-full", type=int, default=16)
    p.add_argument("--m-fast", type=int, default=8)
    p.add_argument("--p-full", type=float, default=0.25)
    p.add_argument("--sample-plies", type=int, default=12, help="plies that sample from the improved policy")
    p.add_argument("--leaves", type=int, default=4)
    p.add_argument("--chunk", type=int, default=200_000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = OUT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    rng = np.random.default_rng(args.seed + int(time.time()))
    evaluate = NetEvaluator(args.net, torch.device(args.device))
    batch = se.SearchBatch(policy_temp=1.0, fpu_reduction=0.0, fpu_reduction_root=0.0)
    writer = ChunkWriter(out, args.chunk)

    games: list[dict] = []

    def new_game(slot: int) -> dict:
        cho, han = FORMATIONS[rng.integers(4)], FORMATIONS[rng.integers(4)]
        batch.set_root(slot, cho, han, [])
        return {"cho": cho, "han": han, "g": se.Game(cho=cho, han=han), "records": {}, "phase": "expand", "full": False}

    for slot in range(args.games):
        games.append(new_game(slot))

    produced, finished, start, last_report = 0, 0, time.time(), 0.0
    while produced < args.positions:
        d = batch.collect(args.leaves)
        if d["count"]:
            batch.apply(*evaluate(d))
        for slot, gm in enumerate(games):
            if gm["phase"] == "expand" and batch.root_expanded(slot):
                gm["full"] = rng.random() < args.p_full
                k = len(batch.root_children(slot))
                m, budget = (args.m_full, args.full) if gm["full"] else (args.m_fast, args.fast)
                batch.gumbel_start(slot, m, budget, rng.gumbel(size=k).astype(np.float32).tolist())
                gm["phase"] = "search"
            elif gm["phase"] == "search" and batch.gumbel_done(slot):
                move, policy, _ = batch.gumbel_result(slot)
                g = gm["g"]
                if gm["full"]:
                    top = sorted(policy, key=lambda x: -x[1])[:POLICY_K]
                    gm["records"][g.plies] = ([m for m, _ in top], [pr for _, pr in top])
                if g.plies < args.sample_plies:
                    moves, probs = zip(*policy)
                    probs = np.asarray(probs, dtype=np.float64)
                    move = moves[rng.choice(len(moves), p=probs / probs.sum())]
                g.push(move)
                batch.advance(slot, move)
                result = g.result()
                if result is not None or g.plies >= MAX_PLIES:
                    produced += writer.add_game(gm["cho"], gm["han"], g.moves(), gm["records"], result, g.point_balance())
                    finished += 1
                    games[slot] = new_game(slot)
                else:
                    gm["phase"] = "expand"
        if time.time() - last_report > 60:
            last_report = time.time()
            rate = produced / (last_report - start)
            print(f"{produced:,} positions from {finished:,} games ({rate:,.0f}/s)", flush=True)
    writer.flush()
    print(f"done: {produced:,} positions from {finished:,} games in {(time.time() - start) / 3600:.1f} h")


if __name__ == "__main__":
    main()
