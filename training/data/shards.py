"""Converts teacher games (generate/teacher_games.py output) into memory-mappable training shards.

One shard per input worker file, written as a directory of .npy arrays (one row per position):

  boards        uint8  [N, 4, 90]   canonical piece codes, current position first (engine/src/features.rs)
  scalars       f32    [N, 10]      see sinsan_engine.SCALAR_NAMES
  legal         uint8  [N, 676]     packed canonical legal-action mask
  cand_actions  int16  [N, 8]       teacher candidates, canonical actions, best first (-1 = none)
  cand_scores   int16  [N, 8]       their scores in centipawns, side-to-move view (mates as +-(30000-n))
  depth         uint8  [N]          teacher search depth
  result        int8   [N]          game outcome for the side to move: +1 win, -1 loss, 0 draw/unfinished
  margin        f32    [N]          final point margin for the side to move, with Han's 1.5 compensation
  plies_left    int16  [N]          plies until the game ended
  game          int32  [N]          game index within the shard
  ply           int16  [N]

Games are split into train/val by a hash of (shard, game index): val is about 1%.

Run: cd training && uv run python -m data.shards --run nnue100k
"""

from __future__ import annotations

import argparse
import gzip
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sinsan_engine as se

TRAINING_DIR = Path(__file__).resolve().parents[1]
TEACHER_DIR = TRAINING_DIR / "datasets" / "teacher"
SHARD_DIR = TRAINING_DIR / "datasets" / "shards"
K = 8
VAL_PERCENT = 1


def split_of(shard: str, game_index: int) -> str:
    return "val" if zlib.crc32(f"{shard}:{game_index}".encode()) % 100 < VAL_PERCENT else "train"


def convert(args: tuple[Path, Path]) -> dict[str, int]:
    src, out_root = args
    shard = src.name.split(".")[0]
    cols: dict[str, dict[str, list]] = {s: {k: [] for k in ("boards", "scalars", "legal", "cand_actions", "cand_scores",
                                                         "depth", "result", "margin", "plies_left", "game", "ply")}
                                        for s in ("train", "val")}
    with gzip.open(src, "rt") as f:
        lines = []
        try:
            for line in f:
                lines.append(line)
        except (EOFError, OSError):
            pass  # torn tail of a file still being written: use the complete games
    for gi, line in enumerate(lines):
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        if g["result"] is None or not g["labels"]:
            continue
        labels = g["labels"]
        cands = [[c[0] for c in lab["c"][:K]] for lab in labels]
        enc = se.encode_game(g["cho"], g["han"], g["moves"], cands)
        p, k = len(labels), enc["k"]
        c = cols[split_of(shard, gi)]
        c["boards"].append(np.frombuffer(enc["boards"], np.uint8).reshape(p, 4, 90))
        c["scalars"].append(np.frombuffer(enc["scalars"], np.float32).reshape(p, -1))
        c["legal"].append(np.frombuffer(enc["legal"], np.uint8).reshape(p, -1))
        actions = np.full((p, K), -1, np.int16)
        actions[:, :k] = np.frombuffer(enc["cand_actions"], np.int16).reshape(p, k)
        c["cand_actions"].append(actions)
        scores = np.zeros((p, K), np.int16)
        for i, lab in enumerate(labels):
            s = [max(-32000, min(32000, x[1])) for x in lab["c"][:K]]
            scores[i, : len(s)] = s
        c["cand_scores"].append(scores)
        c["depth"].append(np.array([lab["d"] for lab in labels], np.uint8))
        stm = np.frombuffer(enc["stm"], np.uint8)
        winner = {"w": 0, "b": 1}[g["result"][0]]
        c["result"].append(np.where(stm == winner, 1, -1).astype(np.int8))
        cho_margin = g["points"] - 1.5
        c["margin"].append(np.where(stm == 0, cho_margin, -cho_margin).astype(np.float32))
        end = len(g["moves"])
        c["plies_left"].append((end - np.arange(p)).astype(np.int16))
        c["game"].append(np.full(p, gi, np.int32))
        c["ply"].append(np.arange(p, dtype=np.int16))
    counts = {}
    for split, c in cols.items():
        if not c["boards"]:
            counts[split] = 0
            continue
        out = out_root / split / shard
        out.mkdir(parents=True, exist_ok=True)
        for name, parts in c.items():
            np.save(out / f"{name}.npy", np.concatenate(parts))
        counts[split] = sum(len(x) for x in c["ply"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="nnue100k")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    out_root = SHARD_DIR / args.run
    sources = sorted((TEACHER_DIR / args.run).glob("w*.jsonl.gz"))
    totals = {"train": 0, "val": 0}
    with Pool(args.workers) as pool:
        for counts in pool.imap_unordered(convert, [(s, out_root) for s in sources]):
            for k, v in counts.items():
                totals[k] += v
    print(f"{len(sources)} shards -> {out_root}: {totals['train']:,} train / {totals['val']:,} val positions")


if __name__ == "__main__":
    main()
