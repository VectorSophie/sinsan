"""Concatenates multiple labeled JSONL datasets into one training set (Phase 6: mixing random and
on-policy self-play data, docs/DATASET_DESIGN.md's "data mixture" methodology). Offsets game_id in
each input file by its index * 1,000,000 before writing, so train.py's split_by_game (game_id %
SPLIT_BUCKET_COUNT) never coincidentally merges unrelated games from different sources into the
same split bucket just because they reused small game_id numbers independently.

Run with: python3 training/generate/combine_datasets.py --input a.jsonl b.jsonl --output combined.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
GAME_ID_OFFSET_STEP = 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="filenames under training/datasets/, in order")
    parser.add_argument("--output", required=True, help="filename under training/datasets/")
    args = parser.parse_args()

    total = 0
    with (DATASETS_DIR / args.output).open("w") as out_file:
        for i, name in enumerate(args.input):
            offset = i * GAME_ID_OFFSET_STEP
            count = 0
            for line in (DATASETS_DIR / name).read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record["game_id"] += offset
                out_file.write(json.dumps(record) + "\n")
                count += 1
            print(f"{name}: {count} positions (game_id offset +{offset})")
            total += count

    print(f"Wrote {total} positions to {DATASETS_DIR / args.output}")


if __name__ == "__main__":
    main()
