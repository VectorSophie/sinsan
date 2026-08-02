#!/usr/bin/env python3
"""Section 24.3's training-time estimator. Extrapolates from this project's own real, host-
measured throughput numbers (docs/BENCHMARK_PLAN.md Sections 24.1/24.2/24.4) - never a guess or
a generic FLOPs model. Only covers the two tiers actually measured so far (32x4 smoke, 48x6
baseline); a different --channels/--blocks would need a real measurement of its own before this
script could honestly estimate it, so it deliberately doesn't try to interpolate across sizes.

Usage: python3 scripts/estimate-training-time.py --positions 50000 --epochs 20 --tier baseline
"""

from __future__ import annotations

import argparse

# docs/BENCHMARK_PLAN.md Section 24.1 - single-process Fairy-Stockfish, nodes=8000, MultiPV=8.
TEACHER_POSITIONS_PER_SEC = 512 / 44.5

# docs/BENCHMARK_PLAN.md Sections 24.2/24.4 - real measured training throughput per tier, at the
# batch size actually used for that measurement (not interpolated to other batch sizes).
TIERS = {
    "smoke": {
        "label": "32x4 Tiny Baseline",
        "batch_size": 32,
        # 79-100ms/step measured, noisy - midpoint of the ~320-400 samples/sec range. Pure
        # step-compute only (no full-run wall-clock total was captured for this tier the way
        # Section 24.4 captured one for "baseline"), so this likely understates real wall time
        # by a similar ~35% eval/checkpoint overhead margin - not yet verified for this tier.
        "samples_per_sec": 360.0,
    },
    "baseline": {
        "label": "48x6 Main Candidate",
        "batch_size": 256,
        # Real completed run (docs/BENCHMARK_PLAN.md Section 24.4): 3140 steps x 256 batch over
        # 73.3 min WALL TIME (not the 286 samples/sec quoted for raw step compute alone) - wall
        # time is ~36% higher than pure step compute because it also includes per-epoch
        # validation/test eval, checkpoint saving, and data loading, which is what an estimate
        # actually needs to be useful. 3140 * 256 / (73.3 * 60) = ~183 samples/sec effective.
        "samples_per_sec": 183.0,
    },
}

TRAIN_SPLIT_FRACTION = 0.8  # approximate - the real split is game-level, not position-level.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=int, required=True, help="total self-play positions to generate+label")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tier", choices=TIERS.keys(), default="baseline")
    args = parser.parse_args()

    tier = TIERS[args.tier]
    labeling_sec = args.positions / TEACHER_POSITIONS_PER_SEC
    train_positions = args.positions * TRAIN_SPLIT_FRACTION
    training_sec = (train_positions * args.epochs) / tier["samples_per_sec"]
    export_sec = 30  # measured <1 min every run so far; not the bottleneck, not worth modeling finely.
    total_sec = labeling_sec + training_sec + export_sec

    def fmt(sec: float) -> str:
        return f"{sec / 60:.1f} min"

    print(f"Tier: {tier['label']} (batch_size={tier['batch_size']}, {tier['samples_per_sec']:.0f} samples/sec measured)")
    print(f"Labeling ~{fmt(labeling_sec)} (extrapolated from {TEACHER_POSITIONS_PER_SEC:.1f} pos/sec, Section 24.1)")
    print(f"Training ~{fmt(training_sec)} ({args.epochs} epochs over ~{train_positions:.0f} train-split positions)")
    print(f"Export   ~{fmt(export_sec)} (measured, not the bottleneck)")
    print(f"Total    ~{fmt(total_sec)}")
    print("\nThis is an extrapolation from smaller/different-sized real runs, not a timed execution")
    print("of this exact configuration - re-measure and update docs/BENCHMARK_PLAN.md after running it.")


if __name__ == "__main__":
    main()
