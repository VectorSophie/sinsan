# Rebuild log (2026-10 onward)

Measured results from the rebuild described in the rebuild plan (P0-P6). Every number here comes
from a script in this repo; the command is given with each result.

## P0: rules (2026-10-15)

- `engine/` (Rust) vs pyffish, `tests/differential/compare_engine.py --games 5000`: 845,503
  positions, 0 mismatches in legal moves, FEN, check, and every game-end function. Endings seen:
  checkmate 868, bikjang 543, double pass 302, move limit 525, repetition 1966, perpetual check 167.
- perft vs Fairy-Stockfish `go perft`: startpos depth 5 = 35,243,995 (exact match).
- `packages/rules` vs the Rust engine, `tests/differential/check-ts-results.ts`: 506,144 positions,
  0 mismatches (after fixing pass-in-check during bikjang).

## P1: evaluation ladder (2026-10-15)

`cd training && uv run python -m eval.ladder --pairs 200 --add ...` - 14,800 games, paired openings
(512-position suite), refereed by `sinsan_engine`. Bradley-Terry ratings, Fairy-Stockfish NNUE at
1,000 nodes per move anchored at 0, 95% intervals:

| Player | Elo |
|---|---|
| random | -2271 ± 188 |
| old v3 (56x7), policy only | -1556 ± 153 |
| old v3 (56x7), 16-visit PUCT | -1424 ± 151 |
| FSF NNUE 64 nodes | -526 ± 45 |
| FSF NNUE 128 | -460 ± 42 |
| FSF classical 1,000 | -431 ± 43 |
| FSF NNUE 256 | -427 ± 38 |
| FSF NNUE 512 | -238 ± 34 |
| FSF NNUE 1,000 | 0 |
| FSF NNUE 2,000 | +312 ± 39 |
| FSF classical 16,000 | +466 ± 48 |
| FSF NNUE 4,000 | +574 ± 46 |
| FSF NNUE 8,000 | +924 ± 61 |
| FSF classical 128,000 | +1094 ± 68 |
| FSF NNUE 16,000 | +1198 ± 67 |
| FSF NNUE 32,000 | +1492 ± 75 |
| FSF NNUE 64,000 | +1771 ± 82 |
| FSF NNUE 128,000 | +2041 ± 87 |
| FSF NNUE 256,000 | +2312 ± 93 |

Observations:
- Each doubling of FSF NNUE nodes is worth +250 to +350 Elo from 512 nodes up; below ~256 nodes
  FSF barely searches and the steps are small.
- NNUE is worth roughly 10x nodes over the classical evaluation (classical 1K ~ NNUE 256,
  classical 16K ~ NNUE 3K, classical 128K ~ NNUE 12K). The original project's labels came from
  classical evaluation at 8K nodes split over MultiPV 8.
- The original project's best model (v3) is about 900 Elo below FSF NNUE at 64 nodes. Its
  ratings are loosely constrained (it lost almost every game to every FSF setting), hence the
  wide intervals.
- Under Fairy-Stockfish janggi rules no game is drawn; FSF itself ends many games deliberately by
  bikjang or passing while ahead on points.

## P2: teacher data

- Fitted value scale on teacher games: P(win) = sigmoid(cp / 105) for FSF NNUE centipawns.
- Median gap between the teacher's best and second-best candidate: 17 cp (90th percentile 158 cp),
  so the policy target uses softmax(cp / 30) (the original project used cp / 800).
- 44% of MultiPV label rows came back with candidates out of score order (lines from different
  iterations); the generator now sorts them.
- Final set: 49,955,861 positions from 283,970 full games (FSF NNUE, 100K nodes, MultiPV 8),
  ~460 positions/s on 64 physical cores (~30 h).

## P3: supervised training

Architecture ablation on the first 9.4M positions, 14K steps x 2048 (~3 epochs), one GPU each:

| Net | Params | Samples/s | Val top-1 | Val top-3 | WDL loss | Policy-only ladder | 800 visits ladder |
|---|---|---|---|---|---|---|---|
| CNN 256x15 | 17.4M | 12.6K | 37.7% | 67.0% | 0.362 | ~-330 | ~+1100 |
| Transformer 384x10 | 34.9M | 5.9K | 41.2% | 70.8% | 0.356 | ~-40 | ~+1750 |

The transformer also wins at equal wall time (39.3% top-1 at 35 minutes vs the CNN's final 37.7%).

Main run `p3-tf512x12`: transformer d=512, 12 layers, 16 heads (60.4M params) on 35.6M positions,
52K steps x 2048, ~6.5 h on one GPU. Final val: top-1 45.1%, top-3 74.9%, WDL loss 0.340, value
accuracy 84.5%. At 800 visits against the ladder (200 games each):

| Opponent | Score | Implied rating |
|---|---|---|
| FSF NNUE 64,000 | 91.5% | +2184 |
| FSF NNUE 128,000 | 71.0% | +2197 |
| FSF NNUE 256,000 | 58.5% | +2372 |
| transformer 384x10 (both 400 visits) | 92.5% (+436) | |

Ladder rating about +2250, above every FSF setting on the ladder.

## P4: search parameters (first pass)

384x10 net, 400 visits, 200 pairs against the default (Lc0) settings:

| Change | Elo |
|---|---|
| cpuct 1.2 (default 1.745) | +28 [-4, +61] |
| fpu 0.5 (default 0.33) | +9 |
| policy temp 1.0 / 1.7 (default 1.36) | -9 / -24 |
| margin utility 0.1 / 0.3 (default 0) | -7 / -23 |
| cpuct 2.5 | -53 |
| fpu 0.2 | -51 |

Defaults kept; cpuct ~1.2 to be confirmed with a longer SPRT on the main net.

## P5: self-play RL

`rl1` started from `p3-tf512x12` on both GPUs: Gumbel self-play (200/32 visits, p_full 0.25)
produces ~35-40 recorded positions/s per GPU (about 3.5 h per 500K-per-GPU iteration).
