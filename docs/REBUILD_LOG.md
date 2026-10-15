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
