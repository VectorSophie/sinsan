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

The first recipe (lr 2e-4, fixed steps) overfit: round 0 lost 217 Elo to its parent. The loop was
reworked into continuous self-play workers (one per GPU in `runs/<run>/gpus.txt`, re-read every
minute) and a trainer that runs once 1M new positions exist, mixing 75% self-play from the latest
8M positions with 25% teacher data, at lr 1e-4 and steps = ceil(new × 4 / (2048 × 0.75)).
Each candidate is gated against the current best at 128 visits, 200 pairs, promoted above 50%.

| round | gate vs best (128 visits) | promoted |
|---|---|---|
| 0 | -217 [-258, -181] | (old recipe) |
| 1 | -65 [-99, -32] | |
| 2 | +38 [+7, +71] | yes |
| 3-4 | -76, -40 | |
| 5 | +33 [+1, +66] | yes |
| 6 | 0 | |
| 7 | +26 [-7, +59] | yes |
| 8 | -19 | |
| 9 | +23 [-12, +57] | yes |
| 10 | +28 [-3, +59] | yes |
| 11-12 | -10, -5 | |
| 13 | +10 [-21, +42] | yes |
| 14-15 | -14, -10 | |
| 16 | +19 [-16, +54] | yes |

Gates at 128 visits look nearly flat, but strength at 800 visits keeps climbing. Ladder checks
(800 visits vs FSF NNUE 256K, 100 pairs; the 256K rung sits at about 2312):

| net | score | ladder Elo |
|---|---|---|
| p3 (before RL) | | ~2250 |
| round 2 | 63.0% | ~2404 |
| round 9 | 64.0% | ~2412 (2363~2464); +51 [+18, +84] head-to-head vs p3 |
| round 13 | 77.0% | ~2520 (2470~2580) |

Web presets measured on the same ladder (results in `eval/results/preset-*.json`):

| preset | match | ladder Elo |
|---|---|---|
| Casual (small net, 1 visit, temperature 1) | 1/200 vs FSF 64 nodes; 200/200 vs random | <= -1256 |
| Club (small net, 16 visits) | 60% vs FSF 1K | 70 (17~127) |
| Strong (latest net, 64 visits) | 76% vs FSF 16K | ~1398 (1346~1460) |
| Maximum (6400 visits) | 93/100 vs FSF 256K | ~2761 (2658~2988) |

Between 2026-10-19 and 10-20 the run used GPU 1 only (GPU 0 was busy with another job), which
halved self-play to ~31 positions/s; edit `gpus.txt` to shrink or grow without a restart.

Round 16 (promoted at +19 over round 13) rated 73.5% vs FSF 256K at 800 visits, about 2489
(2440~2546): level with round 13 within the error.

## P6: browser engine (2026-10-20)

The browser runs the same search as the server: `engine/src/wasm.rs` exposes one search tree to
JavaScript (`WasmSearch`: collect leaves, apply logits and values), built to WebAssembly with
`engine/build-wasm.sh` (148 KB, 57 KB gzipped). The network runs in onnxruntime-web inside a worker
(`apps/web/src/student-worker.ts`), which answers the same requests as `/api/move` with the same
reply, so the analysis panel, hints and presets work unchanged.

Student: a 6-layer, d=192 transformer (6 heads, FFN x2, 10.7M parameters), trained from scratch
for 30K steps of 2048 on 60% RL self-play (all 10.9M rl1 positions) and 40% teacher data, lr 1e-3
(80 min on GPU 0 next to self-play). Validation top-1 40.2%, value accuracy 84.1% (35M net: 41.2%,
83.2%). CPU candidates benchmarked through onnxruntime before training: d=192x6 ran ~130 evals/s
single-threaded, d=256x8 ~40.

Export (`training/export/export_onnx.py`): fp32 ONNX matches PyTorch exactly; int8 dynamic
quantization of the MatMuls makes it 12 MB with policy KL 0.002 and value error 0.02 on validation
positions. In play the int8 file scored 57/100 against its PyTorch original at 64 visits
(+49 [-16, +117]), so quantization costs nothing measurable. `net_arena` plays `.onnx` files on
the CPU through onnxruntime (`OnnxEvaluator`), so the browser model is rated from the exact file.

| player | match | ladder Elo |
|---|---|---|
| student, PyTorch, 800 visits | 64.5% vs FSF 64K | ~1875 (1822~1933) |

In headless Chromium with cross-origin isolation the int8 student searches ~350 visits/s (400
visits in about a second); single-threaded, without the COOP/COEP headers, about 80 visits/s.
