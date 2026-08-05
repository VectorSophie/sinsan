# 神算 Sinsan

A Korean Janggi engine distilled into a browser-native model under 500KiB, in the spirit of
[millionco/moka](https://github.com/millionco/moka)'s approach to Go - reimagined for Janggi's own
rules, action space, and product identity rather than mechanically reskinned. See
`docs/RESEARCH.md` and `docs/PRODUCT_VISION.md` for why this project exists and what it is and
isn't trying to be.

## Status

- **Phase 0 (research + documentation + repo scaffold): complete.** See `docs/` for the nine
  research/design documents and the license audit.
- **Phase 1 (rules engine): implemented, tested, and differentially verified against pyffish.**
  `packages/rules` has immutable position representation, full piece move generation (including
  cannon screening, palace diagonals, horse/elephant blocking), self-check-safe legal move
  generation, all four starting formations, bikjang/repetition/material-count game-result
  detection, and FEN-like serialization. 64 unit tests pass; type-checks clean.
  **Differential testing against `pyffish`** (`tests/differential/`) found and fixed a real gap:
  bikjang must be *resolved* (interposition or pass) once it already exists, not just detected as
  a game-end condition - an over-broad first fix was itself caught by the same harness producing
  a counterexample in the opposite direction. Verified with 10,000 random positions across all
  four rule profiles, 0 mismatches after the fix. See docs/RULES.md's "Confirmed finding" section.
- **Phase 2 (playable board demo): working vertical slice.** `apps/web` (Vite + vanilla TS) +
  `packages/ui` (wraps chessgroundx) render a real 9x10 board with the vendored wooden
  board/piece assets, legal-move dots, last-move/check highlighting, and move animation. Human
  vs. a random-legal-move AI is playable end-to-end; verified live in Chromium. Known gap:
  chessgroundx's built-in coordinate labels are 8x8-only and disabled for now (`coordinates:
  false` in `packages/ui/src/board-view.ts`) rather than reimplemented for 9x10.
- **Phase 3 (teacher engine): built and verified, adapter prototype working.** Fairy-Stockfish
  built from a pinned commit with `largeboards=yes` (`scripts/build-teacher.sh`); the `janggi`
  variant's startpos matches Sinsan's own default exactly. `training/teacher/adapter.py` drives
  it over UCI (MultiPV, fixed node budget, deterministic) — run `python3
  training/teacher/adapter.py` for a live self-check.
- **Phase 4 (smoke pipeline): complete, end to end, verified live in Chromium.**
  `packages/action-space` (90×60+1 templates, cross-language-verified against a TypeScript golden
  fixture); `training/generate/self-play.ts` + `training/generate/label_dataset.py` produced a
  real 512-position labeled smoke dataset; `training/train/train.py` trained the 32×4 Tiny
  Baseline (107,426 params) on it; `training/export/export.py` produced a 107.5KiB INT8 artifact;
  `packages/model-runtime`'s hand-written TS forward pass is numerically verified against the
  PyTorch checkpoint that produced it (`tests/model/model-runtime-parity.test.ts`); a Worker-hosted
  "Sinsan (policy, smoke model)" AI option in `apps/web` runs real inference in the browser
  (fetch + SHA-256 digest verify + Worker + forward pass), confirmed live. Two real bugs were
  found and fixed along the way (self-play not checking bikjang before continuing a game; a
  `0 * -inf = nan` trap in the masked policy loss) — see `docs/BENCHMARK_PLAN.md` for what that
  revealed about current inference latency (301-360ms, unoptimized `conv2d` — a real, named gap,
  not a claimed target).
- **`packages/search` (PUCT/MCTS): implemented and verified live.** Negamax-style PUCT over
  `packages/rules` (legality) and a caller-supplied evaluator (decoupled from model-runtime by
  design); terminal positions short-circuit without a model call; `SearchTree` reuses subtrees
  across a game's moves (proven via a dedicated test, not just asserted). A "Sinsan (16-visit
  search)" AI option in `apps/web` runs real PUCT search with the Worker-hosted model as the leaf
  evaluator, confirmed live in Chromium (16/16 visits used, ~3.6s for 16 sequential Worker calls
  at the smoke model's current per-call latency). 64/128-visit tiers use the same API but aren't
  wired into the UI yet given that latency.
- **Inference latency investigated, not just reported.** Separated one-time cost (digest+construct,
  ~5-30ms) from steady-state `conv2d` cost (the actual bottleneck, ~5-8ms per tower layer). A
  pre-padding optimization was tried, measured worse at the full-model level once allocation cost
  was included (not excluded, as an earlier isolated comparison mistakenly did), and reverted -
  see `docs/BENCHMARK_PLAN.md` for the full writeup and the concrete next thing to try (a reusable
  scratch buffer, not more loop tweaks).
- **Phase 5 (48x6 Main Candidate baseline model): trained, exported, and parity-verified on real
  data.** 50,000 self-play positions (948 games) labeled by the teacher and trained for 20 epochs
  (73.3 min measured, 286 samples/sec) produced `sinsan-baseline-v0`: 293,746 params, 291.8KiB INT8
  export (well inside the 480KiB budget), 100% legal-move rate on the validation split, val loss
  tracking train loss with no overfitting, and all 3 TS<->PyTorch parity tests passing against the
  real export. See `docs/BENCHMARK_PLAN.md` Section 24.4 for the full real-measured numbers.
- **Phase 6 iteration 1 (on-policy self-play): real, measured, mixed picture - not a uniform
  win, reported honestly.** 25,000 positions generated from `sinsan-baseline-v0`'s own policy
  (temperature-sampled, not greedy, for game diversity), combined with the existing 50,000
  random-self-play positions, trained a larger 56x7 model (`sinsan-v2-56x7`, 445,706 params,
  441.7KiB). Across several `apps/arena` comparisons at increasing search depth: v2 clearly beats
  v1 head-to-head at 16-visit search (5-0-15, +89 Elo est.) and clearly beats random at 64-visit
  search (11-1-5, +127 Elo est., the strongest result measured) - but the deepest head-to-head
  (both sides at 64-visit search) came back close to even, with v1 marginally ahead in decisive
  games (1-2-15 over 18 games, cut short by a host memory issue, not enough games to call it
  either way). `apps/arena` itself had a real bug fixed along the way: deterministic players plus
  only 16 possible formation combos meant large `--pairs` counts produced redundant duplicate
  games, not new data - fixed by adding random opening plies (`--random-plies`, default 4). See
  `docs/BENCHMARK_PLAN.md`'s Phase 6 section for the full numbers, the bug-fix story, and an
  honest read on what does and doesn't hold up. All three model variants (smoke/baseline/v2), both
  policy-only and 16/64-visit search, are wired into the playable board's AI dropdown and verified
  live; `apps/arena` (the headless match runner used for all of the above) is documented below.
  Also worth knowing: this host hit three escalating out-of-memory events during these long
  background runs (see `docs/BENCHMARK_PLAN.md`), killing unrelated processes including VS Code
  and Chromium - a real host health issue, not a bug in this project (resolved by a since-happened
  host restart, though the underlying cause in the host's other services was never identified).
- **First real external calibration: `apps/arena` can now play Fairy-Stockfish, not just Sinsan's
  own checkpoints.** A `stockfish` player kind drives a real Fairy-Stockfish subprocess over UCI
  (same protocol `training/teacher/adapter.py` uses for labeling), using `UCI_LimitStrength`/
  `UCI_Elo` to set a target strength - with the explicit, load-bearing caveat that this is a
  chess-calibrated mechanism whose meaning for the Janggi variant is unverified, so results are
  reported as "vs Fairy-Stockfish at UCI_Elo=N," never translated into a dan-rank claim. Three real
  data points characterize v2's strength curve: **UCI_Elo 500 → 65.0% score (4-1-5, v2 wins)**,
  **UCI_Elo 800 → 40.0% (0-2-8, v2 already behind)**, **UCI_Elo 1200 → 0.0% (0-10-0, total
  shutout)** - the 0-10 result was verified not to be a bug (v2 still beat random normally right
  after). The real crossover point (~50% score) sits roughly in the UCI_Elo 600-700 range by
  interpolation. A direct web search for a Janggi dan-to-Elo conversion found none exists publicly
  - Janggi's own amateur rank structure (15급 to 7단) is documented, but nothing connects it
  numerically to Elo or engine strength, so this bracket is reported as exactly what it is (a real
  bound against a real engine) and explicitly not translated into any dan-rank claim. See
  `docs/BENCHMARK_PLAN.md`'s Phase 6 section for the full numbers.
- **Phase 6 iteration 2 (v3): a null result, reported honestly.** Repeated the exact recipe -
  25,000 more on-policy positions, this time generated from `sinsan-v2-56x7`'s policy instead of
  v1's, combined into a 100,000-position mixture, same 56x7 architecture (`sinsan-v3-56x7`).
  Trained cleanly (94.3 min, 100% legal-move rate) but showed **no measurable improvement**: v3 vs
  v2 head-to-head came back dead even (1-1-18, 50.0%), and v3 vs Fairy-Stockfish at UCI_Elo=800
  produced the *identical* win/loss/draw counts as v2 (0-2-8, 40.0%) - two independent
  confirmations of the same null result. The honest read: iteration 1's real gain (random-only to
  on-policy data) didn't repeat when the same recipe was applied a second time - the easy
  improvement was already captured, and pushing the real strength ceiling further (currently ~Elo
  600-700 against Fairy-Stockfish) would need a different lever, not another identical round.
- **Not yet started:** search-guided self-play (rejected on time grounds - too slow to generate at
  scale, see `docs/BENCHMARK_PLAN.md`), value calibration, and the `/analysis`, `/research`,
  `/about` routes. See `docs/BENCHMARK_PLAN.md` for what's measured vs. planned.

## Requirements

Linux Mint (official dev environment - see `docs/RESEARCH.md` §3.1), Node.js 24+ (native
TypeScript execution, no build step needed), Python 3.12 + `uv` (`training/pyproject.toml`
installs PyTorch CPU).

## Running the rules engine tests

```sh
npm install
npm run test:rules          # node's built-in test runner, 45 tests
npm run typecheck --workspace=@sinsan/rules
```

## Running the playable board

```sh
npm install
npm run dev --workspace=@sinsan/web   # http://localhost:5173
```

## Building and running the teacher engine

```sh
./scripts/build-teacher.sh            # clones + builds pinned Fairy-Stockfish, ~1-2 min
python3 training/teacher/adapter.py   # self-check: labels the startpos over UCI
```

## Running the smoke pipeline (generate -> label -> train -> export)

```sh
node training/generate/self-play.ts 512        # ~1s, writes training/datasets/smoke-positions.jsonl
cd training && uv sync                          # installs PyTorch CPU (first run only)
uv run python generate/label_dataset.py         # ~45s for 512 positions (real teacher throughput)
uv run python train/train.py                    # 5 epochs, writes model/checkpoints/tiny-smoke.pt
uv run python export/export.py                  # writes public/model/sinsan-smoke-v0.{bin,json}
cd .. && node --test 'tests/model/**/*.test.ts' # includes the TS<->PyTorch numerical parity check
```

Then pick "Sinsan (policy only, no search)" or "Sinsan (16-visit search)" from the AI dropdown on
the playable board to see real Worker-hosted inference (with or without PUCT/MCTS) driving moves.

## Running the Phase 5 baseline pipeline (48x6 Main Candidate, ~50K positions)

Every script above now takes `--dataset`/`--channels`/`--blocks`/`--epochs`/`--batch-size`/etc.
instead of hardcoding the smoke tier, so the same code produces either model.
`training/datasets/baseline-positions.jsonl` (50,000 self-play positions) is already generated -
labeling and training are long-running and left for you to kick off:

```sh
cd training
uv run python generate/label_dataset.py --input baseline-positions.jsonl --output baseline-labeled.jsonl
# ~70-90 min (real measured throughput ~11.5 pos/sec) - resumable if interrupted (re-run the same
# command; it skips positions already in the output file). Prints an ETA as it goes.

uv run python train/train.py --dataset baseline-labeled.jsonl --channels 48 --blocks 6 \
  --epochs 20 --batch-size 256 --checkpoint-out baseline-48x6.pt
# ~55-60 min extrapolated from measured per-step timing (docs/BENCHMARK_PLAN.md) - watch the
# printed train/val loss each epoch for overfitting, not just that it runs.

uv run python export/export.py --checkpoint baseline-48x6.pt --channels 48 --blocks 6 \
  --model-name sinsan-baseline-v0 --training-run phase5-baseline

uv run python export/dump_parity_fixture.py --checkpoint baseline-48x6.pt --dataset baseline-labeled.jsonl \
  --channels 48 --blocks 6 --model-name sinsan-baseline-v0

cd .. && SINSAN_MODEL_NAME=sinsan-baseline-v0 node --test 'tests/model/model-runtime-parity.test.ts'
```

The playable board's AI dropdown offers all three model variants (smoke, baseline, v2), each with
policy-only and 16/64-visit search - `sinsan-v2-56x7` at 16-visit search is the default selection.

## Running an on-policy self-play iteration (Phase 6)

```sh
node training/generate/self-play.ts 25000 policy-v1-positions.jsonl \
  --player policy:sinsan-baseline-v0 --temperature 1.0   # slow - inference-bound, ~2-3hr for 25K
cd training && uv run python generate/label_dataset.py --input policy-v1-positions.jsonl --output policy-v1-labeled.jsonl
python3 generate/combine_datasets.py --input baseline-labeled.jsonl policy-v1-labeled.jsonl --output combined-v2-labeled.jsonl
uv run python train/train.py --dataset combined-v2-labeled.jsonl --channels 56 --blocks 7 \
  --epochs 20 --batch-size 256 --checkpoint-out baseline-v2-56x7.pt
uv run python export/export.py --checkpoint baseline-v2-56x7.pt --channels 56 --blocks 7 \
  --model-name sinsan-v2-56x7 --training-run phase6-onpolicy-v1
uv run python export/dump_parity_fixture.py --checkpoint baseline-v2-56x7.pt --dataset combined-v2-labeled.jsonl \
  --channels 56 --blocks 7 --model-name sinsan-v2-56x7
cd .. && SINSAN_MODEL_NAME=sinsan-v2-56x7 node --test 'tests/model/**/*.test.ts'
node apps/arena/src/run.ts --a "policy:sinsan-v2-56x7" --b "policy:sinsan-baseline-v0" --pairs 20
```

`--player policy:<model>` uses temperature-sampling (not greedy argmax) over the model's own
policy logits so games stay diverse - an all-greedy self-play policy collapses to a handful of
repeated lines, which is a weak training set regardless of how good the policy itself is.
Search-guided self-play was considered and rejected for this scale: a single 16-visit search call
measures ~10.5s for the baseline model (`docs/BENCHMARK_PLAN.md`), making even a modest dataset
take many hours to generate: policy-only inference is ~15-30x faster and is what actually makes an
on-policy dataset tractable at 10K+ positions in one session.
`training/generate/combine_datasets.py` offsets `game_id` per input file so `train.py`'s
`game_id % 10` split-bucket logic never coincidentally merges unrelated games from different
sources just because they reused small `game_id` numbers independently.

## Arena (paired matches between two players)

```sh
node apps/arena/src/run.ts --a "policy:sinsan-baseline-v0" --b random --pairs 20
node apps/arena/src/run.ts --a "search:sinsan-baseline-v0:16" --b random --pairs 10  # slower - real per-move search
node apps/arena/src/run.ts --a "search:sinsan-baseline-v0:16" --b "policy:sinsan-baseline-v0" --pairs 10
node apps/arena/src/run.ts --a "policy:sinsan-v2-56x7" --b "policy:sinsan-baseline-v0" --pairs 20  # v2 vs v1
node apps/arena/src/run.ts --a "search:sinsan-v2-56x7:16" --b "stockfish:1500:100" --pairs 10  # vs real engine
```

`apps/arena` (docs/ARCHITECTURE.md) calls `SinsanModel.infer()` directly - no Worker/fetch, since
that boundary is specifically about the shipped browser artifact and this tool never ships to the
browser. Each pair plays one random formation-combo opening twice with colors reversed, to cancel
first-move/formation bias (docs/BENCHMARK_PLAN.md Section 18.3). Player specs: `random`,
`policy:<model-name>` (greedy policy, no search), `search:<model-name>:<visits>` (PUCT),
`stockfish:<elo|full>[:<movetimeMs>]` (drives a real Fairy-Stockfish subprocess over UCI, the same
protocol `training/teacher/adapter.py` uses to label data - `elo` sets `UCI_LimitStrength`+
`UCI_Elo`, `full` runs it unrestricted, `movetimeMs` defaults to 100). Reports win/loss/draw counts
and a rough Elo-difference estimate - explicitly labeled small-sample/directional, not a calibrated
rating (that needs many more games than a few dozen). **On the `stockfish` player specifically:**
`UCI_Elo` is a chess-calibrated Stockfish mechanism - whether a given value means the same real
strength in the Janggi variant is unverified, and there is no known Elo-to-Korean-dan conversion
this project has access to, so results are reported as "vs Fairy-Stockfish at UCI_Elo=N," never
translated into a dan-rank or absolute-strength claim (see `docs/BENCHMARK_PLAN.md`).

## Differential testing (rules engine vs. pyffish)

```sh
node tests/differential/generate-cases.ts 10000   # random legal positions, all 4 rule profiles
cd training && uv run python ../tests/differential/compare.py
```

Mismatches (if any) are saved to `artifacts/rule-mismatches/` with the FEN, variant, and the
exact move-set difference - currently empty (0 mismatches at 10,000 positions).

## Benchmarking model-runtime inference

```sh
node benchmarks/model-inference.ts          # one-time cost vs. steady-state per-call latency
node benchmarks/model-inference-profile.ts  # isolates a single conv2d call's cost
```

## Checking your host profile

```sh
./scripts/check-host.sh
```

Reports actual measured CPU/RAM/toolchain/thermal state - re-run before any benchmark or training
run rather than trusting a prior report, since available RAM and thermal headroom drift with what
else is running.

## Project layout

See `docs/ARCHITECTURE.md` for the full layering rationale. Top-level: `packages/` (browser-native
TS: rules, action-space, model-runtime, search all implemented), `training/` (Python + a few TS
generation scripts: teacher, generate, model, train, export all implemented for the smoke tier;
on-policy refinement not yet), `apps/` (web: playable board implemented; arena not yet), `tests/`
(unit, fixtures, model, differential all implemented; browser/Playwright not yet), `benchmarks/`
(model inference latency), `assets/`, `docs/`, `scripts/`, `licenses/`.

## License

Sinsan's own code is AGPL-3.0-or-later (`LICENSE`). Third-party code and assets retain their
original licenses - see `THIRD_PARTY_NOTICES.md` and `docs/licenses.md` for the full audit,
including one still-open item (an unresolved license on the only available dedicated Janggi NNUE
net) that blocks public dataset/model publication until resolved.
