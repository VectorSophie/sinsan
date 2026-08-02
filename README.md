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
  real export. See `docs/BENCHMARK_PLAN.md` Section 24.4 for the full real-measured numbers. Not
  yet wired into the playable board's AI dropdown (still points at the Phase 4 smoke model) or
  arena-evaluated for playing strength.
- **Not yet started:** on-policy refinement (self-play still uses random moves, not teacher- or
  model-guided), value calibration, and the `/analysis`, `/arena`, `/research`, `/about` routes.
  See `docs/BENCHMARK_PLAN.md` for what's measured vs. planned.

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

The playable board's AI dropdown already offers both variants (smoke and baseline, policy-only
and 16-visit search) - `sinsan-baseline-v0` is the default selection.

## Arena (paired matches between two players)

```sh
node apps/arena/src/run.ts --a "policy:sinsan-baseline-v0" --b random --pairs 20
node apps/arena/src/run.ts --a "search:sinsan-baseline-v0:16" --b random --pairs 10  # slower - real per-move search
node apps/arena/src/run.ts --a "search:sinsan-baseline-v0:16" --b "policy:sinsan-baseline-v0" --pairs 10
```

`apps/arena` (docs/ARCHITECTURE.md) calls `SinsanModel.infer()` directly - no Worker/fetch, since
that boundary is specifically about the shipped browser artifact and this tool never ships to the
browser. Each pair plays one random formation-combo opening twice with colors reversed, to cancel
first-move/formation bias (docs/BENCHMARK_PLAN.md Section 18.3). Player specs: `random`,
`policy:<model-name>` (greedy policy, no search), `search:<model-name>:<visits>` (PUCT). Reports
win/loss/draw counts and a rough Elo-difference estimate - explicitly labeled small-sample/
directional, not a calibrated rating (that needs many more games than a few dozen).

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
