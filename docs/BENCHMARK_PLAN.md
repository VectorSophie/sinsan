# Benchmark Plan

This is a plan document (Phase 0 deliverable) — it defines what will be
measured and how. It contains no invented numbers; actual results land here
or in `benchmarks/` only once the corresponding phase produces something
real to measure, and this session's scope (Phase 0, research + docs +
scaffold) does not include running teacher or training benchmarks yet.

## Model quality (Section 18.1)

Teacher top-1 agreement, teacher top-3 inclusion, policy cross-entropy,
value MAE, value Brier score, calibration error, legal-move rate, tactical-
suite success, endgame-conversion rate, rule-edge-case success. **Legal-move
rate must measure 100% after legal masking** — this is a pass/fail gate, not
a metric to trend over time; any non-100% result after masking indicates a
bug in `packages/action-space` or `packages/rules`, not an acceptable model
weakness.

## Browser performance (Section 18.2)

Raw model bytes, gzip/brotli bytes, runtime bundle bytes, cold/warm load,
init p50/p95, inference mean/p50/p95, 16/64/128-visit move latency, Worker
memory, main-thread long tasks, mobile viewport behavior.

**Calibration reference, not a target to match:** Moka's own measured
numbers (Apple Silicon, Chromium) were p50 init 10.1ms and p50 inference
8.8ms — both far under Sinsan's own targets (<200ms init, <20ms inference).
Our primary dev/benchmark machine is an i5-1035G7 (measured this session,
not the spec's assumed i5-1135G7), which will likely be slower than Apple
Silicon; Sinsan's targets already have generous margin built in, so this is
useful context, not pressure to match Moka's absolute numbers.

**First real measurement (Phase 4, Chromium, this host):** two live inference
calls through the actual Worker pipeline (fetch + digest verify + Worker
init + forward pass) measured 301ms and 360ms end-to-end for the smoke
model (32×4, 107K params) - well above the <20ms inference target.

**Follow-up investigation (`benchmarks/model-inference.ts`,
`benchmarks/model-inference-profile.ts`):** separating one-time cost from
steady-state per-call inference (Node, not browser, but same runtime
engine/code path) showed digest verification and model construction are
cheap (~5-30ms combined, one-time), and essentially all cost is in `conv2d`
- an isolated single 32-channel/3×3 conv at the tower's actual shape measured
~5.9ms per call, and the tower alone has 8 of these, roughly matching total
observed inference time.

**A pre-padding optimization (bounds-check-free inner loop) was tried and
reverted** after real measurement, not intuition: in isolation it looked
faster (~5.9ms → ~4.0ms per conv), but that comparison excluded the
padding array's allocation/copy cost from the timed region. Once paid on
every real call (11 times per forward pass, as it must be), full-model
steady-state inference got measurably *worse* (~57ms → ~80ms, reproduced
across 3 runs) - the allocation cost for these small tensors outweighed the
saved branches. Reverted to the simpler bounds-checked version. Repeated
full-model runs in this session ranged from ~57ms to ~104ms with no code
changes between them, most likely reflecting this shared dev machine's
variable background load rather than the algorithm itself - a reminder that
single-digit-run comparisons on a noisy shared host aren't a reliable basis
for further micro-optimization, and chasing tighter margins here without a
quieter, controlled benchmark environment (or moving the comparison into
the actual browser target) would mostly be measuring noise.

**What this means going forward:** the current ~60-100ms range is a real,
disclosed gap against the <20ms target, and the qualitative cause (naive,
unvectorized nested-loop conv2d, no loop-order or SIMD optimization) is
confirmed, not guessed. The concrete next step, if this is prioritized
before Phase 5's bigger model make it moot anyway, is a **reusable scratch
buffer** sized once at model construction (amortizing allocation across all
calls) rather than further loop-microarchitecture tweaks - allocation
overhead, not branch cost, was the actual lesson from this investigation.
Not assumed to need WASM/WebGPU without a controlled measurement, per
Section 15.

**JS vs. WASM vs. WebGPU ordering is a hypothesis to test, not settled
fact.** Research found no rigorous first-party benchmark for tiny-tensor
inference — only directional blog-level claims that WebGPU dispatch
overhead dominates below roughly 100M parameters. Phase 2 must produce our
own measurement on this project's actual network shape before committing to
one runtime over another, per Section 15.

## Playing strength (Section 18.3)

Elo between checkpoints, strength vs. a baseline alpha-beta engine, vs. a
handcrafted-evaluation engine, vs. restricted Fairy-Stockfish, paired
fixed-opening matches (identical opening + formation, reversed colors), and
human matches once verified players are available.

**Quantization-drift check, adopted from a real Moka finding:** their
experiment log showed INT8 export measurably changed which games were won
versus the float checkpoint on identical openings. Sinsan's arena
evaluation must run the actual exported/quantized artifact used in the
browser, not the float training checkpoint — treating them as
interchangeable would repeat a mistake Moka already documented and moved
past.

**Human-rank claims stay labeled as targets until verified**, per Section
18.3/24: never state "Sinsan is an amateur 3-dan player" — state "Target:
amateur 3-dan equivalent, estimated from controlled engine matches, not yet
verified against rated human players," and keep that qualifier until real
verified-human results exist.

## Training-time estimation (Section 24)

`scripts/estimate-training-time.py` (a dedicated estimator script per Section
24.3) has not been written yet, but both prerequisite benchmarks now have
real, host-measured numbers to build it from, superseding the spec's own
pre-measurement planning ranges (Section 24, "~3-10 hours" for a 50K
baseline):

**24.1 teacher throughput (measured, this host, Fairy-Stockfish pinned
commit, single process, nodes=8000, MultiPV=8):** 512 positions in 44.5s ≈
11.5 positions/sec. Extrapolated (not separately re-measured) to 50,000
positions: **~72 minutes**. This is an extrapolation from one config, not a
sweep across the 1×4/2×2/3×1/4×1 process×thread configurations Section 21
asks `benchmark-teacher.sh` to compare — that script doesn't exist yet, so
the "best config" question is still open; this is just what single-process
throughput happens to be.

**24.2 training step time (measured, this host, CPU-only PyTorch):**
32×4 Tiny Baseline at batch_size=32: 79-100ms/step (~320-400 samples/sec,
noisy across repeated runs on this shared machine — see the inference-
latency investigation above for why). 48×6 Main Candidate: ~204ms/step at
batch_size=32 (157 samples/sec), ~239 samples/sec at batch_size=256 (2-step
measurement only — noisy, but directionally confirms larger batches are
more efficient per-sample on this CPU, as Section 14 anticipated). Extrapolated
to a 50,000-position dataset (~40,000 after the game-level train/val/test
split) at batch_size=256, 20 epochs: **~55-60 minutes for the 48×6 model**,
notably faster than the spec's own pre-measurement guess for the whole
50K-baseline phase — because that guess bundled in teacher generation time
(the actual bottleneck, ~72 min) plus dataset conversion, not just training.

**Combined estimate for a 50K-position Phase 5 baseline run (self-play
generation + labeling + training + export), this host:** self-play
generation ~3 min (measured directly: 50,000 positions in 178s) + labeling
~72 min (extrapolated) + training ~55-60 min (extrapolated) + export
<1 min (measured) ≈ **~2.5 hours total**, well under the spec's own
conservative 3-10 hour range for this phase - a genuinely good sign, though
still an extrapolation from smaller measured runs, not a full end-to-end
timed execution of the 50K pipeline.

**24.4 Real end-to-end result (measured, not extrapolated) — the user ran
the full Phase 5 pipeline on this host:** the 48×6 training run itself took
**73.3 minutes** for 20 epochs / 3140 steps (mean=894.4ms/step, p50=877.2ms,
286 samples/sec, batch_size=256) — about 25% longer than the 55-60 min
extrapolation above, a reminder that a 2-step noisy sample underestimated
the real run. Final losses: train policy_loss=2.4133 value_loss=0.0049 |
val policy_loss=2.4368 value_loss=0.0085 | held-out test policy_loss=2.4413
value_loss=0.0060 — val tracks train closely with no sign of overfitting
across 20 epochs. **Legal-move rate on the validation split (masked
argmax): 100.0%** — the pass/fail gate from Section 18.1 holds. Export:
293,746 parameters, 298,800-byte weights blob (**291.8 KiB**, well inside
the 480 KiB budget, 208 KiB of headroom left). All 3
`tests/model/model-runtime-parity.test.ts` checks passed against this real
export (`SINSAN_MODEL_NAME=sinsan-baseline-v0`): digest verification,
manifest/budget check, and TS-vs-PyTorch numerical parity. Labeling and
self-play generation wall-clock weren't captured in this transcript (the
user ran them separately before pasting the training output), so the
labeling portion of the combined estimate above is still an extrapolation,
not yet a real measured number.

## What "done" looks like for this plan

- Every metric above has a named script or test file that produces it,
  referenced from `benchmarks/`.
- Every reported number in any future model card or benchmark report is
  traceable to a script run recorded with its git commit, config, and
  dataset hash — not restated from memory or a prior run without
  re-verification.
