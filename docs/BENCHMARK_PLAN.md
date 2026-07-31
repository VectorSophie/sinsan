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

## Training-time estimation (Section 24) — scoped for a later phase

Sections 24.1 (teacher throughput benchmark) and 24.2 (training-step
benchmark) require a built Fairy-Stockfish binary and an installed PyTorch
CPU environment respectively — both Phase 3+ activities, deliberately out
of this session's scope (Phase 0 research/docs/scaffold, per the session
plan agreed at the start of this work). `scripts/estimate-training-time.py`
and the underlying teacher/training benchmarks will be built and actually
run in that follow-up phase; this document records the plan and the
categories of estimate (measured / extrapolated / conservative / optimistic)
required by Section 24.3, not placeholder numbers standing in for real
measurements.

## What "done" looks like for this plan

- Every metric above has a named script or test file that produces it,
  referenced from `benchmarks/`.
- Every reported number in any future model card or benchmark report is
  traceable to a script run recorded with its git commit, config, and
  dataset hash — not restated from memory or a prior run without
  re-verification.
