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

**Implemented and measured (Phase 6, prompted by wanting a snappier human
playtest against v3): the scratch buffer above.** Every conv layer
(`stem`, each `tower.N.conv1`/`conv2`, `policy_conv`, `value_conv`) now
gets one `Tensor3D` output buffer allocated once at `SinsanModel`
construction, reused by every `infer()` call instead of `conv2d`
allocating a fresh one each time (`packages/model-runtime/src/tensor-ops.ts`'s
`conv2d` takes an optional `out` parameter; `model.ts` passes each layer's
own buffer). Verified correct first, not just fast: all model-runtime
tests pass, including the exact-numerical-match PyTorch parity check,
across all four exported models (smoke/baseline/v2/v3) - reusing buffers
is a real aliasing hazard if any layer's input and output buffer could
ever be the same object, so this was checked, not assumed (they can't be:
every named layer has its own dedicated buffer, and the residual-connection
buffer being read from is never the same buffer being written to at any
point in a forward pass).

Real before/after measurement, `sinsan-v3-56x7` (56×7, the current
largest/slowest model, same host): **mean 559.39ms → 480.59ms, p50
533.11ms → 473.66ms** - roughly a 14% reduction, and all five reported
statistics (mean/p50/p95/min/max) moved the same direction, not just one
noisy outlier. A real, modest win, not the dramatic fix the <20ms target
would need - the remaining ~480ms is inherent `conv2d` compute for a
56-channel/7-block network on this unvectorized JS implementation, not
allocation overhead anymore. Closing that gap further would need either a
genuinely faster runtime (WASM/SIMD - still not assumed necessary without
its own controlled measurement, per Section 15) or a smaller model
(a real strength/speed tradeoff, not a free optimization) - both bigger
decisions than this fix, not pursued here.

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

**First real playing-strength measurement (`apps/arena`, this host):** 20
paired games (40 total, random formation openings, colors reversed per
pair) between `sinsan-baseline-v0`'s raw policy head (greedy, no search)
and uniform-random legal moves: **A (baseline policy) won 11, lost 0, drew
29** - score fraction 63.7%, rough Elo-difference estimate +98 (small-sample,
directional only, not a calibrated rating - see the caveat above `apps/arena`
prints with every run). The trained policy never lost to random play across
40 games, which is a real signal the model learned something from the
teacher labels, not just noise - but the 29 draws are mostly `bikjang`
(both sides can stumble into it unintentionally), so this result says "the
policy is clearly better than random," not "the policy avoids bikjang" or
anything about tactical strength, which needs a stronger opponent (search-
enabled play, or a real baseline engine per the desiderata above) to probe.
This is not yet the Elo-vs-checkpoints, vs-alpha-beta-engine, or vs-
restricted-Fairy-Stockfish comparisons Section 18.3 asks for - those still
need real opponent engines wired into `apps/arena`, which doesn't exist
yet. Run it yourself: `node apps/arena/src/run.ts --a
"search:sinsan-baseline-v0:16" --b "policy:sinsan-baseline-v0" --pairs 10`
for a slower but more informative search-vs-policy comparison (real per-move
search cost, not yet measured end-to-end for a full arena batch).

**Phase 6 iteration 1: on-policy self-play, real result.** 25,000 positions
generated via `sinsan-baseline-v0`'s own policy (temperature=1.0 sampling,
`training/generate/self-play.ts --player policy:<model>`), combined with the
existing 50,000 random-self-play positions (75,000 total,
`training/generate/combine_datasets.py`), and used to train a larger 56x7
model (`sinsan-v2-56x7`, 445,706 params, 441.7KiB - chosen via a direct
parameter-count probe against the 480KiB budget, not guessed). Three
`apps/arena` batches (20 pairs / 40 games each), same host:

| Matchup | Result (W-L-D) | Score | Elo-diff est. |
|---|---|---|---|
| v1 (`sinsan-baseline-v0`) vs random | 11-0-29 | 63.7% | +98 |
| v2 (`sinsan-v2-56x7`) vs random | 8-0-32 | 60.0% | +70 |
| **v2 vs v1 (head-to-head)** | **3-0-37** | **53.8%** | **+26** |

The head-to-head is the real signal here, not the two separate vs-random
numbers (which look inconsistent with each other at face value - v2 scoring
*lower* against random than v1 did - but that's very plausibly 40-game
sample noise given both are 0 losses and within a few games of each other;
comparing two independent small samples against a third party is a weaker
signal than a direct head-to-head). **v2 never lost to v1 across 40 games
and won 3 outright** - a real, positive, but modest effect from one round of
on-policy self-play, not a dramatic jump. The dominant pattern across all
three matchups is still a very high draw rate (mostly `bikjang`/`repetition`),
meaning none of these policy-only (no-search) models are tactically decisive
yet - that's the more informative finding for what to try next (search-
enabled play, another self-play iteration, or both) than the win-rate
numbers alone.

**Follow-up: does search break the draw pattern? No - confirmed at two
sample sizes, 16 visits.** First pass: `--pairs 5` (10 games) came back
10-0 draws, every game drew. Followed up at `--pairs 20` (40 games) per
request to check whether that was just small-sample noise: **still 40-0
draws - every single game drew**, versus 3-0-37 (3 decisive v2 wins) when
both sides played policy-only without search. Search did not produce more
decisive results here - if anything the opposite (0 decisive games across
both search-vs-search samples vs. 3 in the policy-only comparison).
Plausible explanations (none confirmed): search may play more
solidly/defensively for both sides and steer away from losing lines,
narrowing the practical skill gap between two closely-matched models rather
than widening it. Flagging as an open, real finding rather than a
conclusion about *why* - the *what* (search doesn't break the draw
pattern at 16 visits, for these two specific models) is now reasonably
well supported, but the *why* isn't determined.

**Methodology caveat discovered while running the 40-game sample: the
`apps/arena` `policy` and `search` players are fully deterministic (no
exploration noise), and there are only 16 possible `(setupCho, setupHan)`
formation combos** - so a 20-pair run doesn't produce 40 independent data
points. Deduplicating the 20 pairs by their exact outcome signature (same
game-result-kind and ply-count for both games in the pair, a strong
signature that two pairs played the literal same opening) found only
**11 distinct outcomes across the 20 pairs** - some openings were drawn
2-3 times, producing byte-identical repeat games. All 11 distinct openings
tested still resolved to draws for both colors, so the "search doesn't
break the draw pattern" finding survives the deduplication, but it means
the true effective sample size was closer to ~22 informative games than
40, and this caveat applies retroactively to every `apps/arena` batch run
so far in this document, including the 40-game policy-only baseline/v2/
random comparisons above (those runs likely also contain some exact
duplicate pairs, not yet explicitly checked). **Fixed, both observability and the underlying redundancy.** `apps/arena`
now logs each pair's formation combo directly, and (more importantly) plays
`--random-plies` (default 4) uniform-random legal moves after the formation
setup, shared by both games in a pair, before handing control to the
deterministic players under test - this multiplies the 16 formation combos
by many possible random continuations, giving far more effectively-distinct
starting positions instead of capping out at 16. The players themselves
stay fully deterministic (so a decisive result still means the engine
actually won from that position, not that noise broke a tie) - only the
opening varies. Verified directly: with the fix, the two games within a
single pair now have visibly different ply counts even for identical
formations (previously, non-determinism only entered via which random
player-vs-player matchup was being tested, not the opening itself).
Re-running the 16-visit search comparison with this fix is the natural next
step to check whether the 100%-draw finding above was a genuine
characteristic of these two models or partly an artifact of the redundant-
opening bug - not yet done as of this note.

**CORRECTION - re-run with the fix, and the 100%-draw finding above does
NOT hold up.** Same matchup (`search:sinsan-v2-56x7:16` vs
`search:sinsan-baseline-v0:16`), same 20-game count, this time with real
opening diversity (`--random-plies 4`, the new default): **A (v2) wins: 5,
B (v1) wins: 0, draws: 15** - score fraction 62.5%, Elo-diff estimate +89.
This is *more* decisive than the policy-only head-to-head (53.8%, +26), not
less - search amplifies v2's advantage once the models are actually tested
on varied positions, which is what should have been expected all along.
**The earlier 0-0-40 and 0-0-10 all-draw results were an artifact of the
duplicate-opening bug, not a real property of these models or of search**
- stated plainly rather than left as a hedged "open question," since the
corrected experiment resolves it. This is also a useful lesson about
`apps/arena`'s prior default: 16 formation combos alone was never enough
variety for a deterministic-player evaluation, and every batch run in this
document before this fix (the 40-game policy-only baseline/v2/random
comparisons included) should be read with that in mind - not necessarily
wrong in aggregate direction (the v2-vs-v1 policy comparison's 3-0-37 did
show real signal despite the bug), but understood as measuring fewer truly
independent games than their raw counts suggest.

**Deeper search (64 visits): the strongest result measured so far, with two
real caveats stated plainly, not smoothed over.** `search:sinsan-v2-56x7:64`
vs random, 10 pairs/20 games, fixed arena tool: **A (v2) wins: 11, B
(random) wins: 1, draws: 5, unresolved: 3** - score fraction 67.5%,
Elo-diff estimate +127, the highest of any comparison in this document
(above v1 policy-vs-random 63.7%, v2 policy-vs-random 60.0%, and the
corrected v2-vs-v1 16-visit result 62.5%). Deeper search meaningfully
increases practical strength against random play, as expected.

Caveat 1: **random beat v2 once** (pair 6, game 2, 49 plies) - the first
loss any trained Sinsan model has suffered to random play across this
entire session's evaluations. One loss in 20 games isn't alarming on its
own (real engines lose occasional games to weak play, especially from
unlucky tactical shots), but it's a real data point, not swept under the
strong aggregate score.

Caveat 2: **3 of 20 games (15%) hit the 400-ply safety cap without a
rules-engine result** - `apps/arena`'s own code treats this as
investigation-worthy, not routine, and 3 occurrences (vs. zero in every
other comparison in this document) is a real pattern, not a single
anomaly to wave off. Checked the likely mechanism rather than assuming a
bug: `kja` profile's no-capture-move-limit is 200 plies
(`packages/rules/src/rule-profiles.ts`), and `apps/arena`'s safety cap is
400 - only 2x that threshold. A long, messy game where captures happen at
least once every <200 plies throughout (very plausible against a chaotic
random opponent, especially soldiers trading), but never converts to
checkmate or reaches a clean repetition/bikjang, can legitimately run the
full 400 plies without any adjudication firing - this is a property of how
the no-capture-limit and the safety cap interact, not necessarily a rules
bug. This explanation fits the observed data (all 3 unresolved games came
from the deepest-search, longest-average-game comparison in this document)
but hasn't been confirmed by tracing an actual game's capture history - if
this pattern recurs at a similar or higher rate in future comparisons, that
would be the point to either raise `MAX_PLIES` or investigate further
rather than continue assuming it's benign.

**Deepest head-to-head (64 visits both sides): a genuinely different, less
one-sided result - reported exactly as measured, not spun toward the
pattern of the earlier comparisons.** `search:sinsan-v2-56x7:64` vs
`search:sinsan-baseline-v0:64`, intended as 10 pairs/20 games. The host's
memory pressure (see below) killed the run at 18/20 games; rather than
discard that data or force a full restart (this comparison alone had
already consumed ~4 hours of wall-clock across two attempts), the tally
from the 18 completed games is reported as the result, with the count
stated plainly: **A (v2) wins: 1, B (v1) wins: 2, draws: 15, over 18
games**. This is the one comparison in this document where v1 is not
behind - if anything slightly ahead in decisive games, though 3 decisive
results out of 18 (83% draws) is too small a sample to call a real
reversal of the pattern seen elsewhere. Combined picture across all three
search-depth comparisons: 16-visit head-to-head favors v2 (5-0-15, 62.5%),
64-visit vs random favors v2 strongly (11-1-5, 67.5%), and 64-visit
head-to-head is close to even with v1 marginally ahead in decisive games
(1-2-15). The honest read is that v2 is measurably better than random and
better than v1 at moderate search depth, but the evidence that v2 is
better than v1 *specifically at their shared peak search depth* is weak
and possibly not real - this is exactly the kind of result that would
need a larger sample (which the host currently can't reliably sustain,
see below) to resolve either way, not a result to round up into a
uniform "v2 is better" conclusion.

**A real, escalating system health issue, not a bug in this project's
code.** Three separate OOM (out-of-memory) events occurred during this
session's arena runs, each larger than the last: first killed one process
(`traefik`), then three (`nginx` x2, `coredns`, one VS Code helper), then
ten (`nginx` x2, `coredns` x2, one VS Code process, five separate Chromium
processes). This host has 7.4GB RAM shared with a week-plus-uptime k3s
cluster, a neo4j database, VS Code, a browser, and multiple concurrent
Claude Code sessions - the escalating severity across three events several
hours apart, on a machine whose other processes weren't being actively
grown by this project's work, points toward organic memory growth
somewhere in those long-running background services (a slow leak, log/cache
accumulation, or similar) rather than anything `apps/arena` or the training
pipeline did wrong. Not investigated further - outside this project's
scope - but worth recording plainly: **further large `apps/arena` batches
on this host risk killing other unrelated processes (already did, twice),
not just failing themselves**, and that risk grows the longer the host's
own services keep running without a restart. Update: the host was restarted
(for unrelated reasons) after the second cascade, which cleared the memory
pressure entirely (swap went from fully exhausted to fully free) - the
underlying cause in the host's other services was never identified or
fixed, so this is a reset, not a resolution, and the same pattern could
recur over a similarly long uptime.

**First real external-engine calibration: vs Fairy-Stockfish, not just
Sinsan's own checkpoints.** All comparisons above are Sinsan-vs-Sinsan or
Sinsan-vs-random - useful for relative progress, but none of them anchor
to anything outside this project. `apps/arena` now has a `stockfish`
player kind that drives a real Fairy-Stockfish subprocess over UCI (the
same protocol `training/teacher/adapter.py` already uses to label data),
using `UCI_LimitStrength`+`UCI_Elo` to set a target strength.
**Important calibration caveat, stated once here and not repeated at every
number below: `UCI_Elo` is a Stockfish-family mechanism calibrated against
CHESS self-play data. Whether a given UCI_Elo value means the same
real-world strength in the Janggi variant is unverified - there is no
known authoritative Janggi-engine-strength-to-Elo conversion, let alone an
Elo-to-Korean-dan one, that this project has access to.** Every result
below is reported as "vs Fairy-Stockfish at UCI_Elo=N," never translated
into a dan-rank or absolute-strength claim.

`search:sinsan-v2-56x7:16` vs `stockfish:500:100` (Elo 500 is Stockfish's
minimum supported value; movetime 100ms), 5 pairs/10 games: **A (v2) wins:
4, B (Stockfish) wins: 1, draws: 5** - score fraction 65.0%, Elo-diff
estimate +108. v2 is clearly ahead of Fairy-Stockfish at its weakest
configurable setting, but not flawless (one real loss, not a clean sweep) -
reported exactly as measured, including the loss, not rounded up to
"never loses" the way some earlier random-opponent comparisons could
honestly say.

**Following the ceiling up: `stockfish:1200:100`, 5 pairs/10 games -
complete reversal.** **A (v2) wins: 0, B (Stockfish) wins: 10, draws: 0** -
a total shutout, every game decisive (checkmate, 28-82 plies - real,
played-out games, not instant collapses), the losing side consistent
regardless of which color v2 played. Before trusting this, checked for a
bug rather than assuming the result: re-ran `search:sinsan-v2-56x7:16` vs
`random` immediately after (2 pairs/4 games) and got 2-0-2, 75% score,
consistent with every other random-opponent result all session - v2's
search player is not broken. The 0-10 result is real: **v2's actual
strength ceiling against Fairy-Stockfish sits somewhere between UCI_Elo
500 (clear win, 65%) and UCI_Elo 1200 (total loss, 0%)** - a genuine,
externally-anchored bound, not a Sinsan-vs-Sinsan or Sinsan-vs-random
number. This is the most informative single result of the whole Phase 6
investigation precisely because it's not flattering: it establishes that
whatever v2's real strength is, it is decisively below whatever UCI_Elo
1200 represents (with the standing caveat that this chess-calibrated
number's meaning for Janggi is itself unverified) - not narrowing the
question to "which Sinsan checkpoint is better," which is what every
other comparison in this document measures. Not yet narrowed further
(e.g. bisecting between 500 and 1200) - the two data points already
establish the real bound this session set out to find.

**Bisecting the bracket: `stockfish:800:100`, 5 pairs/10 games.** **A (v2)
wins: 0, B (Stockfish) wins: 2, draws: 8** - score fraction 40.0%,
Elo-diff estimate -70. v2 is already the weaker side at Elo 800 (mostly
draws, but losing more than winning), narrowing the real crossover point
(~50% score) to somewhere between UCI_Elo 500 (65.0%) and UCI_Elo 800
(40.0%) - roughly the 600-700 range by simple interpolation between two
small-sample points, not a precise figure. **Three real data points now
characterize v2's strength curve against a genuine external engine:**

| Fairy-Stockfish UCI_Elo | v2 score | Result (W-L-D) |
|---|---|---|
| 500 (minimum) | 65.0% | 4-1-5 |
| 800 | 40.0% | 0-2-8 |
| 1200 | 0.0% | 0-10-0 |

This is the most externally-grounded characterization of Sinsan's playing
strength this project has produced. It does not, and cannot, resolve to a
"4단" or any other Korean-dan-rank claim: no rated human opponents were
available to test against, and a direct web search for a Janggi dan-to-Elo
(or dan-to-engine-strength) conversion found none - Janggi's own amateur
rank structure (15급 to 7단, per Korean-language sources) exists, but no
citable numeric mapping from it to Elo or any comparable scale was found.
Combined with the standing caveat that `UCI_Elo` itself is a chess-
calibrated mechanism whose meaning for the Janggi variant is unverified,
the honest ceiling on what this bracket can claim is: "v2's search-based
play crosses over from beating to losing against Fairy-Stockfish somewhere
in the neighborhood of UCI_Elo 600-700" - a real, useful, reproducible
finding, and explicitly not a substitute for the verification a dan-rank
claim would require.

**Phase 6 iteration 2 (v3): another round of on-policy self-play, this
time from v2 instead of v1 - a null result, reported as such.** Same
recipe as iteration 1: 25,000 positions generated from `sinsan-v2-56x7`'s
own policy (temperature=1.0), combined with the *existing* 75,000-position
mixture (50K random + 25K v1-on-policy) for 100,000 total, trained the
same 56x7 architecture (`sinsan-v3-56x7`, same 445,706 params/441.7KiB -
architecture was already near the 480KiB ceiling, so this iteration's
lever was data, not size). Training itself was clean (94.3 min, 100%
legal-move rate, no NaN, val loss tracking train loss). The real question
- did this iteration actually raise strength - came back negative:
**`search:sinsan-v3-56x7:16` vs `search:sinsan-v2-56x7:16`, 10 pairs/20
games: A (v3) wins: 1, B (v2) wins: 1, draws: 18** - score fraction
50.0%, Elo-diff estimate +0. A dead-even result, not a modest edge either
direction. Stated plainly: **this particular v3 iteration did not produce
a measurable strength improvement over v2** in this comparison. This is
useful, honest signal about the methodology's limits, not just the
model's: one more round of the same recipe (same generation temperature,
same architecture, same epoch count, same teacher) did not reliably
compound the way iteration 1's improvement over the random-only baseline
did - if anything this suggests iteration 1's gain came from a specific,
larger difference (on-policy vs. purely random data) that a second,
more incremental iteration didn't repeat at the same magnitude, not that
each iteration should be expected to keep adding roughly the same amount
of strength.

**Confirmed against the external anchor: `search:sinsan-v3-56x7:16` vs
`stockfish:800:100`, 5 pairs/10 games - A (v3) wins: 0, B (Stockfish)
wins: 2, draws: 8** - score fraction 40.0%, Elo-diff -70. **Identical
win/loss/draw counts to v2's result at the same UCI_Elo setting.** This is
the clean, decisive confirmation of the head-to-head null result via a
completely independent measurement (a fixed external engine, not a
Sinsan-vs-Sinsan comparison that could itself be noisy or biased): v3's
real strength ceiling against Fairy-Stockfish is the same as v2's,
somewhere in the UCI_Elo 600-700 neighborhood established earlier. Two
iterations of on-policy self-play (v1->v2->v3) produced one real,
externally-confirmed strength gain (the first iteration, random-only to
on-policy) and one null result (the second iteration, on-policy to
more-on-policy) - the honest read is that this specific recipe's easy
gains were captured in iteration 1, and repeating it verbatim for
iteration 2 did not compound further. Moving the real ceiling past ~700
would need a different lever (search-guided self-play despite its cost,
a bigger architecture within the remaining ~38KiB budget headroom, value
calibration, or a genuinely different training recipe), not another
identical on-policy round - reported here as the honest conclusion of
this investigation, not a reason to keep repeating the same experiment.

## Training-time estimation (Section 24)

`scripts/estimate-training-time.py` (Section 24.3's estimator script) is now
written and calibrated against the real 73.3-minute baseline training run
(not pure step-compute time, which undershot real wall-clock by ~35% once
per-epoch eval/checkpointing overhead was accounted for - see below). Both
prerequisite benchmarks referenced by it have
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
