# Model Design

## Action encoding: 90 origins × 60 templates + 1 pass = 5,401

Adopted as specified (Section 9), and independently corroborated: two
unrelated AlphaZero-style Janggi engines found during research
(`alphazero_janggi`, `AlphaJanggi`) converged on their own per-origin-square
move-template encodings in the same range (~58-60 templates × 90 squares ≈
5,225-5,400 actions) without knowledge of this spec. That's real external
validation the design isn't arbitrary, though it isn't proof it's optimal —
still to be tested: encode/decode round-trip uniqueness, collision-freedom,
and horizontal-reflection correctness (`packages/action-space` tests,
Phase 1).

The 60-template breakdown (36 orthogonal + 8 palace-diagonal + 8 horse + 8
elephant) maps directly onto a policy head shaped `[60 channels, 10, 9]`:
spatial position encodes the origin square, channel encodes the template —
so the policy head is a single 60-output-channel conv, not a separate
per-origin lookup structure. Pass gets one dedicated scalar logit outside
this tensor.

## Network candidates and a first parameter-count sanity check

Before writing any training code, it's worth confirming the 500KiB budget
is achievable at all with the proposed channel/block counts — arithmetic,
not yet an empirical measurement, but a legitimate first checkpoint in the
research→hypothesis→measurement loop. Assumptions below: 24 input planes
(provisional — see next section), 3×3 convs throughout the tower, a 60-
channel 3×3 policy conv + pass logit, and an 8-channel 1×1 value conv → 32-
hidden linear → scalar. INT8 = 1 byte/weight; biases and per-channel scales
stay FP32 and are small in aggregate (a few KB) at every size below.

| Candidate | Channels × blocks | Tower params | Total params | Est. INT8 weight bytes | Fits 480KiB weight budget? |
|---|---|---|---|---|---|
| Tiny baseline | 32 × 4 | ~74K | ~110K | ~108KB | Yes, comfortably (smoke-test tier) |
| Main candidate | 48 × 6 | ~249K | ~309K | ~302KB | Yes (spec's own estimate: 250-350KB — consistent) |
| Stretch candidate | 56 × 6 | ~339K | ~405K | ~396KB | Yes, but only ~85KB headroom left for manifest/scales/policy-value heads |
| (checked, not proposed) 64 × 6 | — | ~443K | ~515K | ~503KB | **No** — exceeds the 500KiB *total* package budget on tower weights alone, before manifest/heads/scales |

This confirms the spec's own caution in Section 11.1 ("do not assume 64
channels will fit") with an actual number rather than taking it on faith:
64×6 overruns the budget by itself. 56×6 is plausible but leaves little
room for error — its real byte count must be measured from an actual
export (Phase 5), not assumed from this table, since real per-channel scale
counts, manifest JSON size, and checksum overhead aren't fully accounted
for here (this table already includes generous slack estimates for those,
but "generous estimate" is not "measured").

**Decision:** proceed with the Main Candidate (48×6) as the primary target,
Tiny Baseline (32×4) for the Phase 4 smoke pipeline, and treat 56×6 as a
Phase 5 stretch contingent on a measured export coming in under budget.

## Quantization

Adopted from the verified Moka pattern (`REFERENCES.md`): symmetric,
per-output-channel INT8, weight-only. `scale = max(abs(output_channel_
weights)) / 127`; activations are never quantized — the Worker dequantizes
to `Float32Array` once at load and runs float32 arithmetic thereafter. If
BatchNorm is used during training it is folded into the preceding conv
before export (Section 11), so no separate BatchNorm parameters ever reach
the browser artifact.

INT4 (the stretch option mentioned in Section 14) is explicitly **not**
adopted as a default: research found no precedent for INT4 weight-only PTQ
on a CNN this size in a browser context, only general (LLM/vision-centric)
findings that INT4-without-QAT degradation is "non-negligible" for compact
models. If pursued, it must be an isolated, measured experiment (real
arena strength comparison, not just lower loss) before being considered for
the shipped artifact — a mixed-precision variant (INT4 tower, INT8 boundary
layers) is the more plausible version of this idea, per `RESEARCH.md`'s
differentiator ranking, but is unproven and not committed.

## Input planes (provisional, ~20-32 target)

Core 14: current player's {chariot, cannon, horse, elephant, guard, general,
soldier} + opponent's same seven.

**Implemented deviation for the Phase 4 smoke model:** Section 10's
*preferred* default is a canonicalized (always-my-perspective) board, but
that requires a matching 180°-rotation transform for the action space
(mirroring `packages/action-space`'s existing horizontal reflection, but
unimplemented and untested) — real added scope and a real place to
introduce an input/label misalignment bug for a milestone whose only job is
proving the pipeline works end to end. The smoke model
(`training/model/network.py`, `packages/model-runtime/src/features.ts`)
instead uses an **absolute** encoding — fixed Cho/Han planes regardless of
whose turn it is, plus one explicit side-to-move plane (15 planes total) —
since both the input and `packages/action-space`'s action ids already share
the same absolute board coordinates, with zero risk of desync. This was
verified cross-language via `tests/model/features-parity.test.ts` (exact
match against real PyTorch-produced planes), not just asserted. Real
canonicalization remains the Phase 5 default to test, per the ablation plan
below — this is a disclosed, deliberate smoke-tier simplification, not a
silent scope cut.

Candidates under consideration, each to be justified or dropped by ablation
in Phase 5, not added by default to "round out" a target count:

| Feature | Why it might help | Already derivable by rules engine? | Perspective-dependent? | Symmetry-safe? |
|---|---|---|---|---|
| Previous 2-4 move origin/destination planes | Tactical pattern continuity (e.g. mid-cannon-maneuver state) | Yes (history) | Yes | Yes, if reflected consistently |
| Check state | Immediate tactical salience | Yes | No | Yes |
| Repetition proximity / no-capture ply count (normalized) | Helps value head anticipate forced results | Yes | No | Yes |
| Bikjang state | Directly rules-relevant, cheap to compute | Yes | No | Yes |
| Palace / palace-diagonal location planes | Fixed geometric prior, may reduce burden on conv receptive field | Yes (static) | No | Yes (with care - palace is asymmetric between Cho/Han sides) |
| Material-score difference | Coarse value signal | Yes | Yes | Yes |
| Move-count / phase | Opening vs. endgame behavior shift | Yes | No | Yes |

Explicitly **not** planned: a "legal move mask as input plane" — the spec is
clear the model must not be asked to learn legality when
`packages/action-space` can supply a deterministic mask at inference time;
adding it as a training input would blur that boundary rather than reinforce
it.

## Value head

Current-player-perspective scalar, `tanh` output in `[-1, 1]`. This
convention must be identical across training, export, browser runtime,
arena, and the eventual model card — Moka's own experience (a value head
that a search algorithm found "not a useful evaluator" in one internal
probe) is a reminder to validate this specifically under PUCT once search
exists (Phase 6), not just via training-loss metrics.

## Differentiator experiments (ranked; see `RESEARCH.md` for sourcing)

1. Tau-style cross-game soft-policy aggregation (`neural-chess`) — planned
   experiment, Phase 6.
2. Codebook/lookup hybrid leaf evaluation (Rapfi) — stretch idea, unproven
   for Janggi.
3. Mixed-precision INT4/INT8 — stretch idea, requires real arena-strength
   measurement before adoption.
4. Factorized (non-dense) policy head exploiting Janggi's smaller move set
   — our own speculative idea, no external source, candidate only.

None of these four change the Phase 1-5 baseline plan; they are things to
measure against that baseline once it exists.
