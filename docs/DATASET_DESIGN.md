# Dataset Design

## Teacher labeling strategy

**Stage A** (bulk labeling): Fairy-Stockfish over UCI, MultiPV 8-16, fixed
node budget per position, deterministic seed/config where the engine
supports it. Produces a sparse top-K soft policy (candidate moves + scores/
WDL) rather than a single best-move label — legal moves outside the top-K
get an explicit small floor probability rather than silently zero, since a
policy target of exact zero for a legal-but-unsampled move is a stronger
claim than the teacher actually made. Mate scores are transformed to a
bounded value range explicitly documented in the export code (never an
ad hoc formula presented as ground truth — see Value Labels below).

**Stage B** (targeted deep labeling): applied only to positions flagged as
difficult — tactically sharp, misjudged by the current student checkpoint
(high student-confidence + wrong outcome), near repetition, near bikjang, or
in difficult endgames. Uses larger node budgets/MultiPV, and `searchmoves`
for legal-move-specific re-evaluation on selected hard subsets. Evaluating
every legal move deeply for every position is not attempted — confirmed
during research that this machine's realistic teacher throughput (to be
measured in Phase 3, see `BENCHMARK_PLAN.md`) makes that infeasible at
useful dataset scale; Stage B is a targeted refinement, not a blanket
policy.

## Value labels

Prefer the teacher's own WDL output where available over a hand-derived
centipawn→probability formula. Preserved fields per position: raw engine
score, raw WDL, search depth, search nodes, a calibrated value (fit against
held-out self-play outcomes, not assumed a priori), and final game outcome
where available. Calibration is measured and reported (Brier score, ECE),
never presented as exact without that measurement.

**Open licensing question carried from `docs/licenses.md`:** the only
available dedicated Janggi NNUE net has an unresolved license (predates
Fairy-Stockfish's 2026-CC0 cutoff). Whether teacher-generated *labels*
(positions + this net's evaluations) inherit any obligation from the net's
own unclear license is a real, unresolved question — not the same question
as bundling the net file itself, but also not automatically safe. This must
be resolved (net author clarification, legal review, or switching to a
confirmed-license evaluation source) **before** any public dataset release
to Hugging Face (Section 19.2), and is tracked as a publication blocker,
not silently assumed fine because "we're just generating our own data."

## Initial data mixture (Section 12.3, unchanged from spec pending measurement)

Teacher self-play/rollout 40%, opening/setup diversity 15%, Sinsan on-policy
rollout 25%, tactical/endgame/rule-edge positions 15%, licensed human games
≤5% (only if redistribution rights are explicit — none identified during
Phase 0 research; this bucket stays empty until a properly licensed source
is found, not filled with anything of unclear provenance).

## Dataset fields and format

Parquet or NPZ shards (not per-position JSON) with fields: `position`,
`features`, `legal_actions`, `teacher_actions`, `teacher_policy`,
`teacher_value_raw`, `teacher_value_calibrated`, `teacher_score`,
`teacher_wdl`, `search_nodes`, `search_depth`, `multipv`, `source`,
`game_id`, `ply`, `setup_cho`, `setup_han`, `result`, `teacher_version`,
`rule_profile`. `teacher_version` must record the exact Fairy-Stockfish
commit SHA pinned at generation time (see `REFERENCES.md`), since the
engine is under active untagged development and reproducibility depends on
knowing exactly which build produced a given label.

## Splitting

Game-level, not position-level — following Moka's verified `game_id % N`
bucket approach (`REFERENCES.md`), with fixed validation/test bucket
indices that never receive later on-policy additions. This is a hard
requirement, not a style preference: position-level random splitting would
leak near-duplicate positions from the same game across train/val/test and
overstate generalization.

## Symmetry augmentation

Horizontal reflection is the only symmetry planned initially, and only
after explicit verification (position/piece/move/action transform,
legal-move equivalence, value preservation, setup-label transformation) —
not assumed correct just because the palace's diagonal lines happen to look
bilaterally symmetric. Vertical reflection and 180° rotation are **not**
planned: Janggi is not symmetric front-to-back (Cho moves first, occupies a
fixed side, soldiers only move forward/sideways) so these transforms would
require reinterpreting which side is "forward" and are excluded rather than
attempted without proof, per Section 12.6's explicit warning.

## On-policy distillation loop (Section 13)

Adopts two concrete, sourced techniques instead of generic "weight mistakes
higher" (both flagged in `RESEARCH.md` as measured experiments, not
unconditional adoptions):

1. **Goldilocks-weighted sampling** (Moka): a Gaussian weighting function
   peaking where the student's own probability on the teacher's chosen move
   sits near a target difficulty threshold — concentrates relabeling effort
   on positions that are informative (neither trivially agreed-on nor
   wildly divergent) rather than just "biggest disagreement."
2. **Tau-style cross-game soft-policy aggregation** (`neural-chess`):
   aggregate positions across many games/visits into a frequency-tempered
   soft policy target rather than one-hot per-game labels.

Priority areas for on-policy sampling, per spec Section 13: student/teacher
top-move disagreement, large value disagreement, cannon tactics, horse/
elephant blocking, palace diagonal tactics, repetition and bikjang
situations, failed endgame conversion, and high-student-confidence errors.

Validation/test sets remain fixed and untouched by on-policy iteration,
consistent with the splitting policy above.
