# References

Per-source research ledger. Each entry reflects direct reading of the actual
repository (cloned locally during Phase 0 research, 2026-07-30) or a directly
fetched page — not recollection or assumption. "Access Issues" notes anything
that could not be verified this way.

---

## millionco/moka

**What can be learned:** The whole distillation methodology this project is
modeled on. Concretely: (1) symmetric per-output-channel INT8/INT4
weight-only quantization with a straight-through-estimator QAT variant; (2)
a from-scratch JS tensor runtime (no ONNX/WASM) that dequantizes once at
load and runs float32 inference in a Worker, with `crypto.subtle.digest`
SHA-256 verification of the weights buffer before use; (3) an on-policy
"DAgger-style" correction loop (`collect.py`) using a Gaussian "goldilocks"
sample-weighting function peaking where the student's own probability on the
teacher's move sits near a target difficulty (0.55) — more principled than
flat "weight mistakes higher"; (4) strict game-level train/val/test
splitting via `game_id % 10` buckets with fixed validation/test bucket
indices; (5) an arena harness reporting Brier score (predicted win
probability vs. realized outcome) per engine, not just win/loss.

**What should not be copied:** Anything Go-specific — the 82-action policy
head, board representation, KataGo protocol glue, and the specific
32ch×4-block (or the actually-shipped nested-bottleneck×12-block) network
shape. Janggi's action space, palace/diagonal rules, and piece set are
structurally different enough that the network topology must be redesigned,
not reskinned.

**How ideas apply to Sinsan:** Reuse the *pattern*, not the code: weight-only
INT8 quantization + dequant-at-load runtime, Worker + SHA-256 manifest
verification, game-level splitting, goldilocks-weighted on-policy
correction, and Brier-based arena reporting are all directly portable
methodology and are adopted in `MODEL_DESIGN.md`, `DATASET_DESIGN.md`, and
`BENCHMARK_PLAN.md`.

**Remaining uncertainties:** Their README describes a simple 4-block network,
but the actually-committed exported model is a different (nested-bottleneck,
12-block, global-pooling) architecture — a real documentation-drift example.
We don't know why the shipped model diverged from the described one.

**Alternative approaches:** Building a browser runtime on top of ONNX
Runtime Web instead of a hand-written tensor engine — rejected for the same
reason Moka rejected it: framework weight likely exceeds the savings for a
network this small (see `ARCHITECTURE.md`).

**Licensing implications:** MIT. No obligations beyond attribution if we
credit it, and we're not copying code, only method — attribution recorded
here and in `THIRD_PARTY_NOTICES.md` as a courtesy, not a legal requirement.

**Maintenance risk:** N/A — used as a one-time research reference, not a
running dependency.

**Decision:** Adopt the quantization/runtime/splitting/arena *methodology*.
Do not adopt the network shape, action space, or any Go-specific code.
Additionally adopt Moka's own lesson (documented in its `experiment-log.md`):
INT8 export measurably changes which games are won vs. the float checkpoint,
and top-move-agreement metrics overstate real strength — both become
explicit checks in `BENCHMARK_PLAN.md`, not assumptions.

---

## pbaer/neural-chess

Found during the "look for optimization potential" pass, not in the
original reference list — a near-exact chess analog of this project's goal
(strong teacher → tiny in-browser "hero" models, down to 116K params,
hand-written TS forward pass, optional PUCT).

**What can be learned:** A documented technique ("tau" recipe): aggregate
positions across many games and use soft policy histograms with
frequency-tempered sampling, instead of one-hot single-game labels — reported
+79 Elo at equal parameter count versus naive per-game labeling.

**What should not be copied:** Chess-specific network/action encoding and any
chess opening-book data.

**How it applies to Sinsan:** The tau recipe is domain-agnostic and directly
applicable to `DATASET_DESIGN.md`'s teacher-label aggregation step.

**Remaining uncertainties:** Whether the +79 Elo finding transfers to
Janggi's smaller, more tactical action space — untested until we measure it
ourselves.

**Alternative approaches:** Plain single-game one-hot policy targets (Moka's
default, and the original Sinsan spec's default) — simpler, but foregoes a
documented, credible improvement.

**Licensing implications:** MIT — same as Moka, method-only reuse.

**Maintenance risk:** N/A, research reference only.

**Decision:** Adopt the tau-style aggregation as a labeled experiment in
Phase 6 (on-policy refinement), compared against plain per-game labels
before committing to it — per the project's own research→measurement→decision
loop, not adopted uncritically.

---

## Fairy-Stockfish family (fairy-stockfish/Fairy-Stockfish,
## Fairy-Stockfish-NNUE, fairy-stockfish.wasm, variant-nnue-pytorch)

**What can be learned:** Janggi is a first-class hardcoded variant
(`src/variant.cpp`, gated behind a `LARGEBOARDS` build flag — not in
`variants.ini` like most variants), with four rule flavors: `janggi`
(bikjang + material counting), `janggitraditional` (bikjang = draw, no
material counting), `janggimodern` (no bikjang, material counting on,
4-fold repetition illegal, 100-move rule — tuned for the Kakao Janggi app),
`janggicasual` (neither). Bikjang and material-counting adjudication are
native engine features (`position.h::bikjang()`, `JANGGI_MATERIAL` path),
not something we need to reimplement for teacher-side verification. Pass is
encoded as the same square twice (`e2e2`), not `0000`. `pyffish` exposes
rules/move-generation/adjudication only — no search, eval, or MultiPV, so it
is useful for our differential rules testing but not for teacher labeling.

**What should not be copied:** No code is vendored from this project at all
— it is used exclusively as an external offline process (UCI subprocess),
never linked into or shipped with Sinsan's own code.

**How it applies to Sinsan:** `janggi` is the closest match to the spec's
intended default (`kja`) RuleProfile; the other three flavors are documented
as named alternative profiles in `RULES.md` rather than silently discarded,
since real ambiguity exists about which ruleset actual Korean tournament
play uses versus popular apps.

**Remaining uncertainties:** Whether `UCI_ShowWDL`'s win-rate model — the
same generic model used for chess — is meaningfully calibrated for Janggi.
Historical GitHub issues (#40, #186, #198, 2019-2020) documented real
repetition/bikjang rule bugs; current status not rechecked.

**Alternative approaches:** A from-scratch Janggi teacher engine — rejected
as far more work for no accuracy benefit; Fairy-Stockfish's Janggi support
is mature (variant present since ~2019) and its bikjang/material logic is
exactly what a hand-rolled implementation would need to reproduce.

**Licensing implications:** Engine is GPL-3.0 (verified: root `Copying.txt`).
Used only as a separate offline process communicating over UCI (arm's-length,
standard practice for engine tooling) — this does not impose GPL obligations
on Sinsan's own AGPL-3.0-or-later code, since nothing is linked or
distributed together. **The Janggi NNUE net itself
(`janggi-9991472750de.nnue`, released 2025-08-01, community-contributed,
claimed +1128 Elo) has an unresolved license**: Fairy-Stockfish's own NNUE
page grants CC0 only to nets dated 2026-or-later, and this net predates that
cutoff with no other license text found anywhere. **We will not bundle or
redistribute this net file** until either its author clarifies terms or we
find independent license text; local/private use as a research teacher is a
separate, lower-risk question we are not resolving here.

**Maintenance risk:** No fresh tagged release since 2021 (`fairy_sf_14_0_1_xq`);
development continues untagged on `master`. Recommend pinning a specific
commit (candidate: `c19b5f6c66894fdb0e88d0dd100e3885f744760a`, HEAD as of
2026-07-23) rather than the stale tag, re-verified at actual build time.

**Decision:** Use as the offline teacher engine, pinned to a specific
commit + SHA256 once built (Phase 3). Do not ship any part of it to the
browser. Treat the NNUE net's license as a blocking open question for
publication, not just a footnote.

---

## gbtami/pychess-variants

**What can be learned:** A production Janggi (9×10) implementation already
runs on top of chessgroundx, proving that library's fitness for this board
size in practice, not just in theory. Its `static/COPYING.md` demonstrates
the right pattern for per-asset license exceptions inside an otherwise
single-licensed repo — worth mirroring in our own `docs/licenses.md`.

**What should not be copied:** Server/matchmaking/account code — entirely
out of scope for a static, no-backend Sinsan.

**How it applies to Sinsan:** Source of the Janggi wood board images
(`JanggiWood.png`, `JanggiWoodDark.svg`, `JanggiBrown.svg`) and the
`janggihb` piece-set family.

**Remaining uncertainties:** None for the board images (blanket AGPLv3+
claim in `COPYING.md`, uncontradicted). For the piece sets: real, see below.

**Alternative approaches:** Commissioning original board/piece art —
rejected for Phase 0-1 scope as unnecessary effort given verified assets
already exist; may revisit for a polished v1 visual identity later.

**Licensing implications:** Root license AGPL-3.0-or-later (confirmed:
`LICENSE` file is full AGPLv3 text; README explicitly says "or any later
version"). **Important discrepancy found**: `COPYING.md`'s own summary table
claims all 8 `janggihb` piece subfolders are CC-BY-4.0 by Kadagaden, but the
actual per-folder license files disagree — `hanjablue` is really GFDL
(Wikimedia/Leo Ha), `intlblue` says "unknown," and four subfolders
(`hanjagreen`, `intlgreen`, `intlwooden`, `intlkakao`) have no license file
at all. Only `Ka_wooden` and `Ka_kakao` are independently confirmed CC-BY-4.0
(cross-checked against the upstream `Kadagaden/chess-pieces` repo). **We use
`Ka_wooden` specifically, not "the janggihb family."**

**Maintenance risk:** Low — actively maintained (frequent variant work).

**Decision:** Vendor the board images and the `Ka_wooden` piece subfolder
only. Do not vendor the other 7 piece subfolders without independent
license confirmation from their original sources.

---

## gbtami/chessgroundx

**What can be learned:** A working, small (~10KB gzipped), actively
maintained (last commit 2026-05-12) board interaction library with native
non-8×8 board support (`BoardDimensions` in `types.ts`/`config.ts`), CSS
transform-based animation (`anim.ts`), and both drag-and-drop and
click-to-move input (`drag.ts`, `events.ts`), with confirmed touch handling
(single-finger touch guard, `preventDefault()` on touchend).

**What should not be copied:** N/A — used as a dependency, not a code
source.

**How it applies to Sinsan:** Candidate board-interaction layer for `/`
(play) and `/analysis` routes.

**Remaining uncertainties:** Exact production bundle size once tree-shaken
into our own build (the ~10KB figure is from an npm listing, not our own
build output) — to be measured directly once integrated (Phase 2).

**Alternative approaches:** `cm-chessboard` (MIT core, but bundled pieces
are CC-BY-NC-SA — would need swapping, and non-8×8 support is unconfirmed)
and `alepot55/Chessboard.js` (license/9×10 support unverified) were found and
are noted as fallback options if chessgroundx integration hits problems, not
adopted now — no candidate beats chessgroundx's already-proven 9×10 use in
pychess-variants without unverified extra work.

**Licensing implications:** GPL-3.0-or-later (confirmed: README explicitly
says "or any later version," `LICENSE` is standard GPLv3 text). GPLv3 §13
explicitly permits combination with AGPLv3 code — no blocking incompatibility
with Sinsan's AGPL-3.0-or-later.

**Maintenance risk:** Low currently; single-maintainer risk not assessed
further at Phase 0 depth.

**Decision:** Adopt as the board interaction layer, vendored per package
manager (not a CDN), with license recorded in `THIRD_PARTY_NOTICES.md`.

---

## Kadagaden/chess-pieces

**What can be learned:** Original upstream source for the `Ka_wooden` /
`Ka_kakao` Janggi piece sets used by pychess-variants; dedicated Janggi
section with wooden and Kakao-inspired styles.

**What should not be copied:** The other (non-Janggi) piece sets — out of
scope.

**How it applies to Sinsan:** Confirms and cross-validates the CC-BY-4.0
license pychess-variants claims for `Ka_wooden`/`Ka_kakao` specifically.

**Remaining uncertainties:** None for the two verified sets.

**Alternative approaches:** N/A — this is the primary source, not an
alternative.

**Licensing implications:** CC BY 4.0 (confirmed: root `LICENSE.txt` full
text; README: "Where it's not stated otherwise, my work is licensed under
CC-BY-4.0"). Requires attribution — recorded in `THIRD_PARTY_NOTICES.md`.

**Maintenance risk:** Low — active (last commit 2026-05-19).

**Decision:** Use as the authoritative source for piece-set attribution
text, crediting Kadagaden directly per CC-BY-4.0 terms.

---

## Comparative Janggi AI implementations (Aunsiels/alphazero_janggi,
## boardgame1/AlphaJanggi, maksimKorzh/bmcp-janggi, woowacourse/java-janggi)

**What can be learned:** Two independent AlphaZero-style Janggi engines
(`alphazero_janggi`, `AlphaJanggi`) both converged on a per-origin-square,
move-template action encoding (~58 template slots × 90 squares ≈ 5,225–5,400
actions) — real independent validation that the spec's 60-template/5,401-
action design (Section 9) is a well-trodden choice for this exact game, not
an arbitrary guess.

**What should not be copied:** Any code at all. `alphazero_janggi` and
`java-janggi` have no LICENSE (all-rights-reserved by default);
`boardgame1/AlphaJanggi` is GPL-3.0 (copyleft — would need explicit
compliance work we have no reason to take on for comparative reference only);
`bmcp-janggi` is WTFPL-equivalent but explicitly simplified — its own header
states it deliberately drops bikjang, pass moves, and 3-fold-repetition
draw, making it a counter-example for rule fidelity, not a source.

**How it applies to Sinsan:** Comparative validation of action-space design
only; `java-janggi`'s `main` branch is empty bootcamp scaffolding (learners'
work lives in forks/PRs) and contributes nothing usable.

**Remaining uncertainties:** Neither `alphazero_janggi` nor `AlphaJanggi`
documents bikjang handling explicitly in source found during this pass —
unclear whether it's folded into unnamed legality logic or absent.

**Alternative approaches:** N/A — these are the alternatives being compared
against our own design, not alternatives to something else.

**Licensing implications:** See per-repo notes above; none are reused as
code regardless of license.

**Maintenance risk:** N/A, not a dependency.

**Decision:** Cite as comparative validation for the action-encoding
approach in `MODEL_DESIGN.md`. Do not copy code from any of them.

---

## Rapfi (arXiv 2503.13178)

**What can be learned:** A real (verified via arXiv ID), competition-winning
(GomoCup 2024, #1/520 on Botzone) Gomoku engine distills CNN knowledge into
a compact "pattern-based codebook" — a lookup-structure alternative to a
standard quantized NN forward pass, for accelerator-free CPU environments.

**What should not be copied:** Gomoku-specific pattern definitions.

**How it applies to Sinsan:** A candidate hybrid idea — cheap
codebook/lookup shortcuts alongside the tiny net for leaf evaluation — noted
as a speculative, unproven-for-Janggi stretch experiment in `MODEL_DESIGN.md`,
not a committed design.

**Remaining uncertainties:** Only read at abstract/summary level, not the
full paper — implementation details not verified.

**Alternative approaches:** Standard quantized-NN-only approach (our current
default) — simpler and already spec'd; the codebook idea is additive, not a
replacement.

**Licensing implications:** N/A — idea/technique reference, not code or
data reuse.

**Maintenance risk:** N/A.

**Decision:** Note as a ranked (medium-confidence) stretch idea, not adopted
into the Phase 0-1 design.

---

## Access Issues Summary

- Korea Janggi Association (kja.or.kr) — TLS certificate error, content
  unverifiable. `docs/RULES.md` uses 대한장기연맹 (kojf.net, reachable) tournament-
  procedure text as a partial, explicitly-labeled substitute, and does not
  conflate the two organizations.
- Exact `fairy-stockfish.wasm` JS invocation pattern — not confirmed from
  source, only its existence and Janggi support.
- Current (2026) status of historical Fairy-Stockfish repetition/bikjang
  GitHub issues (#40, #186, #198) — not rechecked, flagged as stale info.
- arXiv 2410.05347 ("Bridging Local and Global Knowledge via Transformer in
  Board Games") — title-level lead only, not fetched or evaluated further.
