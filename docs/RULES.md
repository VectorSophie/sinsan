# Rules

## Source hierarchy and honesty about access

The spec's instruction is to treat official Korean Janggi Association rules
as primary, and to document gaps rather than hide them if official material
is inaccessible. During Phase 0 research:

- **대한장기협회** (Korea Janggi Association, kja.or.kr) — the site failed
  with a TLS self-signed-certificate error. Its content could not be
  verified. This is **not** the same organization as:
- **대한장기연맹** (Korea Janggi Federation, kojf.net) — reachable, and its
  tournament-regulation page (`content03.php`) gives real procedural text:
  bikjang cannot be called until after a piece has been captured, and
  repetition is allowed up to 3 times with the 4th occurrence forfeiting.
  This is competition procedure, not a complete from-scratch rules
  document, and it is a different body from the KJA. It is cited here as a
  partial, explicitly-labeled substitute — not represented as "the official
  KJA rules."
- **Fairy-Stockfish and pychess-variants** — used as the practical
  cross-reference baseline per the spec's own contingency plan, since they
  are actively maintained, source-verifiable implementations rather than
  secondary descriptions.

No claim in this document should be read as an assertion that KJA's rules
were consulted directly. Re-attempting kja.or.kr access is an open item in
`RESEARCH.md`.

## RuleProfile design

Fairy-Stockfish hardcodes four distinct Janggi rule flavors (confirmed by
reading `src/variant.cpp`), which map directly onto a `RuleProfile` concept:

| Profile id | Bikjang | Material-count adjudication | Repetition | Notes |
|---|---|---|---|---|
| `kja` (default) | Yes | Yes | standard | Maps to Fairy-Stockfish's `janggi` variant — closest match to competitive/tournament play, combining both bikjang and material-count adjudication. |
| `traditional` | Draw on bikjang | No | standard | Maps to `janggitraditional`. |
| `modern` | No | Yes | 4-fold illegal, 100-move rule | Maps to `janggimodern` — tuned to match the Kakao Janggi mobile app's ruleset, per Fairy-Stockfish's own variant naming. |
| `casual` | No | No | standard | Maps to `janggicasual`. |

Sinsan's rules engine (`packages/rules`) must accept a `RuleProfile` as an
explicit input to `createInitialPosition(options)` and `getGameResult(...)`
rather than hardcoding one ruleset — the spec's own instruction, reinforced
by the fact that even the teacher engine itself doesn't have a single
canonical Janggi ruleset.

**Important honesty note:** naming the default profile `kja` reflects intent
(closest to what the project believes competitive Korean Janggi rules to
be), not verified confirmation that KJA specifically mandates this exact
combination — that verification is blocked by the site access issue above.
If KJA access is restored and its rules differ, `kja` should be corrected
rather than left as a misleading label.

## Starting formations

Each side chooses independently among four horse/elephant configurations.
Internal identifiers are our own (English, stable), chosen for clarity —
not adopted from any single external source, since the spec itself warns
against assuming informal English names online are standardized:

| Internal id | Korean | Default? |
|---|---|---|
| `masang-sangma` | 마상상마 | Yes (both sides) |
| `sangma-masang` | 상마마상 | |
| `masang-masang` | 마상마상 | |
| `sangma-sangma` | 상마상마 | |

## Core rules to implement (Phase 1)

- Cho moves first; Han's compensation (komi/material) is a `RuleProfile`
  parameter, not a hardcoded constant.
- Chariot: orthogonal any distance, plus palace diagonal lines when inside
  the palace (`diagonalLines` bitboard equivalent — palace corner-to-center
  diagonals only, confirmed structure from Fairy-Stockfish source: D1/F1/E2/
  D3/F3 and D8/F8/E9/D10/F10-equivalent squares on our own coordinate
  system).
- Cannon: orthogonal + palace diagonal, but only by jumping exactly one
  non-cannon screening piece; cannots capture another cannon, and cannot use
  a cannon as its own screen (both explicit, testable rules).
- Horse: standard one-orthogonal-then-one-diagonal-outward move, blocked if
  the adjacent orthogonal square is occupied.
- Elephant: one-orthogonal-then-two-diagonal-outward move, blocked if either
  intermediate square is occupied.
- Soldier: forward or sideways one step; inside the palace, forward diagonal
  moves along palace diagonal lines are additionally legal.
- General/Guard: one step in any direction within the palace, including
  palace diagonals.
- Self-check illegal: no move may leave one's own general in check
  (standard "flying general" rule is explicitly **disabled** in
  Fairy-Stockfish's Janggi config — `flyingGeneral=false` — generals do not
  capture each other directly across an open file the way Xiangqi allows).
- Pass: a dedicated, always-legal move (unless in check, per standard
  Janggi convention) — represented internally with a distinct action id
  (see `packages/action-space`), matching Fairy-Stockfish's own
  same-square-twice UCI convention for cross-checking against the teacher.
- Check / checkmate / resignation / timeout: standard.
- Bikjang: native concept — both generals facing each other on an open file
  with no piece between them; handling (draw vs. adjudication trigger)
  depends on the active `RuleProfile`.
- Repetition / perpetual check: profile-dependent thresholds (see table
  above); `perpetualCheckIllegal=true` confirmed as a Fairy-Stockfish default
  across all four flavors.
- No-capture move limit and material-count adjudication: profile-dependent;
  material-count logic itself (which pieces count, at what value) should be
  cross-checked against Fairy-Stockfish's own `JANGGI_MATERIAL` result path
  during differential testing rather than reimplemented from a separate
  source.

## Differences to test for explicitly

Because rule flavors genuinely differ (not just theoretically — Fairy-
Stockfish ships four named variants specifically because ambiguity was
historically a real bug source, per GitHub issues #40/#186/#198 from
2019-2020), `tests/fixtures/rules/` must include at least one fixture
position per profile that produces a *different* legal-move set or game
result depending on which `RuleProfile` is active (e.g., a bikjang position
that's a draw under `traditional` but not under `kja`). A rules engine that
passes tests only under one profile has not actually verified profile
correctness.

## Verification plan

Per Section 8.4: unit tests per piece/rule, fixed regression fixtures under
`tests/fixtures/rules/`, and differential testing against `pyffish` (rules/
legality/adjudication only — confirmed in `REFERENCES.md` that `pyffish`
does not expose search/eval, so it's exactly suited to this role and nothing
more). Mismatches auto-save a minimal repro under `artifacts/rule-mismatches/`
per the spec; none have been generated yet since the rules engine itself is
Phase 1, not Phase 0.
