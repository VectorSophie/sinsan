# Research

This document synthesizes Phase 0 research into a narrative: what question
we're actually answering, what we found, what we're adopting, and what
remains open. Per-source detail lives in `REFERENCES.md`; this document is
the "why," not a link list.

## The question

Moka asks how small a Go policy/value network can be while still living in
a browser. Sinsan asks the Janggi-specific version of that question: how
much of Janggi's strategy — cannon screening restrictions, palace
diagonals, horse/elephant blocking, bikjang, material-count adjudication —
survives compression into a model under 500KiB. That question can't be
answered by reskinning Moka's Go network; it requires Janggi's own action
space, input representation, and rules engine, informed by Janggi-specific
prior art. That's what Phase 0 was for.

## Method

Rather than summarize these projects from training-data memory (explicitly
disallowed by the project's own ground rules), four research passes cloned
and read the actual repositories: Moka itself, the Fairy-Stockfish family,
the board/UI asset chain (pychess-variants → chessgroundx / Kadagaden), and
comparative Janggi engines plus a broader search for post-2024 optimization
ideas. Every claim in `REFERENCES.md` is tagged with whether it was
confirmed by direct file access or is a secondary/unverified lead.

## What's adopted, and why

**Weight-only INT8, dequantize-at-load, float32 Worker inference.** Moka's
runtime never actually computes in int8 — it's a storage/transfer format,
decoded once into `Float32Array` before any arithmetic. This sidesteps the
complexity of real integer kernels for a network this small, where the
JS-to-native boundary cost would likely exceed any compute savings. Adopted
as the default for `packages/model-runtime`.

**SHA-256 pre-verification of the weights buffer**, computed on the main
thread via `crypto.subtle.digest` before ever handing the buffer to the
Worker — not just a manifest byte-length check. Adopted directly.

**Game-level dataset splitting via a hashed bucket id**, with fixed
validation/test buckets that never receive on-policy data. Directly
confirmed working in Moka's `train.py`; matches the spec's own requirement
and is adopted without modification.

**Goldilocks-weighted on-policy correction** (Moka) and **tau-style
aggregated soft-policy labeling** (`pbaer/neural-chess`, a new find, not in
the original reference list) are both adopted as named, measured techniques
in `DATASET_DESIGN.md` rather than the spec's more generic "weight confident
mistakes higher" — both are more principled and have precedent (the tau
recipe reports +79 Elo at equal parameters on chess; untested for Janggi
until we measure it ourselves).

**Template-based action encoding is independently validated**, not just
internally consistent: two unrelated AlphaZero-style Janggi engines
(`alphazero_janggi`, `AlphaJanggi`) converged on their own per-origin-square
move-template schemes in the same ~58-60-templates-per-square range as the
spec's 60-template/5,401-action design. This raises confidence the design
is sound without having built it yet.

## What's flagged as a real risk, not smoothed over

- **The strongest available Janggi NNUE net has an unresolved license.**
  Fairy-Stockfish's NNUE page grants CC0 to nets dated 2026 or later; the
  only dedicated Janggi net found (`janggi-9991472750de.nnue`, 2025-08-01,
  claimed +1128 Elo) predates that cutoff with no other license text
  anywhere. This blocks bundling/redistributing it until clarified — see
  `docs/licenses.md`. We may still use it locally as a private research
  teacher, but that is a separate, lower-stakes question from publication.
- **The spec's assumed hardware doesn't match this machine.** Actual CPU is
  an i5-1035G7 (not i5-1135G7), and available RAM was measured at ~2.7-2.8GB
  free out of 7.4GB total during this session — well below the spec's
  assumed 16GB. Teacher-engine parallelism and training batch-size defaults
  should be tuned conservatively and re-checked via `scripts/check-host.sh`
  before any real run, not assumed from the planning doc.
- **Janggi rules are not a single agreed ruleset even inside Fairy-
  Stockfish.** Four variant flavors exist (`janggi`, `janggitraditional`,
  `janggimodern`, `janggicasual`) differing on bikjang and repetition
  handling. See `RULES.md` for the RuleProfile mapping and the KJA-vs-KJF
  organizational distinction (KJA's site was unreachable — cert error — so
  its rules could not be directly verified this session).
- **The "JS/WASM beats WebGPU for tiny tensors" claim is directional, not
  proven.** No rigorous first-party benchmark was found; sources were
  blog/SEO aggregators. Treated as a hypothesis to test ourselves in Phase
  2's browser benchmarking, consistent with Section 15's evaluation order,
  not as settled fact.
- **INT4 quantization for a network this size is unaddressed in available
  literature.** No PTQ-for-small-CNN-in-browser precedent was found. INT8
  remains the default; INT4 is a labeled stretch experiment requiring its
  own measurement (see `MODEL_DESIGN.md`), not a default assumption.

## Differentiator ideas surfaced (ranked, per `REFERENCES.md`)

1. Tau-style cross-game soft-policy aggregation (sourced, +79 Elo precedent
   on chess) — planned as a measured experiment in Phase 6.
2. Codebook/lookup-style hybrid leaf evaluation (Rapfi, Gomoku) — medium
   confidence, unproven for Janggi, noted as a stretch idea only.
3. Mixed-precision (INT4 core tower, INT8 boundary layers) — our own
   inference from general quantization literature, not domain-specific
   precedent; requires real strength testing before adoption.
4. Factorized (non-dense) policy head exploiting Janggi's more constrained
   move set — speculative, no external source, our own idea; listed as a
   candidate experiment only.

None of these four are committed design decisions. They are experiments to
run and measure against the baseline once Phase 1-5 produce something to
measure against.

## Open questions carried forward

- Confirm the Janggi NNUE net's actual license before any public dataset or
  model-card release references it.
- Re-verify the Fairy-Stockfish commit pin at actual build time (Phase 3) —
  the commit recorded in `REFERENCES.md` is a snapshot from 2026-07-30 of an
  actively developed, untagged `master` branch.
- Attempt KJA (kja.or.kr) access again later — the TLS failure may be
  transient; if it remains inaccessible, `RULES.md`'s kojf.net-based
  procedural citations stand as the best available partial substitute, not
  a claim of completeness.
