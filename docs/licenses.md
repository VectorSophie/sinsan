# Licensing Policy

Status: Phase 0 research complete for the dependencies below (2026-07-30).
See `docs/RESEARCH.md` and `docs/REFERENCES.md` for the underlying analysis
and `THIRD_PARTY_NOTICES.md` for the per-file attribution ledger this policy
feeds. Anything not listed in the audit table below has not been cleared
for use.

## Policy

1. Sinsan's own code is AGPL-3.0-or-later. Full text in `LICENSE`.
2. Third-party code and assets retain their original license. Sinsan does not
   relicense anything it did not author.
3. Nothing is vendored into this repository without a verified license —
   "the GitHub repo is public" is not sufficient. A license must be read
   directly (root LICENSE file, or per-asset attribution file) before reuse.
4. Assets or code under missing, unclear, incompatible, personal-use-only, or
   non-commercial-only terms are not used, regardless of how well they fit.
5. GPL-family engines used only as an offline teacher/development-time
   verification tool (never shipped to the browser) are tracked here with
   their own obligations documented explicitly, since "teacher-only, not
   shipped" is a design choice that must be justified, not assumed safe.
6. Every full third-party license text referenced here has a verbatim copy
   stored under `licenses/` for redistribution purposes.

## Per-dependency audit

| Component | Origin | License (verified) | Role in Sinsan | Shipped to browser? | Notes |
|---|---|---|---|---|---|
| Fairy-Stockfish | fairy-stockfish/Fairy-Stockfish | GPL-3.0 (verified: `Copying.txt`) | Teacher engine, separate UCI subprocess | No (dev-time only) | Pin to commit `c19b5f6c66894fdb0e88d0dd100e3885f744760a` (2026-07-23 snapshot); re-verify SHA at actual build time. Arm's-length process use, not linked/distributed with Sinsan code. |
| Janggi NNUE net (`janggi-9991472750de.nnue`) | fairy-stockfish/Fairy-Stockfish-NNUE, contributor "belzedar_" | **UNRESOLVED — do not bundle or redistribute** | Teacher evaluation | No | Predates Fairy-Stockfish's 2026-CC0 grant cutoff; no other license text found anywhere. Local/private research use is a separate, lower-risk question not resolved here. Blocking for any public dataset/model card that cites it (see `DATASET_DESIGN.md`). |
| Janggi wood board images (`JanggiWood.png`, `JanggiWoodDark.svg`, `JanggiBrown.svg`) | gbtami/pychess-variants, `static/images/board/` | AGPL-3.0-or-later (repo root `LICENSE` + `static/COPYING.md` blanket claim, uncontradicted for this directory) | Vendored board image | Yes | |
| Janggi piece set `Ka_wooden` (CSS family `janggihb`) | gbtami/pychess-variants (`static/images/pieces/janggi/Ka_wooden`), originally Kadagaden/chess-pieces | CC BY 4.0 (verified independently: per-folder `license.txt` in pychess-variants AND root `LICENSE.txt` in Kadagaden/chess-pieces) | Vendored piece set | Yes | Requires attribution to Kadagaden. **Do not** use the other 7 `janggihb` subfolders — `hanjablue` is actually GFDL (not CC-BY as pychess-variants' own summary table incorrectly claims), `intlblue` is marked "unknown," and `hanjagreen`/`intlgreen`/`intlwooden`/`intlkakao` have no license file at all. |
| Board interaction library | gbtami/chessgroundx | GPL-3.0-or-later (verified: README + `LICENSE`) | UI dependency, vendored via package manager | Yes | GPLv3 §13 permits combination with AGPLv3. Actively maintained (last commit 2026-05-12), ~10KB gzipped, confirmed 9×10 support (used in production by pychess-variants for Janggi). |

## Open questions from Phase 0 research

- **Janggi NNUE net license** — the single biggest open item. Options to
  pursue before any public release referencing it: contact the contributor
  ("belzedar_") for explicit terms, wait for/check a future Fairy-Stockfish
  NNUE re-release under the 2026 CC0 policy, or fall back to Fairy-
  Stockfish's classical (non-NNUE) Janggi evaluation as teacher — the latter
  would weaken teacher quality and must be disclosed as such if chosen.
- **Whether teacher-generated dataset labels inherit any obligation from the
  NNUE net's unclear license** — distinct from the net-file question above;
  tracked in `DATASET_DESIGN.md` as a dataset-publication blocker, not
  resolved here.
- **KJA (kja.or.kr) rules document status** — inaccessible (TLS cert error)
  during this research pass; no licensing question yet since no content was
  retrieved, but re-attempt before treating any future extracted rules text
  as clearable for reuse.
