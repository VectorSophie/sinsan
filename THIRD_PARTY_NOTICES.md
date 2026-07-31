# Third-Party Notices

Sinsan's own source code is licensed AGPL-3.0-or-later (see `LICENSE`).
This file records every third-party dependency, asset, copied implementation,
adapted algorithm, or vendored file reused by Sinsan, per the project's
licensing policy in `docs/licenses.md`.

Status: **pending research**. Entries below are populated only after the
corresponding license has been verified by reading the actual upstream
LICENSE file or asset metadata — not assumed from typical conventions.
Do not vendor anything into this repository until it has an entry here.

Entry template:

```
## <Original project name>

- Original project: <org/repo>
- Original author(s): <name(s)>
- Exact file(s)/directory used: <path(s)>
- Original license: <SPDX identifier, verified from source>
- Commit / release / version pinned: <sha or tag>
- Modified: <yes/no, and how>
- Used by Sinsan for: <what>
- Required attribution: <exact text/credit required>
- Redistribution obligations: <copyleft propagation, notice requirements, etc.>
```

---

## Fairy-Stockfish

- Original project: fairy-stockfish/Fairy-Stockfish
- Original author(s): Fairy-Stockfish contributors (fork of Stockfish, Stockfish authors and contributors)
- Exact file(s)/directory used: none vendored — invoked as a separate compiled binary over UCI from `training/teacher/`
- Original license: GPL-3.0 (verified directly from `Copying.txt`)
- Commit / release / version pinned: `c19b5f6c66894fdb0e88d0dd100e3885f744760a` (2026-07-23 snapshot of `master`; no current tagged release since `fairy_sf_14_0_1_xq`, 2021-11-19). Built 2026-07-31 with `make build ARCH=x86-64-bmi2 largeboards=yes` (classical eval, no NNUE file loaded — see the NNUE net entry below for why). Binary SHA256: `c12c8beb85754bdcf32bf9ce9067a6b57c528bf0f49cc12720c09a871919fca3`. Verified: `UCI_Variant value janggi` responds with the expected 9x10 board, matches Sinsan's own default startpos exactly, and returns a legal `bestmove` after `go nodes 10000`.
- Modified: no
- Used by Sinsan for: offline teacher engine (move/position labeling, WDL, differential rules testing support via `pyffish`) — development-time only, never shipped to the browser. Source/binary live in `training/teacher/engine/` (gitignored, not committed) built by `scripts/build-teacher.sh`.
- Required attribution: cite project name, license, and pinned commit in `docs/licenses.md` and any model/dataset card describing how labels were generated
- Redistribution obligations: none triggered — Sinsan does not distribute Fairy-Stockfish's binary or source; it is built/run as an independent local tool communicating over UCI (arm's-length process use)

## Janggi NNUE net (`janggi-9991472750de.nnue`)

- Original project: fairy-stockfish/Fairy-Stockfish-NNUE
- Original author(s): community contributor "belzedar_" (credited on fairy-stockfish.github.io/nnue/)
- Exact file(s)/directory used: not yet used — status UNRESOLVED
- Original license: **unresolved** — Fairy-Stockfish's NNUE page grants CC0 only to nets dated 2026-or-later; this net (released 2025-08-01) predates that cutoff with no other license text found
- Commit / release / version pinned: release tag `janggi-9991472750de` (Fairy-Stockfish-NNUE repo)
- Modified: n/a
- Used by Sinsan for: not yet used pending license resolution — see `docs/licenses.md` open questions
- Required attribution: unknown pending resolution
- Redistribution obligations: **do not bundle, vendor, or redistribute this file** until license status is resolved

## Janggi wood board images

- Original project: gbtami/pychess-variants
- Original author(s): pychess-variants contributors
- Exact file(s)/directory used: `static/images/board/JanggiWood.png`, `JanggiWoodDark.svg`, `JanggiBrown.svg` → vendored to `assets/board/`
- Original license: AGPL-3.0-or-later (repo root `LICENSE`; `static/COPYING.md` blanket-attributes this directory to "pychess-variants contributors," uncontradicted by any more specific file)
- Commit / release / version pinned: `cc4e6a863feac488a8cba9d6a43f982d3ae66d20` (2026-07-30)
- Modified: no (vendored as-is)
- Used by Sinsan for: board rendering in `packages/ui`
- Required attribution: credit pychess-variants contributors, AGPL-3.0-or-later
- Redistribution obligations: AGPL-3.0-or-later copyleft — any modified redistribution of these specific files must remain under compatible terms; consistent with Sinsan's own AGPL-3.0-or-later licensing

## Janggi piece set: Ka_wooden (janggihb family)

- Original project: Kadagaden/chess-pieces (original), vendored via gbtami/pychess-variants (`static/images/pieces/janggi/Ka_wooden`)
- Original author(s): Kadagaden
- Exact file(s)/directory used: `Ka_wooden` subfolder only — explicitly **not** the other 7 `janggihb` subfolders (see `docs/licenses.md` for why: `hanjablue` is GFDL not CC-BY despite pychess-variants' summary table, `intlblue` is "unknown," four others have no per-folder license file at all) → vendored to `assets/pieces/janggihb-ka-wooden/`
- Original license: CC BY 4.0 (verified independently: pychess-variants' own `Ka_wooden/license.txt` AND Kadagaden/chess-pieces' root `LICENSE.txt`)
- Commit / release / version pinned: pychess-variants `cc4e6a863feac488a8cba9d6a43f982d3ae66d20` (2026-07-30); original source Kadagaden/chess-pieces `b035b0cc6a68e9fb99c872c8fe073c3ae3eba8a0` (2026-05-19)
- Modified: no (vendored as-is; filenames use `blue_`/`red_` prefixes matching Cho/Han per pychess-variants' own convention)
- Used by Sinsan for: piece rendering in `packages/ui`
- Required attribution: credit Kadagaden per CC BY 4.0 terms
- Redistribution obligations: CC BY 4.0 attribution requirement only; no copyleft propagation to Sinsan's own code

## chessgroundx

- Original project: gbtami/chessgroundx
- Original author(s): chessgroundx contributors (fork of lichess-org/chessground)
- Exact file(s)/directory used: `chessgroundx` npm package dependency (not a copied/vendored source fork)
- Original license: GPL-3.0-or-later (verified: README states "or any later version," `LICENSE` is standard GPLv3 text)
- Commit / release / version pinned: package version `10.7.5` (npm), matching source commit `112c6b616f9e9e6fb79528c430f602287caa3535` (2026-05-12)
- Modified: no (planned: as-is, used as a library)
- Used by Sinsan for: board rendering/interaction (drag, click-to-move, animation, touch input) in `packages/ui`
- Required attribution: credit chessgroundx, GPL-3.0-or-later
- Redistribution obligations: GPL-3.0-or-later copyleft; GPLv3 §13 explicitly permits combination with AGPLv3-licensed code, so no blocking incompatibility with Sinsan's own licensing
