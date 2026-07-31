# Architecture

## Layering

```
apps/web            Static Korean-Janggi web app (play, analysis, arena, research, about)
apps/arena           Headless parallel-match runner (Node), reuses packages/* directly
packages/rules       Browser-native TS rules engine (immutable position, legal moves, result)
packages/action-space Encoding/decoding between moves and the 5,401-action policy space
packages/model-runtime  Worker-hosted INT8 model loader + float32 tensor ops + digest check
packages/search      MCTS/PUCT over packages/rules + packages/model-runtime, tree reuse
packages/ui          Board rendering (wraps chessgroundx), shared components
training/*           Python: teacher adapter, data generation, dataset build, train, export, evaluate
assets/*             Vendored board/piece/sound assets, license-tracked
tests/*              Unit, differential (vs pyffish/Fairy-Stockfish), model, browser (Playwright)
```

Everything under `packages/` and `apps/web` must run with no server and no
network calls after the model is cached — the static-deployment requirement
in Section 3.3. `training/*` is a separate Python environment that never
ships to the browser.

## Rules engine (`packages/rules`)

Public API, per Section 8.3:

```ts
createInitialPosition(options?: CreatePositionOptions): Position
generateLegalMoves(position: Position): Move[]
applyMove(position: Position, move: Move): Position   // returns new state, no mutation
isCheck(position: Position): boolean
getGameResult(position: Position, clock?: Clock): GameResult | null
getMaterialScore(position: Position): number
serializePosition(position: Position): string
parsePosition(serialized: string): Position
```

**Implemented deviation from the spec's suggested signature:** `getGameResult` takes no separate
`history: Position[]` parameter. `Position` carries its own `positionHistory` (serialized
board+side-to-move keys for every position reached so far), populated automatically by
`createInitialPosition` and `applyMove`. A caller cannot pass a `Position` and a mismatched
history out of sync with each other, since there is only one thing to pass. This is implemented
(`packages/rules/src/{types,position,rules}.ts`), not just proposed - 43 unit tests pass,
including repetition- and bikjang-adjudication cases that exercise `positionHistory` directly.

`Position` is immutable; `applyMove` returns a new object. This matters
specifically for `packages/search`, which explores many branches from the
same position concurrently (or in tight loops) — implicit mutation here is
exactly the class of bug that's expensive to track down inside a tree
search, so it is designed out from the start rather than fixed later.

`RuleProfileId` (`kja` | `traditional` | `modern` | `casual`, see
`RULES.md`) is a required, explicit input — never a hidden default baked
into move-generation logic.

## Teacher adapter boundary

Fairy-Stockfish is invoked as a **separate OS process over UCI** from
`training/teacher/`, in Python (subprocess) — never compiled into, linked
with, or shipped alongside any browser code. This is both a licensing
boundary (GPL-3.0 engine, arm's-length process communication, no
distribution of engine code with Sinsan's own AGPL code — see
`docs/licenses.md`) and an architectural one (the browser artifact must
work fully offline with no engine dependency at all).

`pyffish` is used only inside `tests/differential/` for legality/
adjudication cross-checks against `packages/rules` — confirmed during
research to expose exactly that surface (legal moves, FEN, game-result,
bikjang/material adjudication) and nothing resembling search or eval, so it
cannot accidentally become a hidden teacher-quality dependency.

## Browser model runtime (`packages/model-runtime`)

Modeled on Moka's verified pattern (see `REFERENCES.md`), not
reimplemented from scratch conceptually:

1. Main thread fetches `sinsan-v1.json` (manifest) and `sinsan-v1.bin`
   (weights).
2. Main thread computes `crypto.subtle.digest("SHA-256", weightsBuffer)`
   and compares against `manifest.sha256` **before** transferring the
   buffer to the Worker. A corrupted or mismatched artifact is rejected
   before any inference code touches it.
3. The buffer is handed to a dedicated Web Worker via a `Transferable`
   (zero-copy).
4. The Worker dequantizes INT8 weights to `Float32Array` once at
   initialization (per-output-channel scale, symmetric, weight-only —
   activations are never quantized, matching the verified Moka approach).
5. All convolution/linear/relu/inference happens inside the Worker. The
   main thread never runs tensor math — a hard requirement (Section 2),
   not just a preference.
6. `packages/search` (MCTS/PUCT) also runs Worker-side, batching leaf
   evaluations into the same runtime rather than round-tripping to the main
   thread per node.

**Why not ONNX Runtime Web or WebGPU by default:** for a network this small
(tens to a few hundred KB, 10×9 spatial extent), the JS↔WASM boundary and
WebGPU dispatch/shader-compile/readback overhead are plausible to exceed the
actual compute cost — this is Moka's stated reasoning, corroborated
directionally (not rigorously — see `RESEARCH.md`) by 2025 blog-level
benchmarks on small-tensor inference. Section 15's evaluation order (JS/TS
first, WASM only if measured faster, WebGPU only if it wins real browser
benchmarks) is adopted as-is; the "if measured" qualifier is load-bearing —
Phase 2 must produce our own numbers before this is treated as settled
rather than borrowed.

## Data flow (end to end)

```
Fairy-Stockfish (UCI subprocess, pinned commit)
  -> training/teacher (labeling: policy/value/WDL per position)
  -> training/datasets (Parquet/NPZ shards, game-level train/val/test split)
  -> training/train (PyTorch CPU, float32)
  -> training/export (fold BatchNorm, per-channel INT8 quantize, manifest+sha256)
  -> public/model/sinsan-v1.{json,bin}
  -> packages/model-runtime (Worker: verify digest, dequantize, infer)
  -> packages/search (PUCT using packages/rules for legality + model-runtime for priors/value)
  -> packages/ui + apps/web (render, animate, report)
```

`apps/arena` and `/arena`, `/analysis`, `/research` routes consume the same
`packages/*` stack as `/` — there is deliberately no separate "production"
vs "research" inference path, since divergence between them is exactly how
a project ends up benchmarking something other than what it ships.

## Testing architecture

- `tests/rules`: unit tests per piece/rule (Section 8.4).
- `tests/fixtures/rules/`: fixed regression positions, including at least
  one per-`RuleProfile` divergence case (see `RULES.md`).
- `tests/differential`: randomized legal-position generation, compared
  against `pyffish`; mismatches auto-saved under `artifacts/rule-mismatches/`.
- `tests/model`: action-encoding round-trip (every legal move ↔ exactly one
  action id, no collisions, symmetry transform correctness), quantization
  round-trip, manifest/digest verification.
- `tests/browser`: Playwright, run against Chromium/Firefox/WebKit before
  public release (Section 3.1) — not part of the Phase 0-1 scope for this
  session, since there's no playable board yet to test.
