/**
 * Real, host-measured latency benchmark for packages/model-runtime (Section 18.2), separating
 * one-time cost (fetch/digest/dequantize-at-construction) from steady-state per-call inference
 * time - the smoke-tier live browser test only measured two cold calls (301ms, 360ms), which
 * conflates the two and isn't enough to know whether conv2d itself needs optimizing.
 *
 * Run with: node benchmarks/model-inference.ts
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { SinsanModel, verifyWeightsDigest } from '../packages/model-runtime/src/index.ts';
import { positionToPlanes } from '../packages/model-runtime/src/features.ts';
import { createInitialPosition } from '../packages/rules/src/index.ts';
import type { ModelManifest } from '../packages/model-runtime/src/index.ts';

const repoRoot = join(import.meta.dirname, '..');
const manifest: ModelManifest = JSON.parse(
  readFileSync(join(repoRoot, 'public', 'model', 'sinsan-smoke-v0.json'), 'utf8'),
);
const weightsBuffer = new Uint8Array(readFileSync(join(repoRoot, 'public', 'model', 'sinsan-smoke-v0.bin'))).buffer;

function percentile(sorted: number[], p: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]!;
}

async function main() {
  const t0 = performance.now();
  await verifyWeightsDigest(manifest, weightsBuffer);
  const digestMs = performance.now() - t0;

  const t1 = performance.now();
  const model = new SinsanModel(manifest, weightsBuffer);
  const constructMs = performance.now() - t1;

  const input = positionToPlanes(createInitialPosition());

  // Warm-up (JIT) - not counted, matching standard benchmarking practice. conv2d is called with
  // 9 different tensor shapes per forward pass (stem, 8 tower convs, policy/value heads); V8
  // needs enough calls per *shape*, not just per function, to tier up to optimized code, so this
  // needs to be much larger than a typical single-shape microbenchmark's warm-up count.
  for (let i = 0; i < 200; i++) model.infer(input);

  const N = 100;
  const times: number[] = [];
  for (let i = 0; i < N; i++) {
    const start = performance.now();
    model.infer(input);
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0) / times.length;

  console.log(`SHA-256 digest verification: ${digestMs.toFixed(2)}ms (one-time)`);
  console.log(`Model construction (dequantize all layers): ${constructMs.toFixed(2)}ms (one-time)`);
  console.log(
    `Steady-state inference (${N} calls, after 200 warm-up): mean=${mean.toFixed(2)}ms ` +
      `p50=${percentile(times, 0.5).toFixed(2)}ms p95=${percentile(times, 0.95).toFixed(2)}ms ` +
      `min=${times[0]!.toFixed(2)}ms max=${times[times.length - 1]!.toFixed(2)}ms`,
  );
}

main();
