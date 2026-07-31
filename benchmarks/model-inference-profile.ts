/**
 * Isolates a single conv2d call's cost at the tower's actual shape. Used during Phase 4's
 * inference-latency investigation to confirm conv2d (not digest verification, model construction,
 * or the policy/value heads) is where full-model inference time actually goes - see
 * docs/BENCHMARK_PLAN.md for the full writeup, including a pre-padding optimization attempt that
 * measured *worse* at the full-model level (this isolated benchmark originally made it look
 * better, by excluding the padding allocation cost from the timed region - a lesson in why
 * isolated microbenchmarks of one piece can mislead about the whole).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { conv2d, makeTensor3D } from '../packages/model-runtime/src/tensor-ops.ts';

const repoRoot = join(import.meta.dirname, '..');
const manifest = JSON.parse(readFileSync(join(repoRoot, 'public', 'model', 'sinsan-smoke-v0.json'), 'utf8'));
console.log('layers:', manifest.layers.map((l: any) => l.name));

// Isolate conv2d cost alone, at the tower's actual shape (32ch, 10x9, k=3, pad=1), with much
// heavier warm-up to rule out "V8 hasn't tiered up to optimized code yet" as a confound.
const input = makeTensor3D(32, 10, 9);
for (let i = 0; i < input.data.length; i++) input.data[i] = Math.random();
const weight = new Float32Array(32 * 32 * 3 * 3).map(() => Math.random());
const bias = new Float32Array(32).map(() => Math.random());

for (let i = 0; i < 2000; i++) conv2d(input, weight, bias, 32, 3, 1); // heavy warm-up

const N = 500;
const times: number[] = [];
for (let i = 0; i < N; i++) {
  const t = performance.now();
  conv2d(input, weight, bias, 32, 3, 1);
  times.push(performance.now() - t);
}
times.sort((a, b) => a - b);
const mean = times.reduce((a, b) => a + b, 0) / times.length;
console.log(`single 32ch conv2d (10x9,k3,pad1), ${N} calls after 2000 warm-up:`);
console.log(`  mean=${mean.toFixed(4)}ms p50=${times[N >> 1]!.toFixed(4)}ms min=${times[0]!.toFixed(4)}ms max=${times[N - 1]!.toFixed(4)}ms`);
console.log(`  -> tower has 8 of these + a smaller stem + policy/value heads`);
