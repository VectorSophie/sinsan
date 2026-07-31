// Thin local entry so Vite's static `new Worker(new URL('./model-worker.ts', import.meta.url))`
// analysis can find a real file in this app; the actual worker logic lives in
// @sinsan/model-runtime (all tensor computation happens there, never on the main thread).
import '../../../packages/model-runtime/src/worker.ts';
