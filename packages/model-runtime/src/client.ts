import { fetchManifestAndWeights } from './manifest.ts';
import type { ModelManifest } from './manifest.ts';
import type { WorkerRequest, WorkerResponse } from './worker.ts';

export interface ModelWorkerClient {
  infer(inputPlanes: Float32Array): Promise<{ policyLogits: Float32Array; value: number }>;
  readonly manifest: ModelManifest;
  terminate(): void;
}

/**
 * Fetches the manifest + weights on the main thread, verifies the SHA-256 digest (Moka's
 * verified pattern - see docs/ARCHITECTURE.md), then transfers the buffer into a dedicated
 * Worker where all inference happens. `workerUrl` should be `new URL('./worker.ts',
 * import.meta.url)` from the caller's own module (bundler-relative), not a path this package can
 * know in advance.
 */
export async function createModelWorkerClient(
  workerUrl: string | URL,
  manifestUrl: string,
  weightsUrl: string,
): Promise<ModelWorkerClient> {
  const { manifest, weightsBuffer } = await fetchManifestAndWeights(manifestUrl, weightsUrl);

  const worker = new Worker(workerUrl, { type: 'module' });
  let nextRequestId = 1;
  const pending = new Map<number, { resolve: (v: { policyLogits: Float32Array; value: number }) => void; reject: (e: Error) => void }>();

  const ready = new Promise<void>((resolve, reject) => {
    const onFirstMessage = (event: MessageEvent<WorkerResponse>) => {
      worker.removeEventListener('message', onFirstMessage);
      if (event.data.type === 'ready') resolve();
      else reject(new Error(`model-runtime: expected 'ready' from worker, got ${event.data.type}`));
    };
    worker.addEventListener('message', onFirstMessage);
  });

  worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
    const msg = event.data;
    if (msg.type === 'result') {
      pending.get(msg.requestId)?.resolve({ policyLogits: msg.policyLogits, value: msg.value });
      pending.delete(msg.requestId);
    } else if (msg.type === 'error' && msg.requestId !== undefined) {
      pending.get(msg.requestId)?.reject(new Error(msg.message));
      pending.delete(msg.requestId);
    }
  });

  const initMessage: WorkerRequest = { type: 'init', manifest, weightsBuffer };
  worker.postMessage(initMessage, [weightsBuffer]);
  await ready;

  return {
    manifest,
    infer(inputPlanes: Float32Array) {
      const requestId = nextRequestId++;
      return new Promise((resolve, reject) => {
        pending.set(requestId, { resolve, reject });
        const message: WorkerRequest = { type: 'infer', requestId, inputPlanes };
        worker.postMessage(message, [inputPlanes.buffer]);
      });
    },
    terminate() {
      worker.terminate();
    },
  };
}
