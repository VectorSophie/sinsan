/**
 * Worker entry point - all tensor computation happens here, never on the main thread (Section 2,
 * hard requirement). The main thread (client.ts) fetches the manifest/weights and verifies the
 * SHA-256 digest *before* transferring the buffer here; this file trusts that already happened.
 */
import { SinsanModel } from './model.ts';
import type { ModelManifest } from './manifest.ts';

export type WorkerRequest =
  | { type: 'init'; manifest: ModelManifest; weightsBuffer: ArrayBuffer }
  | { type: 'infer'; requestId: number; inputPlanes: Float32Array };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'result'; requestId: number; policyLogits: Float32Array; value: number }
  | { type: 'error'; requestId?: number; message: string };

let model: SinsanModel | undefined;

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === 'init') {
      model = new SinsanModel(msg.manifest, msg.weightsBuffer);
      const response: WorkerResponse = { type: 'ready' };
      (self as unknown as Worker).postMessage(response);
      return;
    }
    if (msg.type === 'infer') {
      if (!model) throw new Error('model-runtime worker: infer requested before init');
      const result = model.infer(msg.inputPlanes);
      const response: WorkerResponse = {
        type: 'result',
        requestId: msg.requestId,
        policyLogits: result.policyLogits,
        value: result.value,
      };
      (self as unknown as Worker).postMessage(response, [result.policyLogits.buffer]);
      return;
    }
  } catch (err) {
    const response: WorkerResponse = {
      type: 'error',
      requestId: 'requestId' in msg ? msg.requestId : undefined,
      message: err instanceof Error ? err.message : String(err),
    };
    (self as unknown as Worker).postMessage(response);
  }
};
