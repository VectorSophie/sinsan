export interface LayerManifest {
  readonly name: string;
  readonly type: 'conv2d' | 'linear';
  readonly weight_shape: readonly number[];
  readonly weight_offset: number;
  readonly weight_bytes: number;
  readonly scale_offset: number;
  readonly bias_offset: number;
  readonly padding?: number;
}

export interface ModelManifest {
  readonly format_version: number;
  readonly model_name: string;
  readonly architecture: { readonly channels: number; readonly blocks: number; readonly kind: string };
  readonly input_shape: readonly [number, number, number];
  readonly policy_shape: readonly [number];
  readonly parameter_count: number;
  readonly quantization: string;
  readonly weights_bytes: number;
  readonly sha256: string;
  readonly training_run: string;
  readonly rule_profile: string;
  readonly layers: readonly LayerManifest[];
}

function bytesToHex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/** Computes the SHA-256 of the weights buffer and compares it against manifest.sha256 - Moka's
 * verified pre-load integrity check (docs/REFERENCES.md), run before the buffer is ever handed to
 * inference code. Throws on mismatch rather than returning a boolean, since a caller silently
 * ignoring a failed check here is exactly the bug this function exists to prevent. */
export async function verifyWeightsDigest(manifest: ModelManifest, weightsBuffer: ArrayBuffer): Promise<void> {
  if (weightsBuffer.byteLength !== manifest.weights_bytes) {
    throw new Error(
      `weights size mismatch: manifest says ${manifest.weights_bytes} bytes, got ${weightsBuffer.byteLength}`,
    );
  }
  const digest = await crypto.subtle.digest('SHA-256', weightsBuffer);
  const hex = bytesToHex(digest);
  if (hex !== manifest.sha256) {
    throw new Error(`weights digest mismatch: manifest says ${manifest.sha256}, computed ${hex}`);
  }
}

export async function fetchManifestAndWeights(
  manifestUrl: string,
  weightsUrl: string,
): Promise<{ manifest: ModelManifest; weightsBuffer: ArrayBuffer }> {
  const manifest: ModelManifest = await (await fetch(manifestUrl)).json();
  const weightsBuffer = await (await fetch(weightsUrl)).arrayBuffer();
  await verifyWeightsDigest(manifest, weightsBuffer);
  return { manifest, weightsBuffer };
}
