import type { LayerManifest, ModelManifest } from './manifest.ts';
import {
  addInPlace,
  conv2d,
  globalAveragePool,
  linear,
  reluInPlace,
  reluVectorInPlace,
  tanhVector,
  type Tensor3D,
} from './tensor-ops.ts';

interface DequantizedLayer {
  readonly weight: Float32Array;
  readonly bias: Float32Array;
  readonly outChannels: number;
  readonly padding: number;
}

function dequantizeLayer(weightsView: DataView, manifest: LayerManifest): DequantizedLayer {
  const outChannels = manifest.weight_shape[0]!;
  const perChannelElements = manifest.weight_bytes / outChannels;

  const int8 = new Int8Array(weightsView.buffer, weightsView.byteOffset + manifest.weight_offset, manifest.weight_bytes);
  const scales = new Float32Array(
    weightsView.buffer,
    weightsView.byteOffset + manifest.scale_offset,
    outChannels,
  );

  const weight = new Float32Array(int8.length);
  for (let oc = 0; oc < outChannels; oc++) {
    const scale = scales[oc]!;
    const offset = oc * perChannelElements;
    for (let i = 0; i < perChannelElements; i++) weight[offset + i] = int8[offset + i]! * scale;
  }

  const bias = new Float32Array(weightsView.buffer.slice(weightsView.byteOffset + manifest.bias_offset, weightsView.byteOffset + manifest.bias_offset + outChannels * 4));

  return { weight, bias, outChannels, padding: manifest.padding ?? 0 };
}

export interface InferenceResult {
  /** Raw policy logits, length ACTION_SPACE_SIZE (board actions in origin*60+template order,
   * followed by the pass logit as the final element) - matches training/model/network.py's
   * torch.cat([policy_board, pass_logit], dim=1) exactly, and packages/action-space's PASS_ACTION
   * index. Caller applies legal-action masking + softmax, not this module (keeps this module a
   * pure numeric forward pass, matching Section 15's "no tensor computation on the main thread"
   * boundary rather than mixing in game-rule concerns). */
  readonly policyLogits: Float32Array;
  /** Current-player-perspective value in [-1, 1] (docs/MODEL_DESIGN.md's value head convention). */
  readonly value: number;
}

export class SinsanModel {
  private readonly layers: Map<string, DequantizedLayer>;
  private readonly blocks: number;
  private readonly inputShape: readonly [number, number, number];

  constructor(manifest: ModelManifest, weightsBuffer: ArrayBuffer) {
    const view = new DataView(weightsBuffer);
    this.layers = new Map(manifest.layers.map((l) => [l.name, dequantizeLayer(view, l)]));
    this.blocks = manifest.architecture.blocks;
    this.inputShape = manifest.input_shape;
  }

  private layer(name: string): DequantizedLayer {
    const l = this.layers.get(name);
    if (!l) throw new Error(`SinsanModel: missing layer '${name}' in weights`);
    return l;
  }

  private convForward(input: Tensor3D, layerName: string, kernelSize: number): Tensor3D {
    const l = this.layer(layerName);
    return conv2d(input, l.weight, l.bias, l.outChannels, kernelSize, l.padding);
  }

  infer(inputPlanes: Float32Array): InferenceResult {
    const [channels, height, width] = this.inputShape;
    if (inputPlanes.length !== channels * height * width) {
      throw new Error(`SinsanModel.infer: expected ${channels * height * width} input values, got ${inputPlanes.length}`);
    }

    let x: Tensor3D = { channels, height, width, data: inputPlanes };
    x = reluInPlace(this.convForward(x, 'stem', 3));

    for (let b = 0; b < this.blocks; b++) {
      const residual = x;
      let h = reluInPlace(this.convForward(x, `tower.${b}.conv1`, 3));
      h = this.convForward(h, `tower.${b}.conv2`, 3);
      x = reluInPlace(addInPlace(h, residual));
    }

    // Policy: conv -> [templateCount, H, W] -> permute to [H, W, templateCount] -> flatten, so
    // index == row*width*templateCount + col*templateCount + template ==
    // (row*width+col)*templateCount + template == origin*TEMPLATE_COUNT + templateIndex,
    // matching packages/action-space's encodeMove() and training/model/network.py's permute
    // exactly (same reasoning, both sides).
    const policyTensor = this.convForward(x, 'policy_conv', 3);
    const templateCount = policyTensor.channels;
    const boardActions = height * width * templateCount;
    const policyBoard = new Float32Array(boardActions);
    for (let c = 0; c < templateCount; c++) {
      for (let y = 0; y < height; y++) {
        for (let col = 0; col < width; col++) {
          const src = (c * height + y) * width + col;
          const dst = (y * width + col) * templateCount + c;
          policyBoard[dst] = policyTensor.data[src]!;
        }
      }
    }

    const pooled = globalAveragePool(x);
    const passHead = this.layer('pass_head');
    const passLogit = linear(pooled, passHead.weight, passHead.bias, 1)[0]!;

    const policyLogits = new Float32Array(boardActions + 1);
    policyLogits.set(policyBoard, 0);
    policyLogits[boardActions] = passLogit;

    const valueConv = reluInPlace(this.convForward(x, 'value_conv', 1));
    const valueFlat = valueConv.data;
    const fc1 = this.layer('value_fc1');
    const h1 = reluVectorInPlace(linear(valueFlat, fc1.weight, fc1.bias, fc1.outChannels));
    const fc2 = this.layer('value_fc2');
    const h2 = linear(h1, fc2.weight, fc2.bias, 1);
    const value = tanhVector(h2)[0]!;

    return { policyLogits, value };
  }
}
