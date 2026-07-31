/**
 * Hand-written float32 tensor ops for exactly the shapes SinsanTinyNet needs - not a general
 * tensor library. Matches Moka's verified approach (docs/REFERENCES.md): weights are dequantized
 * to Float32Array once at load, and all arithmetic here is plain float32, never actual int8
 * compute. Kept dependency-free deliberately (Section 15: no mandatory ONNX Runtime, JS/TS first).
 *
 * conv2d does PyTorch's cross-correlation (not flipped-kernel true convolution) to exactly match
 * torch.nn.Conv2d, which training/model/network.py uses.
 */

export interface Tensor3D {
  readonly channels: number;
  readonly height: number;
  readonly width: number;
  readonly data: Float32Array; // [channels, height, width], row-major
}

export function makeTensor3D(channels: number, height: number, width: number): Tensor3D {
  return { channels, height, width, data: new Float32Array(channels * height * width) };
}

/**
 * conv2d does bounds-checked indexing directly against the unpadded input, rather than
 * allocating a zero-padded copy first. A pre-padding version was tried and measured: in
 * isolation, a single 32ch/3x3/10x9 conv dropped from ~5.9ms to ~4.0ms once the inner loop no
 * longer bounds-checked every (y+ky, x+kx) - but that isolated comparison excluded the padding
 * allocation/copy itself from the timed region. Once padding was paid on every real call (as it
 * must be, once per conv, 11 times per forward pass), full-model steady-state inference got
 * *worse* (~57ms -> ~80ms, confirmed across 3 runs): the allocation/copy cost for these small
 * tensors outweighed the saved branches. Kept as bounds-checked; if this needs to be faster
 * later, the right next thing to try is a *reusable scratch buffer* sized once at model
 * construction (paid once, not per call) rather than a fresh allocation per conv2d call.
 */
export function conv2d(
  input: Tensor3D,
  weight: Float32Array, // [outChannels, inChannels, kH, kW], row-major
  bias: Float32Array, // [outChannels]
  outChannels: number,
  kernelSize: number,
  padding: number,
): Tensor3D {
  const { channels: inChannels, height, width, data: inData } = input;
  const out = makeTensor3D(outChannels, height, width);
  const outData = out.data;

  for (let oc = 0; oc < outChannels; oc++) {
    const b = bias[oc]!;
    const outChannelOffset = oc * height * width;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = b;
        for (let ic = 0; ic < inChannels; ic++) {
          const inChannelOffset = ic * height * width;
          const weightChannelOffset = (oc * inChannels + ic) * kernelSize * kernelSize;
          for (let ky = 0; ky < kernelSize; ky++) {
            const iy = y + ky - padding;
            if (iy < 0 || iy >= height) continue;
            const inRowOffset = inChannelOffset + iy * width;
            const weightRowOffset = weightChannelOffset + ky * kernelSize;
            for (let kx = 0; kx < kernelSize; kx++) {
              const ix = x + kx - padding;
              if (ix < 0 || ix >= width) continue;
              sum += inData[inRowOffset + ix]! * weight[weightRowOffset + kx]!;
            }
          }
        }
        outData[outChannelOffset + y * width + x] = sum;
      }
    }
  }
  return out;
}

export function reluInPlace(tensor: Tensor3D): Tensor3D {
  const { data } = tensor;
  for (let i = 0; i < data.length; i++) data[i] = Math.max(0, data[i]!);
  return tensor;
}

export function addInPlace(a: Tensor3D, b: Tensor3D): Tensor3D {
  for (let i = 0; i < a.data.length; i++) a.data[i] = a.data[i]! + b.data[i]!;
  return a;
}

export function globalAveragePool(tensor: Tensor3D): Float32Array {
  const { channels, height, width, data } = tensor;
  const size = height * width;
  const out = new Float32Array(channels);
  for (let c = 0; c < channels; c++) {
    let sum = 0;
    const offset = c * size;
    for (let i = 0; i < size; i++) sum += data[offset + i]!;
    out[c] = sum / size;
  }
  return out;
}

export function linear(input: Float32Array, weight: Float32Array, bias: Float32Array, outFeatures: number): Float32Array {
  const inFeatures = input.length;
  const out = new Float32Array(outFeatures);
  for (let o = 0; o < outFeatures; o++) {
    let sum = bias[o]!;
    const rowOffset = o * inFeatures;
    for (let i = 0; i < inFeatures; i++) sum += input[i]! * weight[rowOffset + i]!;
    out[o] = sum;
  }
  return out;
}

export function reluVectorInPlace(v: Float32Array): Float32Array {
  for (let i = 0; i < v.length; i++) v[i] = Math.max(0, v[i]!);
  return v;
}

export function tanhVector(v: Float32Array): Float32Array {
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = Math.tanh(v[i]!);
  return out;
}
