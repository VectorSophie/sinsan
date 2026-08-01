"""INT8 export: symmetric per-output-channel, weight-only quantization - the pattern verified in
Moka's own export code during Phase 0 research (docs/REFERENCES.md): `scale =
max(abs(output_channel_weights)) / 127`, biases stay float32, activations are never quantized
(the browser runtime dequantizes once at load and runs plain float32 arithmetic - see
packages/model-runtime). Same script for the smoke (32x4) and baseline (48x6) tiers - pass
--channels/--blocks/--checkpoint/--model-name matching the training run.

Produces two files consumed by the browser runtime:
  - <name>.bin: concatenated int8 weight bytes + fp32 scale/bias bytes, per the manifest's offsets
  - <name>.json: manifest describing every layer's shape and byte offsets, plus a sha256 of the
    weights blob for the pre-load integrity check in packages/model-runtime (Moka's verified
    pattern, adopted directly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.network import ACTION_SPACE_SIZE, NUM_INPUT_PLANES, BOARD_COLS, BOARD_ROWS, SinsanTinyNet  # noqa: E402

CHECKPOINT_DIR = Path(__file__).parent.parent / "model" / "checkpoints"
OUT_DIR = Path(__file__).parent.parent.parent / "public" / "model"
FORMAT_VERSION = 1


def quantize_per_output_channel(weight: torch.Tensor) -> tuple[bytes, list[float]]:
    """weight: [out_channels, ...]. Returns (int8 bytes in row-major out-channel order, one
    fp32 scale per output channel). Uses struct.pack rather than tensor.numpy() - avoids adding
    numpy as a dependency just to serialize a small int8 buffer."""
    out_channels = weight.shape[0]
    flat = weight.reshape(out_channels, -1)
    scales = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    quantized = torch.round(flat / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    flat_values = quantized.reshape(-1).tolist()
    return struct.pack(f"<{len(flat_values)}b", *flat_values), scales.tolist()


def export(
    checkpoint_path: Path,
    channels: int,
    blocks: int,
    model_name: str,
    training_run: str,
    out_dir: Path = OUT_DIR,
) -> None:
    model = SinsanTinyNet(channels=channels, blocks=blocks)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    weight_blob = bytearray()
    layers: list[dict] = []

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight_bytes, scales = quantize_per_output_channel(module.weight.data)
            weight_offset = len(weight_blob)
            weight_blob.extend(weight_bytes)

            scale_offset = len(weight_blob)
            weight_blob.extend(struct.pack(f"<{len(scales)}f", *scales))

            bias = module.bias.data.tolist() if module.bias is not None else [0.0] * module.weight.shape[0]
            bias_offset = len(weight_blob)
            weight_blob.extend(struct.pack(f"<{len(bias)}f", *bias))

            layer_entry: dict = {
                "name": name,
                "type": "conv2d" if isinstance(module, torch.nn.Conv2d) else "linear",
                "weight_shape": list(module.weight.shape),
                "weight_offset": weight_offset,
                "weight_bytes": len(weight_bytes),
                "scale_offset": scale_offset,
                "bias_offset": bias_offset,
            }
            if isinstance(module, torch.nn.Conv2d):
                layer_entry["padding"] = module.padding[0]
            layers.append(layer_entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{model_name}.bin"
    bin_path.write_bytes(bytes(weight_blob))
    sha256 = hashlib.sha256(bytes(weight_blob)).hexdigest()

    manifest = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "architecture": {"channels": channels, "blocks": blocks, "kind": "tiny-baseline" if channels <= 32 else "main-candidate"},
        "input_shape": [NUM_INPUT_PLANES, BOARD_ROWS, BOARD_COLS],
        "policy_shape": [ACTION_SPACE_SIZE],
        "parameter_count": model.parameter_count(),
        "quantization": "int8-per-output-channel-symmetric-weight-only",
        "weights_bytes": len(weight_blob),
        "sha256": sha256,
        "training_run": training_run,
        "rule_profile": "kja",
        "layers": layers,
    }
    manifest_path = out_dir / f"{model_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"parameters: {model.parameter_count():,}")
    print(f"weights blob: {len(weight_blob):,} bytes ({len(weight_blob) / 1024:.1f} KiB)")
    budget_kib = 480
    if len(weight_blob) / 1024 > budget_kib:
        print(f"WARNING: exceeds the {budget_kib}KiB weight budget (Section 2)")
    print(f"sha256: {sha256}")
    print(f"wrote {bin_path} and {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="tiny-smoke.pt", help="filename under training/model/checkpoints/")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--model-name", default="sinsan-smoke-v0")
    parser.add_argument("--training-run", default="phase4-smoke")
    args = parser.parse_args()
    export(
        checkpoint_path=CHECKPOINT_DIR / args.checkpoint,
        channels=args.channels,
        blocks=args.blocks,
        model_name=args.model_name,
        training_run=args.training_run,
    )


if __name__ == "__main__":
    main()
