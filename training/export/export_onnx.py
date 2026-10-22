"""Exports a transformer checkpoint to ONNX for the browser (apps/web, onnxruntime-web).

Inputs match engine/src/features.rs and the engine's WebAssembly build (WasmSearch.boards() and
.scalars()):
  boards  uint8 [n, 4, 90]
  scalars f32   [n, 10]
Outputs:
  policy       f32 [n, 5401]  move logits over canonical actions
  wdl          f32 [n, 3]     win / draw / loss probabilities for the side to move
  margin_mean  f32 [n]        expected final point margin for the side to move

With --int8 the MatMul weights are quantized (onnxruntime dynamic quantization), which makes the file
about 4x smaller and faster on WebAssembly. Parity against PyTorch is checked on validation positions
and written next to the model as <out>.json, which the web app reads for the model's name and size.

  uv run python -m export.export_onnx --checkpoint runs/student-tf192x6-v3a/ema-0080000.pt \\
      --out ../apps/web/public/student/student.onnx --int8
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model.sinsan_net import MARGIN_BINS, MARGIN_RANGE, NetConfig, build_net, count_params


class BrowserNet(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("centers", torch.linspace(-MARGIN_RANGE, MARGIN_RANGE, MARGIN_BINS))

    def forward(self, boards: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, wdl, margin = self.net.infer(boards, scalars)
        return policy, torch.softmax(wdl, 1), (torch.softmax(margin, 1) * self.centers).sum(1)


def validation_positions(root: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    shard = sorted((root / "val").iterdir())[0]
    boards = np.load(shard / "boards.npy", mmap_mode="r")
    scalars = np.load(shard / "scalars.npy", mmap_mode="r")
    idx = np.random.default_rng(0).choice(len(boards), n, replace=False)
    idx.sort()
    return np.ascontiguousarray(boards[idx]).astype(np.uint8), np.ascontiguousarray(scalars[idx]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--int8", action="store_true", help="quantize MatMul weights to int8")
    p.add_argument("--name", default=None, help="display name (default: the run directory)")
    p.add_argument("--val", type=Path, default=Path("datasets/shards/nnue100k"))
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = NetConfig(**ck["net"])
    if cfg.arch != "transformer":
        raise SystemExit("only transformer checkpoints are supported")
    net = build_net(cfg)
    net.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    model = BrowserNet(net).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = args.out.with_suffix(".fp32.onnx") if args.int8 else args.out
    dummy = (torch.zeros(2, 4, 90, dtype=torch.uint8), torch.zeros(2, 10))
    names = ["policy", "wdl", "margin_mean"]
    torch.onnx.export(model, dummy, fp32, input_names=["boards", "scalars"], output_names=names, opset_version=17,
                      dynamic_axes={k: {0: "n"} for k in ["boards", "scalars", *names]}, dynamo=False)
    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(args.out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        fp32.unlink()

    # Parity on real positions: policy top-1 agreement, policy KL and value error.
    boards, scalars = validation_positions(args.val, 512)
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(boards), torch.from_numpy(scalars))]
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    got = session.run(names, {"boards": boards, "scalars": scalars})
    legal = ref[0] > -1e3

    def log_softmax(x: np.ndarray) -> np.ndarray:
        x = np.where(legal, x, -1e9).astype(np.float64)
        x = x - x.max(1, keepdims=True)
        return x - np.log(np.exp(x).sum(1, keepdims=True))

    lp_ref, lp_got = log_softmax(ref[0]), log_softmax(got[0])
    value_ref, value_got = ref[1][:, 0] - ref[1][:, 2], got[1][:, 0] - got[1][:, 2]
    parity = {
        "top1_agreement": float((lp_ref.argmax(1) == lp_got.argmax(1)).mean()),
        "policy_kl": float((np.exp(lp_ref) * (lp_ref - lp_got)).sum(1).mean()),
        "value_mae": float(np.abs(value_ref - value_got).mean()),
        "margin_mae": float(np.abs(ref[2] - got[2]).mean()),
    }
    data = args.out.read_bytes()
    meta = {
        "name": args.name or args.checkpoint.parent.name,
        "checkpoint": str(args.checkpoint),
        "net": ck["net"],
        "params": count_params(net),
        "int8": args.int8,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "parity": parity,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
