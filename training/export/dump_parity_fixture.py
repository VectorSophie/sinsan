"""Dumps a PyTorch forward-pass fixture (input planes + policy logits + value) for a handful of
real dataset positions, so packages/model-runtime's hand-written TypeScript inference can be
numerically cross-checked against the actual PyTorch model that produced the exported weights -
the real correctness guarantee for a from-scratch reimplementation, not just "should match by
construction". Some divergence from INT8 quantization is expected; the fixture stores the
float-checkpoint output so the TS test can apply its own explicit tolerance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.network import SinsanTinyNet, fen_to_planes  # noqa: E402

CHECKPOINT_PATH = Path(__file__).parent.parent / "model" / "checkpoints" / "tiny-smoke.pt"
DATASET_PATH = Path(__file__).parent.parent / "datasets" / "smoke-labeled.jsonl"
OUT_PATH = Path(__file__).parent.parent.parent / "packages" / "model-runtime" / "parity-fixture.json"


def main() -> None:
    model = SinsanTinyNet(channels=32, blocks=4)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    records = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()][:5]

    cases = []
    with torch.no_grad():
        for r in records:
            side_to_move_is_cho = r["position"].split(" ")[1] == "w"
            planes = fen_to_planes(r["position"], side_to_move_is_cho)
            policy_logits, value = model(planes.unsqueeze(0))
            cases.append(
                {
                    "fen": r["position"],
                    "input_planes": planes.reshape(-1).tolist(),
                    "policy_logits": policy_logits.squeeze(0).tolist(),
                    "value": value.item(),
                }
            )

    OUT_PATH.write_text(json.dumps(cases) + "\n")
    print(f"Wrote {len(cases)} parity cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
