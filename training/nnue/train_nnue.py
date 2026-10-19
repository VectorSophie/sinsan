"""Trains the small NNUE evaluator for the alpha-beta browser student (F1) and exports it as SNNU
(engine/src/alphabeta.rs). The target is a win probability: a blend of the teacher's evaluation,
sigmoid(cp / 105), and the game outcome. Later this switches to the large net's value (distillation).

Run: cd training && uv run python -m nnue.train_nnue --data datasets/shards/nnue100k --name nnue-h128 --hidden 128
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train.data import fixed_sample, loader

TRAINING_DIR = Path(__file__).resolve().parents[1]
FEATURES = 14 * 90
SCORE_SCALE = 105.0


def feature_indices(boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """boards [B, 4, 90] canonical codes -> (ours idx, theirs idx, offsets) for EmbeddingBag."""
    cur = boards[:, 0].long()  # [B, 90]
    b = cur.shape[0]
    sq = torch.arange(90, device=cur.device).expand(b, 90)
    mask = cur > 0
    ours = (cur - 1) * 90 + sq
    swapped = torch.where(cur <= 7, cur + 7, cur - 7)
    theirs = (swapped - 1) * 90 + (89 - sq)
    counts = mask.sum(1)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    return ours[mask], theirs[mask], offsets


class Nnue(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.h = hidden
        self.l1 = nn.EmbeddingBag(FEATURES, hidden, mode="sum")
        self.l1_b = nn.Parameter(torch.zeros(hidden))
        self.l2 = nn.Linear(2 * hidden, 32)
        self.l3 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
        nn.init.normal_(self.l1.weight, 0, 0.05)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        ours, theirs, offsets = feature_indices(boards)
        a = (self.l1(ours, offsets) + self.l1_b).clamp(0, 1)
        t = (self.l1(theirs, offsets) + self.l1_b).clamp(0, 1)
        x = self.l2(torch.cat([a, t], 1)).clamp(0, 1)
        x = self.l3(x).clamp(0, 1)
        return self.out(x).squeeze(1)

    def export(self, path: Path) -> None:
        arrays = [
            self.l1.weight.detach().cpu().numpy(),  # [1260, h]
            self.l1_b.detach().cpu().numpy(),
            self.l2.weight.detach().cpu().numpy(),  # [32, 2h]
            self.l2.bias.detach().cpu().numpy(),
            self.l3.weight.detach().cpu().numpy(),
            self.l3.bias.detach().cpu().numpy(),
            self.out.weight.detach().cpu().numpy().reshape(32),
            self.out.bias.detach().cpu().numpy(),
        ]
        with path.open("wb") as f:
            f.write(b"SNNU" + struct.pack("<II", 1, self.h))
            for a in arrays:
                f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def target(batch: dict[str, torch.Tensor], blend: float) -> torch.Tensor:
    teacher = torch.sigmoid(batch["cand_scores"][:, 0].float() / SCORE_SCALE)
    outcome = (batch["result"].float() + 1) / 2
    return blend * teacher + (1 - blend) * outcome


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--blend", type=float, default=0.75)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = TRAINING_DIR / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Nnue(args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    val = {k: v.to(device) for k, v in fixed_sample([d / "val" for d in args.data], 100_000).items()}
    data = iter(loader([d / "train" for d in args.data], args.batch, 8, seed=1))
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in next(data).items()}
        loss = F.binary_cross_entropy_with_logits(model(batch["boards"]), target(batch, args.blend))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 5000 == 0 or step == args.steps:
            with torch.no_grad():
                z = model(val["boards"])
                vloss = F.binary_cross_entropy_with_logits(z, target(val, args.blend)).item()
                acc = ((z > 0) == (val["result"] > 0)).float().mean().item()
            print(f"step {step} train {loss.item():.4f} val {vloss:.4f} outcome-acc {acc:.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            model.export(out_dir / "nnue.snnu")
    size = (out_dir / "nnue.snnu").stat().st_size
    print(f"exported {out_dir / 'nnue.snnu'} ({size / 1024:.0f} KiB as f32; ~{size / 2048:.0f} KiB at int16)")


if __name__ == "__main__":
    main()
