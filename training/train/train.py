"""Training run for the Tiny Baseline (32x4, smoke tier) or Main Candidate (48x6, Phase 5
baseline) network - same script, different --channels/--blocks/--epochs, per docs/MODEL_DESIGN.md.
Not a strength claim by itself - see docs/BENCHMARK_PLAN.md for what's measured vs. planned.

Also records real, host-measured step timing (seconds/step, samples/sec) rather than an assumed
number, feeding docs/BENCHMARK_PLAN.md's Section 24.2 training-time estimate.

CPU-only by construction (Section 14: "Do not write code that assumes CUDA") - doesn't even check
for a CUDA device, since the official environment is CPU-only Linux Mint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.network import ACTION_SPACE_SIZE, SinsanTinyNet, fen_to_planes  # noqa: E402

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
CHECKPOINT_DIR = Path(__file__).parent.parent / "model" / "checkpoints"
VALIDATION_BUCKET = 0
TEST_BUCKET = 1
SPLIT_BUCKET_COUNT = 10  # matches docs/DATASET_DESIGN.md's game_id % N approach (Moka-derived)


def load_records(dataset_path: Path) -> list[dict]:
    return [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]


def split_by_game(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train, val, test = [], [], []
    for r in records:
        bucket = r["game_id"] % SPLIT_BUCKET_COUNT
        if bucket == VALIDATION_BUCKET:
            val.append(r)
        elif bucket == TEST_BUCKET:
            test.append(r)
        else:
            train.append(r)
    return train, val, test


def build_batch(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.stack(
        [fen_to_planes(r["position"], side_to_move_is_cho=r["position"].split(" ")[1] == "w") for r in records]
    )

    policy_targets = torch.zeros(len(records), ACTION_SPACE_SIZE)
    legal_masks = torch.zeros(len(records), ACTION_SPACE_SIZE, dtype=torch.bool)
    for i, r in enumerate(records):
        for action_id in r["legal_actions"]:
            legal_masks[i, action_id] = True
        for action_id, prob in zip(r["teacher_actions"], r["teacher_policy"]):
            policy_targets[i, action_id] = prob

    value_targets = torch.tensor([r["teacher_value_raw"] for r in records], dtype=torch.float32)
    return inputs, policy_targets, legal_masks, value_targets


def masked_soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy against a soft (sparse) target, normalized over legal actions only (Section
    11.2/14.1: "legal-action-only normalization") - illegal actions get -inf logits before
    softmax so they contribute exactly zero probability mass, not an approximately-small one.

    log_probs is -inf at every illegal position (masked_logits was -inf there). targets is always
    exactly 0 at illegal positions (teacher probability mass only ever lands on legal actions,
    verified against the dataset). Mathematically 0 * -inf should contribute 0 to the sum, but
    IEEE-754 float multiplication gives 0 * -inf = nan, not 0 - discovered the hard way (every
    epoch's policy_loss printed nan despite clean input data). torch.where sidesteps the actual
    multiplication at those positions instead of relying on it evaluating to zero.
    """
    masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=1)
    term = torch.where(targets > 0, targets * log_probs, torch.zeros_like(log_probs))
    return -term.sum(dim=1).mean()


def evaluate(model: SinsanTinyNet, records: list[dict], value_loss_fn: nn.Module, batch_size: int) -> tuple[float, float]:
    """Mean policy/value loss over `records`, batched to bound peak memory on large val/test
    splits rather than building one giant tensor (fine at 512 positions, not at 50,000+)."""
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            inputs, policy_targets, legal_masks, value_targets = build_batch(batch)
            policy_logits, value_pred = model(inputs)
            total_policy_loss += masked_soft_cross_entropy(policy_logits, policy_targets, legal_masks).item()
            total_value_loss += value_loss_fn(value_pred, value_targets).item()
            num_batches += 1
    return total_policy_loss / num_batches, total_value_loss / num_batches


def legal_move_rate(model: SinsanTinyNet, records: list[dict], batch_size: int) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            inputs, _policy_targets, legal_masks, _value_targets = build_batch(batch)
            policy_logits, _value_pred = model(inputs)
            masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
            chosen = masked_logits.argmax(dim=1)
            correct += legal_masks.gather(1, chosen.unsqueeze(1)).sum().item()
    return correct / len(records)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # stdout defaults to fully block-buffered when
    # redirected to a file (not a TTY) - a 20-epoch run's total print output is only a few KB,
    # small enough that the buffer might never auto-flush before the process exits, making a
    # long background run look silent/stuck even when it's progressing normally.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="smoke-labeled.jsonl", help="filename under training/datasets/")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-out", default="tiny-smoke.pt", help="filename under training/model/checkpoints/")
    args = parser.parse_args()

    torch.manual_seed(0)
    records = load_records(DATASETS_DIR / args.dataset)
    train_records, val_records, test_records = split_by_game(records)
    print(f"split: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test (by game_id)")

    model = SinsanTinyNet(channels=args.channels, blocks=args.blocks)
    print(f"SinsanTinyNet(channels={args.channels}, blocks={args.blocks}): {model.parameter_count():,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    value_loss_fn = nn.MSELoss()

    step_times: list[float] = []
    training_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_records))
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0

        for start in range(0, len(train_records), args.batch_size):
            batch_indices = perm[start : start + args.batch_size]
            batch = [train_records[i] for i in batch_indices]
            inputs, policy_targets, legal_masks, value_targets = build_batch(batch)

            step_start = time.perf_counter()
            optimizer.zero_grad()
            policy_logits, value_pred = model(inputs)
            policy_loss = masked_soft_cross_entropy(policy_logits, policy_targets, legal_masks)
            value_loss = value_loss_fn(value_pred, value_targets)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            step_times.append(time.perf_counter() - step_start)

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            num_batches += 1

        val_policy_loss, val_value_loss = evaluate(model, val_records, value_loss_fn, args.batch_size)
        elapsed_min = (time.time() - training_start) / 60
        print(
            f"epoch {epoch + 1}/{args.epochs} ({elapsed_min:.1f} min elapsed): "
            f"train policy_loss={epoch_policy_loss / num_batches:.4f} value_loss={epoch_value_loss / num_batches:.4f} | "
            f"val policy_loss={val_policy_loss:.4f} value_loss={val_value_loss:.4f}"
        )

    # Legal-move rate check on the held-out validation split (Section 18.1: must be 100% after
    # masking - here that just means argmax-over-legal-logits always lands on a legal action,
    # which is true by construction of masked_soft_cross_entropy's -inf masking, but worth
    # asserting rather than assuming).
    legal_rate = legal_move_rate(model, val_records, args.batch_size)
    print(f"legal-move rate on validation split (masked argmax): {legal_rate * 100:.1f}%")
    assert legal_rate == 1.0, "masked argmax picked an illegal action - masking bug"

    test_policy_loss, test_value_loss = evaluate(model, test_records, value_loss_fn, args.batch_size)
    print(f"held-out test split: policy_loss={test_policy_loss:.4f} value_loss={test_value_loss:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_out
    torch.save(model.state_dict(), checkpoint_path)

    step_times_ms = sorted(t * 1000 for t in step_times)
    p50 = step_times_ms[len(step_times_ms) // 2]
    mean_ms = sum(step_times_ms) / len(step_times_ms)
    samples_per_sec = args.batch_size / (mean_ms / 1000)
    total_min = (time.time() - training_start) / 60
    print(
        f"step timing (measured, this host, batch_size={args.batch_size}): "
        f"mean={mean_ms:.1f}ms p50={p50:.1f}ms samples/sec={samples_per_sec:.0f} "
        f"over {len(step_times)} steps, {total_min:.1f} min total"
    )
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
