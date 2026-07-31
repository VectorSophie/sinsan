"""Smoke training run (Phase 4 / Section 23 item 12): a few epochs of the Tiny Baseline network
on the 512-position smoke dataset. Validates the pipeline (data loads, loss decreases, a
checkpoint is produced) - explicitly not a strength claim, per docs/BENCHMARK_PLAN.md.

Also records real, host-measured step timing (seconds/step, samples/sec) rather than an assumed
number, feeding docs/BENCHMARK_PLAN.md's Section 24.2 training-time estimate.

CPU-only by construction (Section 14: "Do not write code that assumes CUDA") - this smoke run
doesn't even check for a CUDA device, since the official environment is CPU-only Linux Mint.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.network import ACTION_SPACE_SIZE, SinsanTinyNet, fen_to_planes  # noqa: E402

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "smoke-labeled.jsonl"
CHECKPOINT_DIR = Path(__file__).parent.parent / "model" / "checkpoints"
VALIDATION_BUCKET = 0
TEST_BUCKET = 1
SPLIT_BUCKET_COUNT = 10  # matches docs/DATASET_DESIGN.md's game_id % N approach (Moka-derived)


def load_records() -> list[dict]:
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


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


def main() -> None:
    torch.manual_seed(0)
    records = load_records()
    train_records, val_records, test_records = split_by_game(records)
    print(f"split: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test (by game_id)")

    model = SinsanTinyNet(channels=32, blocks=4)
    print(f"SinsanTinyNet: {model.parameter_count():,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    value_loss_fn = nn.MSELoss()

    batch_size = 32
    epochs = 5
    step_times: list[float] = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_records))
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0

        for start in range(0, len(train_records), batch_size):
            batch_indices = perm[start : start + batch_size]
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

        print(
            f"epoch {epoch + 1}/{epochs}: policy_loss={epoch_policy_loss / num_batches:.4f} "
            f"value_loss={epoch_value_loss / num_batches:.4f}"
        )

    # Legal-move rate check on the held-out validation split (Section 18.1: must be 100% after
    # masking - here that just means argmax-over-legal-logits always lands on a legal action,
    # which is true by construction of masked_soft_cross_entropy's -inf masking, but worth
    # asserting rather than assuming).
    model.eval()
    with torch.no_grad():
        inputs, _policy_targets, legal_masks, _value_targets = build_batch(val_records)
        policy_logits, _value_pred = model(inputs)
        masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
        chosen = masked_logits.argmax(dim=1)
        legal_rate = legal_masks.gather(1, chosen.unsqueeze(1)).float().mean().item()
    print(f"legal-move rate on validation split (masked argmax): {legal_rate * 100:.1f}%")
    assert legal_rate == 1.0, "masked argmax picked an illegal action - masking bug"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "tiny-smoke.pt"
    torch.save(model.state_dict(), checkpoint_path)

    step_times_ms = sorted(t * 1000 for t in step_times)
    p50 = step_times_ms[len(step_times_ms) // 2]
    mean_ms = sum(step_times_ms) / len(step_times_ms)
    samples_per_sec = batch_size / (mean_ms / 1000)
    print(
        f"step timing (measured, this host, batch_size={batch_size}): "
        f"mean={mean_ms:.1f}ms p50={p50:.1f}ms samples/sec={samples_per_sec:.0f} "
        f"over {len(step_times)} steps"
    )
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
