"""Tiny Baseline network (docs/MODEL_DESIGN.md): 32 channels x 4 residual blocks. Smoke-tier
model whose job is to validate the generate -> label -> train -> export -> infer pipeline, not to
make a strength claim (Section 23 Phase 4 / docs/BENCHMARK_PLAN.md).

Input representation - a deliberate, disclosed simplification for the smoke tier: docs/
MODEL_DESIGN.md's *preferred* default is a canonicalized (always-my-perspective) board, but
correctly implementing that also requires a matching 180-degree action-space rotation (mirroring
packages/action-space's existing horizontal reflection, but untested) - real extra scope and a
real place to introduce a subtle input/label misalignment bug. For this smoke milestone, whose
purpose is proving the pipeline end-to-end, this module instead uses an ABSOLUTE board encoding
(fixed Cho/Han planes, not perspective-relative) plus an explicit side-to-move plane - zero risk
of desyncing from the action space, since both already share the same absolute coordinate system.
Canonicalization is deferred to a real Phase 5 ablation, as docs/MODEL_DESIGN.md already flags.

No BatchNorm - the smoke tier skips it entirely rather than adding fold-at-export machinery that
isn't needed to prove the pipeline works (Section 11 requires folding BN into convs at export
*if* BN is used - simplest to not use it yet).
"""

from __future__ import annotations

import torch
from torch import nn

BOARD_ROWS = 10
BOARD_COLS = 9
PIECE_TYPES = ["chariot", "cannon", "horse", "elephant", "guard", "general", "soldier"]
NUM_PIECE_PLANES = len(PIECE_TYPES) * 2  # 14: 7 types x {cho, han}
NUM_INPUT_PLANES = NUM_PIECE_PLANES + 1  # +1 side-to-move plane
POLICY_TEMPLATE_COUNT = 60
POLICY_BOARD_ACTIONS = BOARD_ROWS * BOARD_COLS * POLICY_TEMPLATE_COUNT  # 5400
ACTION_SPACE_SIZE = POLICY_BOARD_ACTIONS + 1  # +1 pass, matches packages/action-space


def fen_to_planes(fen: str, side_to_move_is_cho: bool) -> torch.Tensor:
    """FEN board part (e.g. "rnba1abnr/4k4/.../RNBA1ABNR") -> [NUM_INPUT_PLANES, 10, 9] float
    tensor. Absolute encoding (Section 10 fallback, see module docstring): plane order is fixed
    [cho x 7 piece types, han x 7 piece types, side-to-move], independent of whose turn it is."""
    board_part = fen.split(" ")[0]
    planes = torch.zeros(NUM_INPUT_PLANES, BOARD_ROWS, BOARD_COLS)
    letter_to_type_index = {"r": 0, "c": 1, "n": 2, "b": 3, "a": 4, "k": 5, "p": 6}

    row = 0
    col = 0
    for ch in board_part:
        if ch == "/":
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            type_index = letter_to_type_index[ch.lower()]
            side_offset = 0 if ch.isupper() else len(PIECE_TYPES)  # uppercase = cho
            planes[side_offset + type_index, row, col] = 1.0
            col += 1

    if side_to_move_is_cho:
        planes[NUM_INPUT_PLANES - 1, :, :] = 1.0
    return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.relu(x + residual)


class SinsanTinyNet(nn.Module):
    """32 channels x 4 residual blocks (docs/MODEL_DESIGN.md's Tiny Baseline candidate)."""

    def __init__(self, channels: int = 32, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(NUM_INPUT_PLANES, channels, kernel_size=3, padding=1)
        self.tower = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, POLICY_TEMPLATE_COUNT, kernel_size=3, padding=1)
        self.pass_head = nn.Linear(channels, 1)

        self.value_conv = nn.Conv2d(channels, 4, kernel_size=1)
        self.value_fc1 = nn.Linear(4 * BOARD_ROWS * BOARD_COLS, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.stem(x))
        for block in self.tower:
            x = block(x)

        # Policy: [batch, 60, 10, 9] -> permute to [batch, row, col, template] -> flatten so that
        # index == row*9*60 + col*60 + template == (row*9+col)*60 + template == origin*60+template,
        # matching packages/action-space's encodeMove() exactly (from*TEMPLATE_COUNT+templateIndex).
        policy_board = self.policy_conv(x).permute(0, 2, 3, 1).reshape(x.shape[0], POLICY_BOARD_ACTIONS)
        pooled = x.mean(dim=(2, 3))
        pass_logit = self.pass_head(pooled)
        policy_logits = torch.cat([policy_board, pass_logit], dim=1)  # [batch, 5401]

        v = torch.relu(self.value_conv(x))
        v = v.reshape(v.shape[0], -1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)  # [batch], current-player perspective

        return policy_logits, value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
