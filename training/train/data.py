"""Training data: random positions sampled across memory-mapped shards (data/shards.py), plus
GPU-side target construction and left-right mirror augmentation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from model.sinsan_net import ACTIONS, MARGIN_BINS, MARGIN_RANGE

FIELDS = ("boards", "scalars", "legal", "cand_actions", "cand_scores", "result", "margin", "plies_left", "game", "ply")


class ShardSet:
    """Teacher shards, or self-play shards (which add pol_probs: an improved-policy target over
    the 64 cand_actions)."""

    def __init__(self, roots: list[Path]) -> None:
        self.shards = []
        dirs = [d for root in roots for d in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.endswith(".tmp"))]
        self.fields = FIELDS + (("pol_probs",) if dirs and (dirs[0] / "pol_probs.npy").exists() else ())
        for d in dirs:
            self.shards.append({f: np.load(d / f"{f}.npy", mmap_mode="r") for f in self.fields})
        if not self.shards:
            raise FileNotFoundError(f"no shards under {roots}")
        sizes = np.array([len(s["ply"]) for s in self.shards])
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.size = int(self.offsets[-1])

    def gather(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.sort(idx)
        shard_of = np.searchsorted(self.offsets, idx, side="right") - 1
        parts: dict[str, list[np.ndarray]] = {f: [] for f in self.fields}
        parts["opp_action"] = []
        for s in np.unique(shard_of):
            rows = idx[shard_of == s] - self.offsets[s]
            sh = self.shards[s]
            for f in self.fields:
                parts[f].append(np.asarray(sh[f][rows]))
            # Opponent's reply target: the teacher's best move in the next position of the same game.
            nxt = np.minimum(rows + 1, len(sh["ply"]) - 1)
            same = (sh["game"][nxt] == sh["game"][rows]) & (sh["ply"][nxt] == sh["ply"][rows] + 1)
            parts["opp_action"].append(np.where(same, sh["cand_actions"][nxt, 0], -1).astype(np.int16))
        return {f: np.concatenate(v) for f, v in parts.items()}


class RandomBatches(IterableDataset):
    def __init__(self, roots: list[Path], batch_size: int, seed: int) -> None:
        self.roots, self.batch_size, self.seed = roots, batch_size, seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        data = ShardSet(self.roots)
        rng = np.random.default_rng(self.seed * 1000 + wid)
        while True:
            batch = data.gather(rng.integers(0, data.size, self.batch_size))
            yield {k: torch.from_numpy(v) for k, v in batch.items()}


def loader(roots: list[Path], batch_size: int, workers: int, seed: int) -> DataLoader:
    return DataLoader(RandomBatches(roots, batch_size, seed), batch_size=None, num_workers=workers,
                      pin_memory=True, prefetch_factor=4, persistent_workers=True)


def fixed_sample(roots: list[Path], n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    data = ShardSet(roots)
    rng = np.random.default_rng(seed)
    batch = data.gather(rng.choice(data.size, min(n, data.size), replace=False))
    return {k: torch.from_numpy(v) for k, v in batch.items()}


# ----------------------------------------------------------------------------------------------
# GPU-side targets


class Targets:
    """Builds training targets from raw batch fields on the GPU.

    cp_scale: centipawns -> win probability, p = sigmoid(cp / cp_scale), fitted on the data.
    policy_temp: candidate scores -> policy target, softmax(cp / policy_temp).
    value_blend: weight of the teacher's evaluation (vs the game outcome) in the value target.
    """

    def __init__(self, mirror_table: list[int], cp_scale: float, policy_temp: float, soft_mult: float,
                 value_blend: float, device: torch.device) -> None:
        self.mirror = torch.tensor(mirror_table, dtype=torch.long, device=device)
        self.cp_scale, self.policy_temp, self.soft_mult, self.value_blend = cp_scale, policy_temp, soft_mult, value_blend
        centers = torch.linspace(-MARGIN_RANGE, MARGIN_RANGE, MARGIN_BINS, device=device)
        self.margin_edges = torch.cat([centers - (centers[1] - centers[0]) / 2, centers[-1:] + (centers[1] - centers[0]) / 2])
        self.margin_centers = centers
        self.margin_sigma = 0.75 * float(centers[1] - centers[0])
        self.bit_weights = (1 << torch.arange(8, device=device)).to(torch.uint8)

    def unpack_legal(self, packed: torch.Tensor) -> torch.Tensor:
        bits = (packed.unsqueeze(-1) & self.bit_weights).ne(0)
        return bits.flatten(1)[:, :ACTIONS]

    def build(self, batch: dict[str, torch.Tensor], augment: bool) -> dict[str, torch.Tensor]:
        boards = batch["boards"]
        legal = self.unpack_legal(batch["legal"])
        cand = batch["cand_actions"].long()
        opp = batch["opp_action"].long()
        if augment:
            flip = torch.rand(boards.shape[0], device=boards.device) < 0.5
            mirrored = boards.view(-1, 4, 10, 9).flip(-1).reshape(boards.shape)
            boards = torch.where(flip[:, None, None], mirrored, boards)
            legal = torch.where(flip[:, None], legal[:, self.mirror], legal)  # mirror is an involution
            mcand = torch.where(cand >= 0, self.mirror[cand.clamp(min=0)], cand)
            cand = torch.where(flip[:, None], mcand, cand)
            mopp = torch.where(opp >= 0, self.mirror[opp.clamp(min=0)], opp)
            opp = torch.where(flip, mopp, opp)

        b = cand.shape[0]
        selfplay = "pol_probs" in batch
        if selfplay:
            probs = batch["pol_probs"].float().masked_fill(cand < 0, 0.0)
            policy = torch.zeros(b, ACTIONS, device=cand.device).scatter_add_(1, cand.clamp(min=0), probs)
            sharp = probs.pow(1.0 / self.soft_mult)
            soft = torch.zeros(b, ACTIONS, device=cand.device).scatter_add_(
                1, cand.clamp(min=0), sharp / sharp.sum(1, keepdim=True).clamp(min=1e-12))
            cand = cand[:, :8]
        scores = batch["cand_scores"].float()
        valid = (cand >= 0) & (not selfplay)


        def dist(temp: float) -> torch.Tensor:
            z = (scores / temp).masked_fill(~valid, -1e9)
            p = torch.softmax(z, 1).masked_fill(~valid, 0.0)
            out = torch.zeros(b, ACTIONS, device=cand.device)
            return out.scatter_add_(1, cand.clamp(min=0), p)

        if not selfplay:
            policy = dist(self.policy_temp)
            soft = dist(self.policy_temp * self.soft_mult)
        q_target = torch.sigmoid(scores / self.cp_scale)

        result = batch["result"].float()
        teacher_win = torch.sigmoid(scores[:, 0] / self.cp_scale)
        outcome_win = (result + 1) / 2
        blend = 0.0 if selfplay else self.value_blend  # self-play has no teacher evaluation
        win = (1 - blend) * outcome_win + blend * teacher_win
        wdl = torch.stack([win, torch.zeros_like(win), 1 - win], 1)

        # HL-Gauss: probability mass of N(margin, sigma) in each bin.
        m = batch["margin"].float().clamp(-MARGIN_RANGE, MARGIN_RANGE)
        cdf = torch.special.ndtr((self.margin_edges[None, :] - m[:, None]) / self.margin_sigma)
        margin = cdf[:, 1:] - cdf[:, :-1]
        margin = margin / margin.sum(1, keepdim=True)

        return {
            "boards": boards, "scalars": batch["scalars"], "legal": legal, "policy": policy, "soft": soft,
            "cand": cand, "valid": valid, "q_target": q_target, "opp": opp, "wdl": wdl,
            "outcome_win": outcome_win, "margin": margin, "margin_value": batch["margin"].float(),
            "moves_left": batch["plies_left"].float() / 100.0,
        }


def fit_cp_scale(scores: torch.Tensor, result: torch.Tensor) -> float:
    """Fits p(win) = sigmoid(cp / s) to game outcomes by 1-D maximum likelihood (grid + refine)."""
    cp = scores.float().clamp(-3000, 3000)
    y = (result.float() + 1) / 2
    best, best_s = math.inf, 400.0
    for s in np.geomspace(50, 3000, 200):
        p = torch.sigmoid(cp / float(s)).clamp(1e-6, 1 - 1e-6)
        nll = float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())
        if nll < best:
            best, best_s = nll, float(s)
    return best_s
