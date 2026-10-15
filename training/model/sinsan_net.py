"""Sinsan networks (P3): a KataGo-style ResNet and a 90-token transformer, with identical outputs.

Inputs are the compact canonical encoding from engine/src/features.rs (side to move at the bottom):
  boards  uint8 [B, 4, 90]  piece codes for the current and 3 previous positions
  scalars f32   [B, 10]     repetition, check, bikjang, pass, side, rule50, point margin, ply, ...
`build_planes` turns these into 4*14 + 10 + 1 = 67 input planes on the GPU.

Outputs (dict):
  policy     [B, 5401]  move logits (canonical action ids, see packages/action-space)
  soft       [B, 5401]  auxiliary policy trained on a flatter target (KataGo's soft policy)
  opp        [B, 5401]  auxiliary: the opponent's reply, in the opponent's own frame
  q          [B, 5401]  action-value logits: win probability for the mover after each action
  wdl        [B, 3]     win / draw / loss logits for the side to move
  margin     [B, NB]    final point margin distribution (HL-Gauss bins over MARGIN_RANGE)
  moves_left [B]        predicted plies until the game ends / 100
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

ROWS, COLS, SQUARES = 10, 9, 90
TEMPLATES = 60
ACTIONS = SQUARES * TEMPLATES + 1
PASS = ACTIONS - 1
HISTORY = 4
PIECE_CODES = 14
NUM_SCALARS = 10
IN_PLANES = HISTORY * PIECE_CODES + NUM_SCALARS + 1
MARGIN_BINS = 121
MARGIN_RANGE = 60.0  # bins cover [-60, 60] points


def template_offsets() -> list[tuple[int, int]]:
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        out += [(dr * d, dc * d) for d in range(1, 10)]
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        out += [(dr * d, dc * d) for d in range(1, 3)]
    out += [(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2)]
    out += [(-3, -2), (-3, 2), (3, -2), (3, 2), (-2, -3), (2, -3), (-2, 3), (2, 3)]
    return out


def action_to_square_pair() -> torch.Tensor:
    """[ACTIONS-1] index into a flattened 90x90 (from, to) grid; -1 where the template leaves the board."""
    idx = torch.full((SQUARES * TEMPLATES,), -1, dtype=torch.long)
    for frm in range(SQUARES):
        r, c = divmod(frm, COLS)
        for t, (dr, dc) in enumerate(template_offsets()):
            rr, cc = r + dr, c + dc
            if 0 <= rr < ROWS and 0 <= cc < COLS:
                idx[frm * TEMPLATES + t] = frm * SQUARES + rr * COLS + cc
    return idx


def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish written out explicitly: with torch 2.9's inductor, compiling F.mish (and the fused
    kernels around it) produced NaN gradients in bf16, while this form trains cleanly."""
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)


def build_planes(boards: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
    b = boards.shape[0]
    onehot = F.one_hot(boards.long(), PIECE_CODES + 1)[..., 1:]  # [B, 4, 90, 14]
    planes = onehot.permute(0, 1, 3, 2).reshape(b, HISTORY * PIECE_CODES, ROWS, COLS)
    sc = scalars[:, :, None, None].expand(b, NUM_SCALARS, ROWS, COLS)
    ones = torch.ones(b, 1, ROWS, COLS, device=boards.device)
    return torch.cat([planes.to(scalars.dtype), sc, ones.to(scalars.dtype)], 1)


@dataclass
class NetConfig:
    arch: str = "cnn"  # "cnn" or "transformer"
    channels: int = 256  # CNN width / transformer d_model
    blocks: int = 15  # CNN residual blocks / transformer layers
    heads: int = 8  # transformer only
    ffn_mult: int = 4  # transformer only
    gpool_every: int = 3  # CNN: every n-th block is a global-pooling block
    head_channels: int = 64
    value_hidden: int = 256


# ----------------------------------------------------------------------------------------------
# Shared heads


class ValueHeads(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Linear(features, hidden)
        self.wdl = nn.Linear(hidden, 3)
        self.margin = nn.Linear(hidden, MARGIN_BINS)
        self.moves_left = nn.Linear(hidden, 1)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        h = mish(self.fc(pooled))
        return {"wdl": self.wdl(h), "margin": self.margin(h), "moves_left": F.softplus(self.moves_left(h)).squeeze(-1)}


# ----------------------------------------------------------------------------------------------
# CNN (KataGo-style)


class GlobalPoolBias(nn.Module):
    """Mean and max over the board of `pool_ch` channels -> per-channel bias for the rest."""

    def __init__(self, channels: int, pool_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_ch = pool_ch
        self.bn = nn.BatchNorm2d(pool_ch)
        self.fc = nn.Linear(2 * pool_ch, out_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (remaining channels, their bias, the pooled vector)."""
        g = mish(self.bn(x[:, : self.pool_ch]))
        pooled = torch.cat([g.mean((2, 3)), g.amax((2, 3))], 1)
        return x[:, self.pool_ch :], self.fc(pooled)[:, :, None, None], pooled


class ResBlock(nn.Module):
    def __init__(self, c: int, gpool: bool) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c)
        self.gpool = gpool
        if gpool:
            pool_ch = c // 4
            self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.pool = GlobalPoolBias(c, pool_ch, c - pool_ch)
            self.bn2 = nn.BatchNorm2d(c - pool_ch)
            self.conv2 = nn.Conv2d(c - pool_ch, c, 3, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(c)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        nn.init.zeros_(self.conv2.weight)  # start each block as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(mish(self.bn1(x)))
        if self.gpool:
            h, bias, _ = self.pool(h)
            h = h + bias
        h = self.conv2(mish(self.bn2(h)))
        return x + h


class SinsanCNN(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        c, hc = cfg.channels, cfg.head_channels
        self.stem = nn.Conv2d(IN_PLANES, c, 3, padding=1, bias=False)
        self.blocks = nn.ModuleList([ResBlock(c, (i + 1) % cfg.gpool_every == 0) for i in range(cfg.blocks)])
        self.trunk_bn = nn.BatchNorm2d(c)
        # Policy-like heads: 4 x 60 template maps (policy, soft, opp, q) + 4 pass logits.
        self.p_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.p_pool = GlobalPoolBias(hc, hc // 2, hc - hc // 2)
        self.p_bn = nn.BatchNorm2d(hc - hc // 2)
        self.p_out = nn.Conv2d(hc - hc // 2, 4 * TEMPLATES, 1)
        self.p_pass = nn.Linear(2 * (hc // 2), 4)
        self.v_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(hc)
        self.value = ValueHeads(2 * hc, cfg.value_hidden)

    def forward(self, boards: torch.Tensor, scalars: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(build_planes(boards, scalars))
        for blk in self.blocks:
            x = blk(x)
        x = mish(self.trunk_bn(x))
        b = x.shape[0]

        p_rest, bias, pooled = self.p_pool(self.p_conv(x))
        pass_logits = self.p_pass(pooled)  # [B, 4]
        maps = self.p_out(mish(self.p_bn(p_rest + bias)))  # [B, 240, 10, 9]
        maps = maps.view(b, 4, TEMPLATES, SQUARES).permute(0, 1, 3, 2).reshape(b, 4, SQUARES * TEMPLATES)
        logits = torch.cat([maps, pass_logits[:, :, None]], 2)  # [B, 4, 5401]

        v = mish(self.v_bn(self.v_conv(x)))
        out = self.value(torch.cat([v.mean((2, 3)), v.amax((2, 3))], 1))
        out.update(policy=logits[:, 0], soft=logits[:, 1], opp=logits[:, 2], q=logits[:, 3])
        return out

    def infer(self, boards: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Search-time forward: only the main policy, WDL and margin heads (the auxiliary policy
        maps are three quarters of the head cost and unused in search)."""
        x = self.stem(build_planes(boards, scalars))
        for blk in self.blocks:
            x = blk(x)
        x = mish(self.trunk_bn(x))
        b = x.shape[0]
        p_rest, bias, pooled = self.p_pool(self.p_conv(x))
        pass_logit = F.linear(pooled, self.p_pass.weight[:1], self.p_pass.bias[:1])
        h = mish(self.p_bn(p_rest + bias))
        maps = F.conv2d(h, self.p_out.weight[:TEMPLATES], self.p_out.bias[:TEMPLATES])
        policy = torch.cat([maps.view(b, TEMPLATES, SQUARES).transpose(1, 2).reshape(b, SQUARES * TEMPLATES), pass_logit], 1)
        v = mish(self.v_bn(self.v_conv(x)))
        out = self.value(torch.cat([v.mean((2, 3)), v.amax((2, 3))], 1))
        return policy, out["wdl"], out["margin"]


# ----------------------------------------------------------------------------------------------
# Transformer (Lc0-style, one token per square)


class GeometricAttention(nn.Module):
    """Multi-head self-attention with a learned per-head 90x90 position bias plus a board-dependent
    bias generated from a compressed summary of the position (Lc0's "smolgen")."""

    def __init__(self, d: int, heads: int, smolgen_dim: int = 256) -> None:
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.static_bias = nn.Parameter(torch.zeros(heads, SQUARES, SQUARES))
        self.sg_compress = nn.Linear(d, 32, bias=False)
        self.sg_mid = nn.Linear(32 * SQUARES, smolgen_dim)
        self.sg_heads = nn.Linear(smolgen_dim, heads * smolgen_dim // 4)
        self.sg_proj = nn.Linear(smolgen_dim // 4, SQUARES * SQUARES)
        self.sg_dim = smolgen_dim // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        s = mish(self.sg_mid(self.sg_compress(x).reshape(b, -1)))
        s = mish(self.sg_heads(s)).view(b, self.heads, self.sg_dim)
        bias = self.sg_proj(s).view(b, self.heads, n, n) + self.static_bias
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = GeometricAttention(d, heads)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), Mish(), nn.Linear(ffn_mult * d, d))
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.attn.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class SinsanTransformer(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.channels
        self.embed = nn.Linear(IN_PLANES, d)
        self.pos = nn.Parameter(torch.randn(SQUARES, d) * 0.02)
        self.layers = nn.ModuleList([EncoderLayer(d, cfg.heads, cfg.ffn_mult) for _ in range(cfg.blocks)])
        self.ln = nn.LayerNorm(d)
        # Bilinear from-to heads for policy, soft, opp, q: logits[h, f, t] = <W_f[h] x_f, W_t[h] x_t>.
        self.hq = nn.Linear(d, 4 * d)
        self.hk = nn.Linear(d, 4 * d)
        self.pass_head = nn.Linear(d, 4)
        self.value = ValueHeads(2 * d, cfg.value_hidden)
        self.register_buffer("pair_index", action_to_square_pair(), persistent=False)
        self.d = d

    def forward(self, boards: torch.Tensor, scalars: torch.Tensor) -> dict[str, torch.Tensor]:
        b = boards.shape[0]
        planes = build_planes(boards, scalars)  # [B, C, 10, 9]
        x = self.embed(planes.flatten(2).transpose(1, 2)) + self.pos  # [B, 90, d]
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        q = self.hq(x).view(b, SQUARES, 4, self.d).transpose(1, 2)
        k = self.hk(x).view(b, SQUARES, 4, self.d).transpose(1, 2)
        grid = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)  # [B, 4, 90, 90]
        flat = grid.reshape(b, 4, SQUARES * SQUARES)
        idx = self.pair_index.clamp(min=0)
        acts = flat[:, :, idx].masked_fill(self.pair_index < 0, -1e4)  # [B, 4, 5400]
        pooled = torch.cat([x.mean(1), x.amax(1)], 1)
        pass_logits = self.pass_head(x.mean(1))
        logits = torch.cat([acts, pass_logits[:, :, None]], 2)
        out = self.value(pooled)
        out.update(policy=logits[:, 0], soft=logits[:, 1], opp=logits[:, 2], q=logits[:, 3])
        return out

    def infer(self, boards: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(boards, scalars)
        return out["policy"], out["wdl"], out["margin"]


def build_net(cfg: NetConfig) -> nn.Module:
    if cfg.arch == "cnn":
        return SinsanCNN(cfg)
    if cfg.arch == "transformer":
        return SinsanTransformer(cfg)
    raise ValueError(cfg.arch)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
