"""Supervised training of a Sinsan network on teacher shards (P3).

Single GPU per run (pick with --device, default cuda:0), so an architecture ablation can run on
each GPU at once and a run survives shrinking to one GPU. bf16 autocast, torch.compile, AdamW with
warmup + cosine decay, an EMA of the weights (what gets evaluated and exported), periodic
validation, and resumable checkpoints.

Run: cd training && uv run python -m train.trainer --name cnn-b15c256 --arch cnn --channels 256 --blocks 15 \
       --data datasets/shards/nnue100k --steps 200000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import sinsan_engine as se
import torch
import torch.nn.functional as F

from model.sinsan_net import MARGIN_BINS, NetConfig, build_net, count_params
from train.data import Targets, fit_cp_scale, fixed_sample, loader

TRAINING_DIR = Path(__file__).resolve().parents[1]
RUNS = TRAINING_DIR / "runs"

LOSS_WEIGHTS = {"policy": 1.0, "soft": 0.5, "opp": 0.15, "q": 0.5, "wdl": 1.5, "margin": 0.1, "moves_left": 0.05}


def losses(out: dict[str, torch.Tensor], t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    legal = t["legal"]

    def masked_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits.float().masked_fill(~legal, -1e9), 1)
        return -(target * logp.masked_fill(~legal, 0.0)).sum(1).mean()

    opp_ok = t["opp"] >= 0
    opp_logp = torch.log_softmax(out["opp"].float(), 1)
    opp_loss = -(opp_logp.gather(1, t["opp"].clamp(min=0)[:, None]).squeeze(1) * opp_ok).sum() / opp_ok.sum().clamp(min=1)

    q_logits = out["q"].float().gather(1, t["cand"].clamp(min=0))
    q_loss = (F.binary_cross_entropy_with_logits(q_logits, t["q_target"], reduction="none") * t["valid"]).sum() / t["valid"].sum()

    wdl_loss = -(t["wdl"] * torch.log_softmax(out["wdl"].float(), 1)).sum(1).mean()
    margin_loss = -(t["margin"] * torch.log_softmax(out["margin"].float(), 1)).sum(1).mean()
    ml_loss = F.huber_loss(out["moves_left"].float(), t["moves_left"], delta=1.0)
    return {"policy": masked_ce(out["policy"], t["policy"]), "soft": masked_ce(out["soft"], t["soft"]), "opp": opp_loss,
            "q": q_loss, "wdl": wdl_loss, "margin": margin_loss, "moves_left": ml_loss}


@torch.no_grad()
def evaluate(model: torch.nn.Module, val: dict[str, torch.Tensor], targets: Targets, device: torch.device,
             batch: int = 4096) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n = len(val["ply"])
    for start in range(0, n, batch):
        raw = {k: v[start:start + batch].to(device, non_blocking=True) for k, v in val.items()}
        t = targets.build(raw, augment=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(t["boards"], t["scalars"])
        parts = losses(out, t)
        legal = t["legal"]
        pol = out["policy"].float().masked_fill(~legal, -1e9)
        best = t["cand"][:, 0]
        top3 = pol.topk(3, 1).indices
        q_pick = out["q"].float().masked_fill(~legal, -1e9).argmax(1)
        p_win = torch.softmax(out["wdl"].float(), 1)[:, 0]
        centers = targets.margin_centers
        margin_mean = (torch.softmax(out["margin"].float(), 1) * centers).sum(1)
        m = {f"loss_{k}": float(v) * len(best) for k, v in parts.items()}
        m.update({
            "top1": float((pol.argmax(1) == best).sum()),
            "top3": float((top3 == best[:, None]).any(1).sum()),
            "q_top1": float((q_pick == best).sum()),
            "value_brier": float(((p_win - t["outcome_win"]) ** 2).sum()),
            "value_acc": float(((p_win > 0.5) == (t["outcome_win"] > 0.5)).sum()),
            "margin_mae": float((margin_mean - t["margin_value"].clamp(-60, 60)).abs().sum()),
        })
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v
    model.train()
    return {k: v / n for k, v in sums.items()}


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True, help="shard roots containing train/ and val/")
    p.add_argument("--selfplay", type=Path, nargs="*", default=[], help="self-play shard roots (chunk dirs)")
    p.add_argument("--selfplay-frac", type=float, default=0.0, help="fraction of each batch from self-play")
    p.add_argument("--init", type=Path, default=None, help="start from this checkpoint's EMA weights")
    p.add_argument("--arch", default="cnn")
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--blocks", type=int, default=15)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--policy-temp", type=float, default=30.0)
    p.add_argument("--soft-mult", type=float, default=4.0)
    p.add_argument("--value-blend", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--val-size", type=int, default=200_000)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"

    train_roots = [d / "train" for d in args.data]
    val = fixed_sample([d / "val" for d in args.data], args.val_size)
    cfg = NetConfig(arch=args.arch, channels=args.channels, blocks=args.blocks, heads=args.heads, ffn_mult=args.ffn_mult)
    model = build_net(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    decay, no_decay = [], []
    for n_, prm in model.named_parameters():
        (no_decay if prm.ndim <= 1 or "bias" in n_ or "pos" in n_ or "static_bias" in n_ else decay).append(prm)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=True)

    step = 0
    if args.init is not None and not ckpt_path.exists():
        init = torch.load(args.init, map_location=device)
        model.load_state_dict(init["ema"])
        ema.load_state_dict(init["ema"])
        print(f"initialized from {args.init}")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, cp_scale = ck["step"], ck["cp_scale"]
        print(f"resumed {args.name} at step {step}")
    else:
        cp_scale = fit_cp_scale(val["cand_scores"][:, 0], val["result"])
    print(f"{args.name}: {cfg} {count_params(model) / 1e6:.1f}M params; cp_scale={cp_scale:.0f}; "
          f"val {len(val['ply']):,} positions", flush=True)
    (run_dir / "config.json").write_text(json.dumps({"net": asdict(cfg), "args": {k: str(v) for k, v in vars(args).items()},
                                                     "cp_scale": cp_scale, "margin_bins": MARGIN_BINS}, indent=1))

    targets = Targets(se.mirror_table(), cp_scale, args.policy_temp, args.soft_mult, args.value_blend, device)
    fwd = model if args.no_compile else torch.compile(model)
    sp_batch = int(args.batch * args.selfplay_frac) if args.selfplay else 0
    data = iter(loader(train_roots, args.batch - sp_batch, args.workers, seed=step + 1))
    sp_data = iter(loader(args.selfplay, sp_batch, max(2, args.workers // 2), seed=step + 7)) if sp_batch else None
    log = (run_dir / "log.jsonl").open("a")
    t0, seen = time.time(), 0
    running: dict[str, float] = {}
    skipped = 0
    while step < args.steps:
        raw = {k: v.to(device, non_blocking=True) for k, v in next(data).items()}
        t = targets.build(raw, augment=True)
        if sp_data is not None:
            sp_raw = {k: v.to(device, non_blocking=True) for k, v in next(sp_data).items()}
            sp = targets.build(sp_raw, augment=True)
            t = {k: torch.cat([t[k], sp[k]]) for k in t}
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd(t["boards"], t["scalars"])
        parts = losses(out, t)
        loss = sum(LOSS_WEIGHTS[k] * v for k, v in parts.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(grad_norm):
            skipped += 1
            print(f"step {step}: non-finite gradient, skipping update ({skipped} so far)", flush=True)
            if skipped > 50:
                raise RuntimeError("too many non-finite steps; try --no-compile")
            continue
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        step += 1
        seen += args.batch
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + float(v.detach())
        if step % 200 == 0:
            rate = seen / (time.time() - t0)
            msg = " ".join(f"{k}={v / 200:.3f}" for k, v in running.items())
            print(f"step {step} lr={opt.param_groups[0]['lr']:.2e} {rate:,.0f} samples/s | {msg}", flush=True)
            running, seen, t0 = {}, 0, time.time()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(ema, val, targets, device)
            print(f"[eval step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            log.write(json.dumps({"step": step, **metrics}) + "\n")
            log.flush()
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                  "cp_scale": cp_scale, "net": asdict(cfg)}
            torch.save(ck, run_dir / "last.tmp")
            (run_dir / "last.tmp").replace(ckpt_path)
            torch.save({"ema": ema.state_dict(), "net": asdict(cfg), "cp_scale": cp_scale, "step": step},
                       run_dir / f"ema-{step:07d}.pt")


if __name__ == "__main__":
    main()
