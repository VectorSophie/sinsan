"""Reinforcement-learning loop (P5): self-play -> train -> gate -> repeat.

Each iteration:
  1. self-play with the current best net on every GPU listed in <run>/gpus.txt (one process per
     GPU; edit the file at any time, e.g. to "0" to free GPU 1 - it is re-read every iteration);
  2. fine-tune the best net on a window of recent self-play chunks mixed with teacher data;
  3. gate: SPRT match new vs best (net_arena, fixed visits); promote the new net only if it wins.

Everything is restartable: the loop state lives in <run>/state.json.

Run: cd training && uv run python -m selfplay.loop --run rl1 --init runs/<p3 run>/ema-XXXXXXX.pt \
        --teacher datasets/shards/nnue100k
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

TRAINING_DIR = Path(__file__).resolve().parents[1]
RUNS = TRAINING_DIR / "runs"
SELFPLAY = TRAINING_DIR / "datasets" / "selfplay"


def gpus(run_dir: Path) -> list[str]:
    f = run_dir / "gpus.txt"
    if not f.exists():
        f.write_text("0,1\n")
    return [g.strip() for g in f.read_text().split(",") if g.strip()]


def sh(cmd: list[str], gpu: str, log: Path) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    return subprocess.Popen(cmd, cwd=TRAINING_DIR, env=env, stdout=log.open("a"), stderr=subprocess.STDOUT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--init", required=True, help="starting checkpoint (EMA weights)")
    p.add_argument("--teacher", type=Path, nargs="+", required=True, help="teacher shard roots (train/ val/)")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--positions", type=int, default=500_000, help="self-play positions per GPU per iteration")
    p.add_argument("--window", type=int, default=40, help="most recent self-play chunks used for training")
    p.add_argument("--train-steps", type=int, default=4000)
    p.add_argument("--selfplay-frac", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gate-visits", type=int, default=128)
    p.add_argument("--gate-pairs", type=int, default=200)
    p.add_argument("--no-gate", action="store_true")
    args = p.parse_args()

    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"iteration": 0, "best": args.init, "history": []}
    sp_root = SELFPLAY / args.run
    log = run_dir / "loop.log"
    py = sys.executable

    while state["iteration"] < args.iterations:
        it = state["iteration"]
        devices = gpus(run_dir)
        print(f"[iteration {it}] best={state['best']} gpus={devices}", flush=True)

        # 1. Self-play on every allowed GPU.
        procs = [sh([py, "-m", "selfplay.selfplay", "--net", state["best"], "--run", f"{args.run}/g{g}",
                     "--positions", str(args.positions), "--seed", str(it * 100 + int(g))], g, log) for g in devices]
        if any(proc.wait() != 0 for proc in procs):
            raise RuntimeError(f"self-play failed; see {log}")

        # 2. Train from the best net on the recent window.
        chunks = sorted(sp_root.glob("g*/c[0-9]*"), key=lambda d: d.stat().st_mtime)[-args.window:]
        window_dir = run_dir / "window"
        if window_dir.exists():
            for link in window_dir.iterdir():
                link.unlink()
        window_dir.mkdir(exist_ok=True)
        for i, c in enumerate(chunks):
            (window_dir / f"c{i:05d}").symlink_to(c.resolve())
        name = f"{args.run}/it{it:04d}"
        net_cfg = torch.load(state["best"], map_location="cpu")["net"]
        train = sh([py, "-m", "train.trainer", "--name", name, "--data", *map(str, args.teacher),
                    "--selfplay", str(window_dir), "--selfplay-frac", str(args.selfplay_frac),
                    "--arch", net_cfg["arch"], "--channels", str(net_cfg["channels"]), "--blocks", str(net_cfg["blocks"]),
                    "--heads", str(net_cfg["heads"]), "--init", state["best"], "--steps", str(args.train_steps),
                    "--lr", str(args.lr), "--warmup", "200", "--eval-every", str(args.train_steps)], devices[0], log)
        if train.wait() != 0:
            raise RuntimeError(f"training failed; see {log}")
        candidate = str(RUNS / name / f"ema-{args.train_steps:07d}.pt")

        # 3. Gate.
        promoted = True
        if not args.no_gate:
            out = subprocess.run([py, "-m", "eval.net_arena", "--a", f"net:{candidate}:{args.gate_visits}",
                                  "--b", f"net:{state['best']}:{args.gate_visits}", "--pairs", str(args.gate_pairs)],
                                 cwd=TRAINING_DIR, env={**os.environ, "CUDA_VISIBLE_DEVICES": devices[0]},
                                 capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if " vs " in l][-1]
            score = float(line.split("(")[1].split("%")[0])
            promoted = score > 50.0
            print(f"[iteration {it}] gate: {line}", flush=True)
        if promoted:
            state["best"] = candidate
        state["history"].append({"iteration": it, "candidate": candidate, "promoted": promoted})
        state["iteration"] = it + 1
        state_path.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
