"""Reinforcement-learning loop (P5): continuous self-play, train, gate, repeat.

Self-play workers run continuously, one per GPU listed in <run>/gpus.txt. Edit that file at any
time (e.g. to "0" to free GPU 1); it is re-read every minute and workers are started or stopped to
match. Workers play with the net named in <run>/best.txt and hot-swap it when it changes.

Each iteration:
  1. wait until the workers have written --new-positions recorded positions since the last iteration;
  2. fine-tune the best net on the most recent --window-positions of self-play mixed with teacher
     data, for enough steps that each new self-play position is sampled about --reuse times;
  3. gate: match new vs best (net_arena, fixed visits); promote the new net only if it wins.
Training and gating run on the first GPU in gpus.txt, sharing it with that GPU's worker.

Everything is restartable: the loop state lives in <run>/state.json and stale workers from a
previous driver are stopped on startup.

Run: cd training && uv run python -m selfplay.loop --run rl1 --init runs/<p3 run>/ema-XXXXXXX.pt \\
        --teacher datasets/shards/nnue100k
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parents[1]
RUNS = TRAINING_DIR / "runs"
SELFPLAY = TRAINING_DIR / "datasets" / "selfplay"
BATCH = 2048


def gpus(run_dir: Path) -> list[str]:
    f = run_dir / "gpus.txt"
    if not f.exists():
        f.write_text("0,1\n")
    return [g.strip() for g in f.read_text().split(",") if g.strip()]


def sh(cmd: list[str], gpu: str, log: Path) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    return subprocess.Popen(cmd, cwd=TRAINING_DIR, env=env, stdout=log.open("a"), stderr=subprocess.STDOUT)


def chunk_sizes(sp_root: Path) -> list[tuple[float, Path, int]]:
    """(mtime, path, positions) for every finished self-play chunk, oldest first."""
    out = []
    for c in sp_root.glob("*/c[0-9]*"):
        if c.suffix == ".tmp" or not (c / "ply.npy").exists():
            continue
        out.append((c.stat().st_mtime, c, len(np.load(c / "ply.npy", mmap_mode="r"))))
    return sorted(out)


class Workers:
    """Keeps one self-play process per allowed GPU; pids persist in <run>/workers.json."""

    def __init__(self, run: str, run_dir: Path, log: Path) -> None:
        self.run, self.run_dir, self.log = run, run_dir, log
        self.pid_file = run_dir / "workers.json"
        self.procs: dict[str, subprocess.Popen] = {}
        self.stop_stale()

    def stop_stale(self) -> None:
        if not self.pid_file.exists():
            return
        for pid in json.loads(self.pid_file.read_text()).values():
            try:
                if b"selfplay.selfplay" in Path(f"/proc/{pid}/cmdline").read_bytes():
                    os.kill(pid, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass
        self.pid_file.unlink()

    def sync(self, best: str) -> None:
        wanted = gpus(self.run_dir)
        for g, proc in list(self.procs.items()):
            if g not in wanted or proc.poll() is not None:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                else:
                    print(f"worker on GPU {g} exited with {proc.returncode}; restarting", flush=True)
                del self.procs[g]
        for g in wanted:
            if g not in self.procs:
                self.procs[g] = sh([sys.executable, "-m", "selfplay.selfplay", "--net", best,
                                    "--net-file", str(self.run_dir / "best.txt"), "--run", f"{self.run}/w{g}",
                                    "--positions", str(10**12), "--chunk", "50000", "--seed", str(int(g))], g, self.log)
                print(f"started self-play on GPU {g}", flush=True)
        self.pid_file.write_text(json.dumps({g: p.pid for g, p in self.procs.items()}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--init", required=True, help="starting checkpoint (EMA weights)")
    p.add_argument("--teacher", type=Path, nargs="+", required=True, help="teacher shard roots (train/ val/)")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--new-positions", type=int, default=1_000_000, help="new self-play positions per iteration")
    p.add_argument("--window-positions", type=int, default=8_000_000, help="most recent self-play positions trained on")
    p.add_argument("--reuse", type=float, default=4.0, help="times each new self-play position is sampled in training")
    p.add_argument("--selfplay-frac", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gate-visits", type=int, default=128)
    p.add_argument("--gate-pairs", type=int, default=200)
    p.add_argument("--no-gate", action="store_true")
    args = p.parse_args()

    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"iteration": 0, "best": args.init, "history": []}
    state.setdefault("trained_upto", 0.0)  # mtime of the newest chunk already counted as "new"
    (run_dir / "best.txt").write_text(state["best"] + "\n")
    sp_root = SELFPLAY / args.run
    log = run_dir / "loop.log"
    py = sys.executable
    workers = Workers(args.run, run_dir, log)

    try:
        while state["iteration"] < args.iterations:
            it = state["iteration"]

            # 1. Wait for enough new self-play data, keeping workers in line with gpus.txt.
            while True:
                workers.sync(state["best"])
                chunks = chunk_sizes(sp_root)
                new = sum(n for t, _, n in chunks if t > state["trained_upto"])
                if new >= args.new_positions:
                    break
                time.sleep(60)
            devices = gpus(run_dir)
            print(f"[iteration {it}] best={state['best']} new={new:,} gpus={devices}", flush=True)

            # 2. Train from the best net on the most recent window.
            window, total = [], 0
            for _, c, n in reversed(chunks):
                if total >= args.window_positions:
                    break
                window.append(c)
                total += n
            window_dir = run_dir / "window"
            if window_dir.exists():
                for link in window_dir.iterdir():
                    link.unlink()
            window_dir.mkdir(exist_ok=True)
            for i, c in enumerate(window):
                (window_dir / f"c{i:05d}").symlink_to(c.resolve())
            steps = max(500, math.ceil(new * args.reuse / (BATCH * args.selfplay_frac)))
            name = f"{args.run}/it{it:04d}"
            net_cfg = torch.load(state["best"], map_location="cpu")["net"]
            train = sh([py, "-m", "train.trainer", "--name", name, "--data", *map(str, args.teacher),
                        "--selfplay", str(window_dir), "--selfplay-frac", str(args.selfplay_frac),
                        "--arch", net_cfg["arch"], "--channels", str(net_cfg["channels"]), "--blocks", str(net_cfg["blocks"]),
                        "--heads", str(net_cfg["heads"]), "--init", state["best"], "--steps", str(steps),
                        "--batch", str(BATCH), "--lr", str(args.lr), "--warmup", "200", "--eval-every", str(steps)],
                       devices[0], log)
            while train.poll() is None:
                workers.sync(state["best"])
                time.sleep(60)
            if train.returncode != 0:
                raise RuntimeError(f"training failed; see {log}")
            candidate = str(RUNS / name / f"ema-{steps:07d}.pt")

            # 3. Gate.
            promoted, gate = True, None
            if not args.no_gate:
                out = subprocess.run([py, "-m", "eval.net_arena", "--a", f"net:{candidate}:{args.gate_visits}",
                                      "--b", f"net:{state['best']}:{args.gate_visits}", "--pairs", str(args.gate_pairs)],
                                     cwd=TRAINING_DIR, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus(run_dir)[0]},
                                     capture_output=True, text=True)
                gate = [l for l in out.stdout.splitlines() if " vs " in l][-1]
                promoted = float(gate.split("(")[1].split("%")[0]) > 50.0
                print(f"[iteration {it}] gate: {gate}", flush=True)
            if promoted:
                state["best"] = candidate
                (run_dir / "best.txt").write_text(candidate + "\n")
            state["trained_upto"] = chunks[-1][0]
            state["history"].append({"iteration": it, "candidate": candidate, "promoted": promoted,
                                     "steps": steps, "new_positions": new, "gate": gate})
            state["iteration"] = it + 1
            state_path.write_text(json.dumps(state, indent=1))
    finally:
        for proc in workers.procs.values():
            proc.terminate()
        workers.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
