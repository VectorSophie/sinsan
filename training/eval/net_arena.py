"""Batched matches for network players (P1/P3/P4 evaluation).

One process drives many concurrent games. Network players search with sinsan_engine.SearchBatch,
evaluating leaves of all their games in one GPU batch per iteration; Fairy-Stockfish opponents
move in a thread pool. The referee is sinsan_engine (FSF janggi rules) and openings come from
eval/openings.jsonl, each played twice with colors swapped.

Player specs:
  net:<checkpoint path>:<visits>   visits=0 plays the raw policy's top move
  fsf:nnue:<nodes> / fsf:classical:<nodes>

Run: cd training && uv run python -m eval.net_arena --a net:runs/x/ema-0100000.pt:0 --b fsf:nnue:1000 --pairs 200
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sinsan_engine as se
import torch

from eval.rating import pentanomial_elo
from eval.uci import fairy_stockfish
from model.sinsan_net import MARGIN_BINS, MARGIN_RANGE, NetConfig, build_net

OPENINGS = Path(__file__).parent / "openings.jsonl"
MAX_PLIES = 600


class NetEvaluator:
    """Loads a checkpoint and evaluates SearchBatch leaves on the GPU.

    margin_utility: weight of the point-margin term added to the value (F2):
    v = P(win) - P(loss) + margin_utility * tanh(E[margin] / 20).
    """

    def __init__(self, path: str, device: torch.device, compile_model: bool = True, margin_utility: float = 0.0) -> None:
        self.margin_utility = margin_utility
        ck = torch.load(path, map_location=device)
        self.model = build_net(NetConfig(**ck["net"])).to(device).eval()
        self.model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        self.device = device
        self.infer = torch.compile(self.model.infer, dynamic=True) if compile_model else self.model.infer

    def reload(self, path: str) -> None:
        """Swaps in another checkpoint of the same architecture without recompiling."""
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])

    @torch.inference_mode()
    def __call__(self, batch: dict) -> tuple[bytes, bytes]:
        n = batch["count"]
        boards = torch.frombuffer(bytearray(batch["boards"]), dtype=torch.uint8).view(n, 4, 90).to(self.device, non_blocking=True)
        scalars = torch.frombuffer(bytearray(batch["scalars"]), dtype=torch.float32).view(n, -1).to(self.device, non_blocking=True)
        actions = torch.frombuffer(bytearray(batch["actions"]), dtype=torch.int32).to(self.device).long()
        offsets = np.frombuffer(batch["offsets"], np.int32)
        rows = torch.repeat_interleave(torch.arange(n, device=self.device),
                                       torch.from_numpy(np.diff(offsets).astype(np.int64)).to(self.device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy, wdl_logits, margin_logits = self.infer(boards, scalars)
        logits = policy.float()[rows, actions]
        wdl = torch.softmax(wdl_logits.float(), 1)
        values = wdl[:, 0] - wdl[:, 2]
        if self.margin_utility:
            centers = torch.linspace(-MARGIN_RANGE, MARGIN_RANGE, MARGIN_BINS, device=self.device)
            expected = (torch.softmax(margin_logits.float(), 1) * centers).sum(1)
            values = (values + self.margin_utility * torch.tanh(expected / 20.0)).clamp(-1, 1)
        return logits.cpu().numpy().tobytes(), values.cpu().numpy().tobytes()


class OnnxEvaluator:
    """Evaluates leaves with an exported browser model (export/export_onnx.py) on the CPU through
    onnxruntime, i.e. exactly the file and arithmetic the web app's in-browser engine uses."""

    def __init__(self, path: str, device: torch.device | None = None, margin_utility: float = 0.0, threads: int = 16) -> None:
        import onnxruntime as ort

        self.margin_utility = margin_utility
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    def __call__(self, batch: dict) -> tuple[bytes, bytes]:
        n = batch["count"]
        boards = np.frombuffer(batch["boards"], np.uint8).reshape(n, 4, 90)
        scalars = np.frombuffer(batch["scalars"], np.float32).reshape(n, -1)
        actions = np.frombuffer(batch["actions"], np.int32)
        offsets = np.frombuffer(batch["offsets"], np.int32)
        policy, wdl, margin = self.session.run(["policy", "wdl", "margin_mean"], {"boards": boards, "scalars": scalars})
        rows = np.repeat(np.arange(n), np.diff(offsets))
        logits = policy[rows, actions].astype(np.float32)
        values = wdl[:, 0] - wdl[:, 2]
        if self.margin_utility:
            values = np.clip(values + self.margin_utility * np.tanh(margin / 20.0), -1, 1)
        return logits.tobytes(), values.astype(np.float32).tobytes()


class NetPlayer:
    """Spec: net:<checkpoint>:<visits>[:key=value,...] with keys cpuct, cpuct_base, cpuct_factor,
    fpu, fpu_root, ptemp (search policy temperature), util (margin utility), leaves, temp and tplies
    (move sampling temperature and the plies it applies to)."""

    SEARCH_KEYS = {"cpuct": "cpuct", "cpuct_base": "cpuct_base", "cpuct_factor": "cpuct_factor", "fpu": "fpu_reduction",
                   "fpu_root": "fpu_reduction_root", "ptemp": "policy_temp"}

    def __init__(self, spec: str, device: torch.device, leaves_per_tree: int = 8, evaluators: dict | None = None) -> None:
        parts = spec.split(":")
        options = {}
        if "=" in parts[-1]:
            options = dict(kv.split("=") for kv in parts[-1].split(","))
            parts = parts[:-1]
        path, visits = ":".join(parts[1:-1]), int(parts[-1])
        self.spec, self.visits = spec, visits
        util = float(options.get("util", 0.0))
        key = (path, util)
        if evaluators is not None and key in evaluators:
            self.evaluate = evaluators[key]
        else:
            evaluator_cls = OnnxEvaluator if path.endswith(".onnx") else NetEvaluator
            self.evaluate = evaluator_cls(path, device, margin_utility=util)
            if evaluators is not None:
                evaluators[key] = self.evaluate
        params = {self.SEARCH_KEYS[k]: float(v) for k, v in options.items() if k in self.SEARCH_KEYS}
        self.batch = se.SearchBatch(**params)
        self.leaves = int(options.get("leaves", leaves_per_tree))
        # Move sampling as in the web app's presets: temp > 0 samples from visits^(1/temp) (or from the
        # priors when visits = 0) during the first tplies plies (default: the whole game).
        self.temp = float(options.get("temp", 0.0))
        self.tplies = int(options.get("tplies", 100_000))
        self.rng = np.random.default_rng()
        self.active: set[int] = set()
        self.synced: dict[int, int] = {}

    def start(self, slot: int, game: dict) -> None:
        """Begins searching for `slot`'s side to move, reusing the slot's tree when it is only a
        couple of moves behind the game (tree reuse across the game)."""
        moves = game["g"].moves()
        synced = self.synced.get(slot)
        behind = moves[synced:] if synced is not None else None
        if behind is not None and len(behind) <= 2 and self.batch.has_slot(slot):
            for m in behind:
                self.batch.advance(slot, m)
        else:
            self.batch.set_root(slot, game["cho"], game["han"], moves)
        self.synced[slot] = len(moves)
        self.active.add(slot)

    def forget(self, slot: int) -> None:
        self.synced.pop(slot, None)
        self.batch.clear_slot(slot)

    def step(self) -> dict[int, str]:
        """One batched search iteration over all active slots; returns finished slots' moves."""
        if not self.active:
            return {}
        # The first iteration only expands roots, so collect one leaf per tree until expanded.
        d = self.batch.collect(self.leaves, sorted(self.active))
        if d["count"]:
            self.batch.apply(*self.evaluate(d))
        done = {}
        for slot in list(self.active):
            if not self.batch.root_expanded(slot):
                continue
            target = max(1, self.visits)
            if self.visits == 0:
                children = self.batch.root_children(slot)
                done[slot] = max(children, key=lambda c: c[3])[0]
            elif self.batch.root_visits(slot) >= target:
                done[slot] = self.batch.best_move(slot)
        if self.temp > 0:
            for slot in done:
                if self.synced[slot] < self.tplies:
                    children = self.batch.root_children(slot)
                    searched = self.visits > 0 and any(c[1] > 0 for c in children)
                    w = np.array([(c[1] if searched else c[3]) for c in children], np.float64) ** (1.0 / self.temp)
                    done[slot] = children[self.rng.choice(len(children), p=w / w.sum())][0]
        for slot, move in done.items():
            self.active.discard(slot)
            self.batch.advance(slot, move)
            self.synced[slot] += 1
        return done


class FsfPool:
    """Fairy-Stockfish opponents: one engine per pool thread, created lazily."""

    # pychess.org's AI levels 0-8 (gbtami/fairyfishnet constants.py): Skill Level, max depth and
    # move time in ms for one thread; NNUE on. Spec "fsf:level:<n>" reproduces level n.
    PYCHESS_SKILL = [-4, 0, 3, 6, 10, 14, 16, 18, 20]
    PYCHESS_DEPTH = [1, 1, 1, 2, 3, 5, 8, 13, 22]
    PYCHESS_MOVETIME = [50, 50, 100, 150, 200, 300, 400, 500, 1000]

    def __init__(self, spec: str, threads: int) -> None:
        _, kind, n = spec.split(":")
        if kind == "level":
            lvl = int(n)
            self.nnue, self.skill = True, self.PYCHESS_SKILL[lvl]
            self.go = f"depth {self.PYCHESS_DEPTH[lvl]} movetime {self.PYCHESS_MOVETIME[lvl]}"
        else:
            self.nnue, self.skill, self.go = kind == "nnue", None, f"nodes {int(n)}"
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(threads)

    def submit(self, game: se.Game) -> Future:
        root, moves, legal = game.root_fen(), game.moves(), game.legal_moves()

        def run() -> str:
            engine = getattr(self.local, "engine", None)
            if engine is None:
                engine = self.local.engine = fairy_stockfish(nnue=self.nnue)
                if self.skill is not None:
                    engine.send(f"setoption name Skill Level value {self.skill}")
                    engine.ready()
            info = engine.search(root, moves, self.go)
            if info.bestmove in legal:
                return info.bestmove
            passes = [m for m in legal if m[: len(m) // 2] == m[len(m) // 2 :]]
            return passes[0] if passes else legal[0]

        return self.pool.submit(run)


class RandomPool:
    """The ladder's random player: a uniformly random legal move, never passing unless forced."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def submit(self, game: se.Game) -> Future:
        legal = game.legal_moves()
        moves = [m for m in legal if m[: len(m) // 2] != m[len(m) // 2 :]] or legal
        f: Future = Future()
        f.set_result(moves[self.rng.integers(len(moves))])
        return f


def play_match(a: str, b: str, pairs: int, device: torch.device, concurrency: int, fsf_threads: int,
               out: Path | None = None, leaves_per_tree: int = 8) -> list[float]:
    openings = [json.loads(l) for l in OPENINGS.read_text().splitlines() if l.strip()]
    players: dict[str, object] = {}
    evaluators: dict = {}  # players sharing a checkpoint share one loaded model
    for spec in {a, b}:
        players[spec] = (NetPlayer(spec, device, leaves_per_tree, evaluators) if spec.startswith("net:")
                         else RandomPool() if spec == "random" else FsfPool(spec, fsf_threads))

    # Queue of (pair, A plays Cho?) games; each slot holds one running game.
    queue = [(p, a_cho) for p in range(pairs) for a_cho in (True, False)]
    games: dict[int, dict] = {}
    pending: dict[int, Future] = {}
    pair_points: dict[int, float] = {}
    pair_games: dict[int, int] = {}
    reasons: dict[str, int] = {}
    sink = out.open("a") if out else None
    next_slot = 0
    start = time.time()

    def launch() -> None:
        nonlocal next_slot
        while queue and len(games) < concurrency:
            p, a_cho = queue.pop(0)
            op = openings[p % len(openings)]
            g = se.Game(cho=op["cho"], han=op["han"])
            for m in op["moves"]:
                g.push(m)
            games[next_slot] = {"pair": p, "a_cho": a_cho, "g": g, "cho": op["cho"], "han": op["han"],
                                "sides": {"w": a if a_cho else b, "b": b if a_cho else a}, "thinking": False}
            next_slot += 1

    def finish(slot: int, result: tuple[str, str] | None) -> None:
        gm = games.pop(slot)
        for p in players.values():
            if isinstance(p, NetPlayer):
                p.forget(slot)
        winner, reason = result if result else (None, "max-plies")
        a_side = "w" if gm["a_cho"] else "b"
        pts = 1.0 if winner == a_side else (0.5 if winner is None else 0.0)
        pair_points[gm["pair"]] = pair_points.get(gm["pair"], 0.0) + pts
        pair_games[gm["pair"]] = pair_games.get(gm["pair"], 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1
        if sink:
            sink.write(json.dumps({"a": a, "b": b, "pair": gm["pair"], "a_cho": gm["a_cho"], "winner": winner,
                                   "reason": reason, "moves": gm["g"].moves()}) + "\n")

    def play(slot: int, move: str) -> None:
        gm = games[slot]
        gm["g"].push(move)
        gm["thinking"] = False
        result = gm["g"].result()
        if result is not None or gm["g"].plies >= MAX_PLIES:
            finish(slot, result)

    launch()
    while games or queue:
        for slot, gm in list(games.items()):
            if gm["thinking"]:
                continue
            player = players[gm["sides"][gm["g"].side_to_move]]
            gm["thinking"] = True
            if isinstance(player, NetPlayer):
                player.start(slot, gm)
            else:
                pending[slot] = player.submit(gm["g"])
        for p in players.values():
            if isinstance(p, NetPlayer):
                for slot, move in p.step().items():
                    play(slot, move)
        for slot, fut in list(pending.items()):
            if fut.done():
                del pending[slot]
                play(slot, fut.result())
        if not any(isinstance(p, NetPlayer) and p.active for p in players.values()) and pending:
            time.sleep(0.002)
        launch()
    if sink:
        sink.close()
    scores = [pair_points[p] for p in sorted(pair_points) if pair_games[p] == 2]
    elo, lo, hi = pentanomial_elo(scores)
    pts = sum(scores)
    print(f"{a} vs {b}: {pts:.1f}/{2 * len(scores)} ({100 * pts / max(1, 2 * len(scores)):.1f}%), "
          f"Elo {elo:+.0f} [{lo:+.0f}, {hi:+.0f}] in {time.time() - start:.0f}s; endings {dict(sorted(reasons.items()))}", flush=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--fsf-threads", type=int, default=64)
    parser.add_argument("--leaves", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    play_match(args.a, args.b, args.pairs, torch.device(args.device), args.concurrency, args.fsf_threads, args.out, args.leaves)


if __name__ == "__main__":
    main()
