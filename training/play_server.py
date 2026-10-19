"""Local HTTP endpoint so the web app (apps/web) can play against full-size checkpoints.

GET  /api/nets  -> [{"id", "label", "path"}]: the networks a request may choose from.
POST /api/fsf   {"cho", "han", "moves", "nodes", "nnue": bool}
             -> {"move", "cp" (side to move), "mate", "depth", "nodes", "pv", "ms"}: Fairy-Stockfish,
                single thread and a fixed node budget, as on the strength ladder.
GET  /api/progress -> {"ladder": [...], "nets": [...], "rl": [...], "selfplay": [...]}: Fairy-Stockfish
                 ladder anchors, Sinsan nets rated from their saved matches against it, the RL loop's
                 gate history and the self-play workers' latest progress lines.
POST /api/move  {"cho": formation, "han": formation, "moves": [uci...], plus optional settings:
                 "net" (id from /api/nets), "visits", "time_ms" (stop at whichever limit comes first),
                 "cpuct", "fpu", "ptemp" (search policy temperature), "util" (point-margin utility),
                 "temperature" (0 = play the most-visited move; >0 samples from visits^(1/T)),
                 "temp_plies" (sample only while fewer than this many plies have been played)}
             -> {"move", "best" (most-visited move), "cp", "value" (search value for the side to move,
                 -1..1), "net_value" (raw network), "visits", "ms", "depth", "seldepth", "pv",
                 "candidates": [{move, visits, q, prior}],
                 "tree": [[parent row, depth, move, visits, q, prior, terminal], ...]}
                Each q is from the point of view of the side that played that move.

The "best" network follows --net-file (e.g. runs/rl1/best.txt), re-read on every request. Requests
are handled one at a time.

Run: cd training && CUDA_VISIBLE_DEVICES=1 uv run python play_server.py --net-file runs/rl1/best.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import sinsan_engine as se
import torch

from eval.net_arena import NetEvaluator
from eval.uci import fairy_stockfish
from uci_engine import Engine, value_to_cp

TRAINING_DIR = Path(__file__).resolve().parent
RESULTS = TRAINING_DIR / "eval" / "results"
RL_RUN = TRAINING_DIR / "runs" / "rl1"
# Saved match files (eval/results/<prefix>-v800-fsf_*.json) for nets shown on the progress page.
RATED_NETS = [
    ("abl-cnn256x15", "CNN 17M (9M positions)"),
    ("abl-tf384x10", "Transformer 35M (9M positions)"),
    ("p3", "Transformer 60M, supervised (36M positions)"),
    ("rl1-it2", "After RL round 2"),
    ("rl1-it9", "After RL round 9"),
]
MAX_VISITS = 50_000
MAX_TIME_MS = 60_000
DEFAULTS = {"cpuct": 1.745, "fpu": 0.33, "ptemp": 1.36}
# Fixed reference networks offered next to the live RL net (ladder ratings vs FSF, docs/REBUILD_LOG.md).
REFERENCE_NETS = [
    ("p3", "Before RL (60M supervised, ~+2250)", "runs/p3-tf512x12/ema-0052000.pt"),
    ("small", "Small (35M, 9M positions, ~+1750)", "runs/abl-tf384x10/ema-0014000.pt"),
]


def match_rating(f: Path, anchors: dict[str, float]) -> dict | None:
    """Rating implied by a saved match (one JSON game per line) against a ladder anchor."""
    games = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if not games or games[0]["b"] not in anchors:
        return None
    opp = games[0]["b"]
    # winner is a colour: "w" = Cho, "b" = Han; a_cho says which colour player a had.
    score = sum(1.0 if (g["winner"] == "w") == g["a_cho"] else 0.0 for g in games) / len(games)
    score = min(max(score, 0.5 / len(games)), 1 - 0.5 / len(games))
    return {"opponent": opp, "games": len(games), "score": score,
            "elo": round(anchors[opp] - 400 * math.log10(1 / score - 1))}


def progress() -> dict:
    """Everything the web app's progress page shows, computed from files on disk."""
    anchors = {k: v[0] for k, v in json.loads((RESULTS / "ladder-ratings.json").read_text()).items()}
    names = {"random": "Random moves", "oldnet:baseline-v3-56x7:56x7:0": "Original Sinsan V3, no search",
             "oldnet:baseline-v3-56x7:56x7:16": "Original Sinsan V3, 16 visits"}
    ladder = []
    for spec, elo in anchors.items():
        parts = spec.split(":")
        label = names.get(spec) or (f"Fairy-Stockfish {'NNUE' if parts[1] == 'nnue' else 'classical'} "
                                    f"{int(parts[2]):,} nodes")
        ladder.append({"id": spec, "label": label, "elo": round(elo), "kind": "fsf" if spec.startswith("fsf") else "other"})
    nets = []
    for prefix, label in RATED_NETS:
        estimates = [e for e in (match_rating(f, anchors) for f in sorted(RESULTS.glob(f"{prefix}-v800-fsf_*.json"))) if e]
        if estimates:
            nets.append({"id": prefix, "label": label, "visits": 800, "matches": estimates,
                         "elo": round(sum(e["elo"] for e in estimates) / len(estimates))})
    rl = []
    state_file = RL_RUN / "state.json"
    if state_file.exists():
        for h in json.loads(state_file.read_text())["history"]:
            gate = h.get("gate") or ""
            m = re.search(r"\(([\d.]+)%\), Elo ([+-]?\d+) \[([+-]?\d+), ([+-]?\d+)\]", gate)
            rl.append({"iteration": h["iteration"], "promoted": h["promoted"],
                       "score": float(m.group(1)) if m else None, "elo": int(m.group(2)) if m else None,
                       "low": int(m.group(3)) if m else None, "high": int(m.group(4)) if m else None})
    selfplay = []
    log = RL_RUN / "loop.log"
    if log.exists():
        lines = [l for l in log.read_text(errors="replace").splitlines()[-200:] if "positions from" in l]
        selfplay = lines[-2:]
    curves = []
    for run, label in [("abl-cnn256x15", "CNN 17M"), ("abl-tf384x10", "Transformer 35M"), ("p3-tf512x12", "Transformer 60M")]:
        f = TRAINING_DIR / "runs" / run / "log.jsonl"
        if f.exists():
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            curves.append({"label": label, "points": [{"samples": r["step"] * 2048, "top1": r["top1"], "value_acc": r["value_acc"]}
                                                      for r in rows if "top1" in r]})
    endings: dict[str, int] = {}
    if state_file.exists():
        for h in json.loads(state_file.read_text())["history"]:
            m = re.search(r"endings (\{.*\})", h.get("gate") or "")
            if m:
                for k, v in json.loads(m.group(1).replace("'", '"')).items():
                    endings[k] = endings.get(k, 0) + v
    # Web-app presets rated by eval/results/preset-<id>-<opponent>.json; "full" is the latest rated RL net at 800 visits.
    presets = {}
    for pid in ("casual", "club", "strong", "max"):
        est = [match_rating(f, anchors) for f in sorted(RESULTS.glob(f"preset-{pid}-*.json"))]
        est = [e for e in est if e]
        # A near-0% or near-100% match only bounds the rating; use real estimates when there are any.
        real = [e for e in est if 0.02 < e["score"] < 0.98]
        if real:
            presets[pid] = {"elo": round(sum(e["elo"] for e in real) / len(real)), "bound": ""}
        elif est:
            lows = [e for e in est if e["score"] <= 0.02]
            presets[pid] = ({"elo": min(e["elo"] for e in lows), "bound": "≤"} if lows
                            else {"elo": max(e["elo"] for e in est), "bound": "≥"})
    rl_nets = [n for n in nets if n["id"].startswith("rl1-")]
    if rl_nets:
        presets["full"] = {"elo": rl_nets[-1]["elo"], "bound": ""}
    return {"ladder": sorted(ladder, key=lambda x: x["elo"]), "nets": nets, "rl": rl, "selfplay": selfplay,
            "curves": curves, "endings": endings, "presets": presets}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--net", default=None)
    p.add_argument("--net-file", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    def nets() -> list[dict]:
        best = args.net_file.read_text().strip() if args.net_file else args.net
        out = [{"id": "best", "label": f"Latest RL ({Path(best).parent.name})", "path": best}]
        out += [{"id": i, "label": label, "path": path} for i, label, path in REFERENCE_NETS if Path(path).exists()]
        return out

    device = torch.device(args.device)
    evaluators: dict[str, tuple[str, NetEvaluator]] = {}  # net id -> (loaded path, evaluator)

    def evaluator(net_id: str, path: str) -> NetEvaluator:
        if net_id not in evaluators:
            evaluators[net_id] = (path, NetEvaluator(path, device))  # compiles once per id
        elif evaluators[net_id][0] != path:  # the RL net was promoted: same architecture, new weights
            evaluators[net_id][1].reload(path)
            evaluators[net_id] = (path, evaluators[net_id][1])
        return evaluators[net_id][1]

    engine = Engine(nets()[0]["path"], args.device, 0, 8)
    engine.evaluate = evaluator("best", engine.net_path)
    state = {"params": None}

    def search(req: dict) -> dict:
        visits = max(1, min(MAX_VISITS, int(req.get("visits", 800))))
        time_ms = req.get("time_ms")
        budget = None if not time_ms else max(0.05, min(MAX_TIME_MS, float(time_ms)) / 1000)
        if budget is not None and "visits" not in req:
            visits = MAX_VISITS
        params = tuple(float(req.get(k, v)) for k, v in DEFAULTS.items())
        choices = {n["id"]: n["path"] for n in nets()}
        net_id = req.get("net", "best") if req.get("net") in choices else "best"
        net = choices[net_id]
        if net != engine.net_path:
            engine.root = None
        engine.evaluate = evaluator(net_id, net)
        engine.net_path = net
        if params != state["params"]:
            cpuct, fpu, ptemp = params
            engine.batch = se.SearchBatch(cpuct=cpuct, fpu_reduction=fpu, fpu_reduction_root=fpu, policy_temp=ptemp)
            engine.root = None
            state["params"] = params
        util = max(0.0, min(1.0, float(req.get("util", 0.0))))
        if util != engine.evaluate.margin_utility:
            engine.evaluate.margin_utility = util
            engine.root = None  # cached values were computed with the old utility
        fen = se.Game(cho=req["cho"], han=req["han"]).root_fen()
        moves = [str(m) for m in req["moves"]]
        engine.set_position(fen, moves)
        start = time.time()
        best = engine.search(budget, visits)
        ms = (time.time() - start) * 1000
        b = engine.batch
        children = sorted(b.root_children(0), key=lambda c: (-c[1], -c[3]))
        if children and children[0][1] == 0:
            best = children[0][0]  # too few visits to search: play the network's own top choice
        move = best
        temperature = float(req.get("temperature", 0.0))
        if temperature > 0 and len(moves) < int(req.get("temp_plies", 1000)) and children:
            # Sample from visit counts, or from the network's priors when there was no real search.
            searched = children[0][1] > 0
            weights = [(n if searched else pr) ** (1.0 / temperature) for _, n, _, pr in children]
            move = random.choices([m for m, _, _, _ in children], weights=weights)[0]
        value = b.root_value(0)
        total = b.root_visits(0)
        depth_max, depth_mean = b.depth_stats(0)
        return {
            "move": move, "best": best, "cp": value_to_cp(value), "value": value,
            "net_value": b.root_net_value(0), "visits": total, "ms": ms,
            "depth": round(depth_mean, 1), "seldepth": depth_max,
            "pv": b.principal_variation(0, 16), "legal": len(children), "net": net,
            "candidates": [{"move": m, "visits": n, "q": None if math.isnan(q) else q, "prior": pr}
                           for m, n, q, pr in children[:12]],
            "tree": [[par, d, m, n, None if math.isnan(q) else q, pr, term]
                     for par, d, m, n, q, pr, term in b.dump_tree(0, 6, 4, max(2, total // 300))],
        }

    fsf_engines: dict[bool, object] = {}

    def fsf_move(req: dict) -> dict:
        nnue = bool(req.get("nnue", True))
        if nnue not in fsf_engines:
            fsf_engines[nnue] = fairy_stockfish(nnue=nnue)
        eng = fsf_engines[nnue]
        nodes = max(1, min(10_000_000, int(req.get("nodes", 64000))))
        fen = se.Game(cho=req["cho"], han=req["han"]).root_fen()
        moves = [str(m) for m in req["moves"]]
        start = time.time()
        info = eng.search(fen, moves, f"nodes {nodes}")
        game = se.Game(fen)
        for m in moves:
            game.push(m)
        legal = game.legal_moves()
        move = info.bestmove
        if move not in legal:  # FSF answers "(none)" when it thinks the game is over; any legal move will do
            passes = [m for m in legal if m[: len(m) // 2] == m[len(m) // 2:]]
            move = passes[0] if passes else legal[0]
        return {"move": move, "cp": info.score_cp, "mate": info.score_mate, "depth": info.depth,
                "nodes": info.nodes, "pv": info.pv[:16], "ms": (time.time() - start) * 1000}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/api/nets":
                self.reply(200, nets())
            elif self.path == "/api/progress":
                try:
                    self.reply(200, progress())
                except Exception as e:
                    self.reply(500, {"error": repr(e)})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            handlers = {"/api/move": search, "/api/fsf": fsf_move}
            if self.path not in handlers:
                self.reply(404, {"error": "not found"})
                return
            try:
                self.reply(200, handlers[self.path](json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            except Exception as e:  # report bad requests to the page instead of dropping the connection
                self.reply(400, {"error": repr(e)})

        def log_message(self, fmt: str, *a) -> None:
            pass

    print(f"serving {engine.net_path} on http://{args.host}:{args.port}", flush=True)
    # Single-threaded: the search engine is bound to the thread that created it.
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
