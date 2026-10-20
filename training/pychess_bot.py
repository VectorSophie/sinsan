"""Plays Sinsan on pychess.org through its BOT API (the `sin-san` BOT account).

Protocol (pychess-variants server/bot_api.py, Lichess-style ndjson streams):
  GET  /api/stream/event                     challenge / gameStart / ping events
       (header X-PyChess-Bot-Capabilities: {"version": 1, "variants": ["janggi"]})
  POST /api/challenge/{id}/accept|decline
  GET  /api/bot/game/stream/{id}             gameFull, then gameState / gameEnd
  POST /api/bot/game/{id}/move/{uci}
BOT accounts cannot create challenges on pychess, so the bot waits to be challenged. For a bot side
the server picks the Janggi formation itself and sends the resulting start position as initialFen.

Moves are FSF UCI (passes are the general's square twice), the same notation as sinsan_engine.
The engine's search objects must stay on one thread, so all searches run on a single worker thread
fed by the per-game stream threads.

  CUDA_VISIBLE_DEVICES=1 uv run python pychess_bot.py --level browser
Reads PYCHESS_API_KEY from ../.env (or the environment). Games are logged to runs/pychess/.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

from uci_engine import Engine, value_to_cp

ROOT = Path(__file__).resolve().parent
SERVER = "https://www.pychess.org"
CAPABILITIES = json.dumps({"version": 1, "variants": ["janggi"]})
LOG_DIR = ROOT / "runs" / "pychess"

# Strength levels, as the web app's presets (ladder ratings in docs/REBUILD_LOG.md).
LEVELS = {
    "browser": ("runs/student-tf192x6/ema-0030000.pt", 400),
    "strong": (None, 64),  # None: the current best RL net (runs/rl1/best.txt)
    "full": (None, 800),
}


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Api:
    def __init__(self, token: str) -> None:
        self.headers = {"Authorization": f"Bearer {token}", "X-PyChess-Bot-Capabilities": CAPABILITIES,
                        "User-Agent": "sinsan-bot"}

    def post(self, path: str, data: dict | None = None) -> dict | None:
        body = json.dumps(data).encode() if data is not None else b""
        req = urllib.request.Request(SERVER + path, data=body, method="POST",
                                     headers={**self.headers, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            log(f"POST {path} -> {e.code} {e.read()[:200]!r}")
            return None

    def get_json(self, path: str) -> dict:
        req = urllib.request.Request(SERVER + path, headers=self.headers)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def stream(self, path: str):
        """Yields parsed ndjson lines until the server closes the stream."""
        req = urllib.request.Request(SERVER + path, headers=self.headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                line = line.strip()
                if line:
                    yield json.loads(line)


class Searcher(threading.Thread):
    """Owns the engine; answers (fen, moves, budget_s, visits) requests one at a time."""

    def __init__(self, net: str, device: str) -> None:
        super().__init__(daemon=True)
        self.net, self.device = net, device
        self.requests: queue.Queue = queue.Queue()

    def run(self) -> None:
        engine = Engine(self.net, self.device, visits=0, leaves=16)
        engine.ensure_loaded()
        engine.info = lambda nodes, elapsed: None  # no UCI info lines
        while True:
            fen, moves, budget, visits, reply = self.requests.get()
            try:
                engine.set_position(fen, moves)
                start = time.time()
                move = engine.search(budget, visits)
                reply.put((move, engine.batch.root_visits(0), value_to_cp(engine.batch.root_value(0)), time.time() - start))
            except Exception as e:  # noqa: BLE001 - report to the game thread
                reply.put(e)

    def think(self, fen: str, moves: list[str], budget: float, visits: int) -> tuple[str, int, int, float]:
        reply: queue.Queue = queue.Queue()
        self.requests.put((fen, moves, budget, visits, reply))
        out = reply.get()
        if isinstance(out, Exception):
            raise out
        return out


def move_budget(remaining_ms: float, inc_ms: float, plies: int) -> float:
    """Seconds for this move: a share of the clock plus most of the increment, minus lag."""
    remaining, inc = remaining_ms / 1000, inc_ms / 1000
    moves_to_go = max(20, 60 - plies // 2)
    budget = min(remaining * 0.4, remaining / moves_to_go + 0.8 * inc)
    return max(0.1, budget - 0.3)


class Bot:
    def __init__(self, api: Api, searcher: Searcher, username: str, visits: int, level: str, max_games: int,
                 min_base_s: int) -> None:
        self.api, self.searcher, self.username = api, searcher, username
        self.visits, self.level, self.max_games, self.min_base_s = visits, level, max_games, min_base_s
        self.games: set[str] = set()
        self.lock = threading.Lock()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def on_challenge(self, c: dict) -> None:
        cid, who = c["id"], c.get("challenger", {}).get("name")
        variant = c.get("variant", {}).get("key")
        tc = c.get("timeControl", {})
        reason = None
        if variant != "janggi":
            reason = "variant"
        elif tc.get("type") != "clock" or int(tc.get("limit", 0)) + 40 * int(tc.get("increment", 0)) < self.min_base_s:
            reason = "timeControl"
        else:
            with self.lock:
                if len(self.games) >= self.max_games:
                    reason = "later"
        log(f"challenge {cid} from {who}: {variant} {tc} rated={c.get('rated')} -> {'decline ' + reason if reason else 'accept'}")
        if reason:
            self.api.post(f"/api/challenge/{cid}/decline", {"reason": reason})
        else:
            self.api.post(f"/api/challenge/{cid}/accept")

    def on_game_start(self, game_id: str) -> None:
        with self.lock:
            if game_id in self.games:
                return
            self.games.add(game_id)
        threading.Thread(target=self.play, args=(game_id,), daemon=True, name=f"game-{game_id}").start()

    def play(self, game_id: str) -> None:
        record: dict = {"id": game_id, "level": self.level, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "moves": []}
        try:
            fen, color, last_ply = None, None, -1
            for ev in self.api.stream(f"/api/bot/game/stream/{game_id}"):
                kind = ev.get("type")
                if kind == "gameFull":
                    fen = ev["initialFen"] or None
                    color = "w" if ev["white"]["name"].lower() == self.username.lower() else "b"
                    record.update(white=ev["white"]["name"], black=ev["black"]["name"], initialFen=fen)
                    log(f"game {game_id}: {ev['white']['name']} (Cho) vs {ev['black']['name']} (Han), we play {color}")
                    state = ev["state"]
                elif kind == "gameState":
                    state = ev
                elif kind == "gameEnd":
                    record.update(end=ev)
                    break
                else:
                    continue
                if fen is None or state.get("status", "started") != "started":
                    continue
                moves = state["moves"].split() if state["moves"] else []
                record["moves"] = moves
                ours = (len(moves) % 2 == 0) == (color == "w")
                if not ours or len(moves) == last_ply:
                    continue
                last_ply = len(moves)
                remaining = state["wtime"] if color == "w" else state["btime"]
                inc = state["winc"] if color == "w" else state["binc"]
                budget = move_budget(float(remaining), float(inc) * (1000 if float(inc) < 100 else 1), len(moves))
                move, n, cp, secs = self.searcher.think(fen, moves, budget, self.visits)
                log(f"game {game_id} ply {len(moves)}: {move} ({n} visits, cp {cp}, {secs:.2f}s of {budget:.1f}s)")
                if self.api.post(f"/api/bot/game/{game_id}/move/{move}") is None:
                    last_ply = -1  # rejected: try again on the next state
        except Exception as e:  # noqa: BLE001 - one broken game must not stop the bot
            log(f"game {game_id} stream ended with {e!r}")
            record["error"] = repr(e)
        finally:
            with self.lock:
                self.games.discard(game_id)
            record["finish"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_DIR / "games.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            log(f"game {game_id} finished: {record.get('end', {}).get('result', '?')}")

    def run(self) -> None:
        while True:
            try:
                log("connecting to the event stream")
                for ev in self.api.stream("/api/stream/event"):
                    kind = ev.get("type")
                    if kind == "challenge":
                        self.on_challenge(ev["challenge"])
                    elif kind == "gameStart":
                        self.on_game_start(ev["game"]["id"])
                    elif kind != "ping":
                        log(f"event {kind}: {json.dumps(ev)[:200]}")
            except Exception as e:  # noqa: BLE001 - reconnect after network errors
                log(f"event stream dropped: {e!r}")
            time.sleep(5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=sorted(LEVELS), default="browser")
    p.add_argument("--max-games", type=int, default=1, help="games played at once")
    p.add_argument("--min-base", type=int, default=60, help="decline clocks shorter than this (base + 40 x increment, seconds)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--check", action="store_true", help="only verify the token and the engine, then exit")
    args = p.parse_args()

    load_env(ROOT.parent / ".env")
    token = os.environ.get("PYCHESS_API_KEY")
    if not token:
        raise SystemExit("PYCHESS_API_KEY is not set (../.env)")
    api = Api(token)
    account = api.get_json("/api/account")
    if account.get("title") != "BOT":
        raise SystemExit(f"{account.get('username')} is not a BOT account on pychess")
    net, visits = LEVELS[args.level]
    net = net or (ROOT / "runs" / "rl1" / "best.txt").read_text().strip()
    searcher = Searcher(net, args.device if torch.cuda.is_available() else "cpu")
    searcher.start()
    start_fen = "rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1"
    move, n, cp, secs = searcher.think(start_fen, [], 5.0, visits)
    log(f"account {account['username']} (BOT), level {args.level}: {Path(net).parent.name} at {visits} visits; "
        f"test search {move} ({n} visits, {secs:.2f}s)")
    if args.check:
        return
    Bot(api, searcher, account["username"], visits, args.level, args.max_games, args.min_base).run()


if __name__ == "__main__":
    main()
