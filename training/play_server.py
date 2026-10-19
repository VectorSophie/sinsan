"""Local HTTP endpoint so the web app (apps/web) can play against a full-size checkpoint.

POST /api/move  {"cho": formation, "han": formation, "moves": [uci...], "visits": N}
             -> {"move": uci, "cp": centipawns for the side to move, "visits": N, "pv": [uci...]}
GET  /api/info  -> {"net": checkpoint path}

The net is read from --net, or from --net-file (e.g. runs/rl1/best.txt) re-read on every request so
the server always plays the current best RL net. Requests are handled one at a time.

Run: cd training && CUDA_VISIBLE_DEVICES=1 uv run python play_server.py --net-file runs/rl1/best.txt
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import sinsan_engine as se

from uci_engine import Engine, value_to_cp

MAX_VISITS = 20_000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--net", default=None)
    p.add_argument("--net-file", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    def current_net() -> str:
        return args.net_file.read_text().strip() if args.net_file else args.net

    engine = Engine(current_net(), args.device, 0, 8)
    engine.ensure_loaded()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/api/info":
                self.reply(200, {"net": engine.net_path})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/move":
                self.reply(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                visits = max(1, min(MAX_VISITS, int(req.get("visits", 800))))
                fen = se.Game(cho=req["cho"], han=req["han"]).root_fen()
                moves = [str(m) for m in req["moves"]]
                net = current_net()
                if net != engine.net_path:
                    engine.evaluate.reload(net)
                    engine.net_path = net
                    engine.root = None
                engine.set_position(fen, moves)
                move = engine.search(None, visits)
                cp = value_to_cp(engine.batch.root_value(0))
                pv = engine.batch.principal_variation(0, 12)
                self.reply(200, {"move": move, "cp": cp, "visits": visits, "pv": pv, "net": net})
            except Exception as e:  # report bad requests to the page instead of dropping the connection
                self.reply(400, {"error": repr(e)})

        def log_message(self, fmt: str, *a) -> None:
            pass

    print(f"serving {engine.net_path} on http://{args.host}:{args.port}", flush=True)
    # Single-threaded: the search engine is bound to the thread that created it.
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
