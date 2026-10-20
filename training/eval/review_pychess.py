"""Reviews the pychess bot's games (runs/pychess/games.jsonl) with the full GPU network.

For every game not reviewed yet, each position is searched with the current best RL net, giving
Cho's win chance before and after every move. The report lists both sides' costly moves; the bot's
mistakes (moves that lost at least --threshold win chance against the reviewer's best move) are
appended to runs/pychess/bot_mistakes.jsonl as positions to test, or train, the student on.

  CUDA_VISIBLE_DEVICES=1 uv run python -m eval.review_pychess [--visits 800] [--all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uci_engine import Engine

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "runs" / "pychess" / "games.jsonl"
REVIEWED = ROOT / "runs" / "pychess" / "reviewed.jsonl"
MISTAKES = ROOT / "runs" / "pychess" / "bot_mistakes.jsonl"
BOT = "sin-san"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--visits", type=int, default=800)
    p.add_argument("--threshold", type=float, default=0.08, help="win-chance drop that counts as a mistake")
    p.add_argument("--all", action="store_true", help="review every game again")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    done = set() if args.all or not REVIEWED.exists() else {json.loads(l)["id"] for l in REVIEWED.read_text().splitlines() if l}
    games = [json.loads(l) for l in GAMES.read_text().splitlines() if l] if GAMES.exists() else []
    todo = [g for g in games if g["id"] not in done and g.get("initialFen") and g.get("moves")]
    if not todo:
        print("no new games")
        return
    net = (ROOT / "runs" / "rl1" / "best.txt").read_text().strip()
    engine = Engine(net, args.device, args.visits, 16)
    engine.ensure_loaded()
    engine.info = lambda *a: None

    for g in todo:
        fen, moves = g["initialFen"], g["moves"]
        bot = "w" if g.get("white", "").lower() == BOT else "b"
        cho, best = [], []
        for ply in range(len(moves) + 1):
            engine.set_position(fen, moves[:ply])
            if ply < len(moves):
                best.append(engine.search(None, args.visits))
            else:
                engine.search(None, args.visits)
            v = engine.batch.root_value(0)
            cho.append(v if ply % 2 == 0 else -v)  # root value is for the side to move
        # Cho's win chance (0..1) before each ply, and after the last one.
        win = [(c + 1) / 2 for c in cho]
        rows, bot_moves, bot_agree = [], 0, 0
        for ply, mv in enumerate(moves):
            mover = "w" if ply % 2 == 0 else "b"
            drop = (win[ply] - win[ply + 1]) if mover == "w" else (win[ply + 1] - win[ply])
            if mover == bot:
                bot_moves += 1
                bot_agree += mv == best[ply]
            if drop >= args.threshold and mv != best[ply]:
                rows.append((ply, mover, mv, best[ply], drop))
                if mover == bot:
                    with open(MISTAKES, "a") as f:
                        f.write(json.dumps({"game": g["id"], "fen": fen, "moves": moves[:ply], "played": mv,
                                            "best": best[ply], "drop": round(drop, 3), "level": g.get("level")}) + "\n")
        opp = g.get("black") if bot == "w" else g.get("white")
        res = g.get("result", {})
        print(f"\n== {g['id']} vs {opp} ({(g.get('challenger') or {}).get('rating', '?')}), bot {'Cho' if bot == 'w' else 'Han'}, "
              f"{len(moves)} plies, result {res.get('reason')} {'(bot won)' if res.get('bot_won') else ''}")
        print(f"bot agreed with the reviewer on {bot_agree}/{bot_moves} moves")
        for ply, mover, mv, b, drop in rows:
            who = "BOT" if mover == bot else "opp"
            print(f"  ply {ply + 1:>3} {who}: {mv:7} (best {b:7}) -{100 * drop:.0f}% win chance")
        with open(REVIEWED, "a") as f:
            f.write(json.dumps({"id": g["id"], "bot_agree": bot_agree, "bot_moves": bot_moves,
                                "bot_mistakes": sum(1 for r in rows if r[1] == bot),
                                "win_cho": [round(w, 3) for w in win]}) + "\n")


if __name__ == "__main__":
    main()
