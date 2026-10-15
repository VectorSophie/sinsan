"""Players for the match runner, built from spec strings:

  fsf:nnue:<nodes>        Fairy-Stockfish with the Janggi NNUE, fixed node budget per move
  fsf:classical:<nodes>   Fairy-Stockfish classical evaluation, fixed node budget per move
  random                  uniform random legal move (never passes unless forced)
  oldnet:<ckpt>:<CxB>:<visits>
                          an original-project checkpoint (training/model/checkpoints/<ckpt>.pt, a
                          CxB SinsanTinyNet); visits=0 plays the policy argmax, otherwise PUCT
                          with that many visits. Kept only as a reference point on the ladder.

The referee is always sinsan_engine.Game (Fairy-Stockfish janggi rules); players only choose moves.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import sinsan_engine as se

from eval.uci import fairy_stockfish

TRAINING_DIR = Path(__file__).resolve().parents[1]


class Player:
    name: str

    def new_game(self) -> None:
        pass

    def choose(self, game: se.Game) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FsfPlayer(Player):
    def __init__(self, nnue: bool, nodes: int) -> None:
        self.name = f"fsf:{'nnue' if nnue else 'classical'}:{nodes}"
        self.nodes = nodes
        self.engine = fairy_stockfish(nnue=nnue)

    def new_game(self) -> None:
        self.engine.new_game()

    def choose(self, game: se.Game) -> str:
        info = self.engine.search(game.root_fen(), game.moves(), f"nodes {self.nodes}")
        if info.bestmove not in game.legal_moves():
            # FSF reports "(none)" when it considers the game over; the referee decides, so fall
            # back to a pass (always legal unless in check) or any legal move.
            legal = game.legal_moves()
            passes = [m for m in legal if m[: len(m) // 2] == m[len(m) // 2 :]]
            return passes[0] if passes else legal[0]
        return info.bestmove

    def close(self) -> None:
        self.engine.close()


class RandomPlayer(Player):
    def __init__(self, seed: int = 0) -> None:
        self.name = "random"
        self.rng = random.Random(seed)

    def choose(self, game: se.Game) -> str:
        legal = game.legal_moves()
        moves = [m for m in legal if m[: len(m) // 2] != m[len(m) // 2 :]] or legal
        return self.rng.choice(moves)


class OldNetPlayer(Player):
    """Original-project network (absolute 15-plane input) with optional plain PUCT."""

    def __init__(self, checkpoint: str, shape: str, visits: int) -> None:
        import torch

        sys.path.insert(0, str(TRAINING_DIR))
        from model.network import SinsanTinyNet, fen_to_planes

        torch.set_num_threads(1)
        channels, blocks = (int(x) for x in shape.split("x"))
        self.torch = torch
        self.fen_to_planes = fen_to_planes
        self.net = SinsanTinyNet(channels, blocks)
        self.net.load_state_dict(torch.load(TRAINING_DIR / "model" / "checkpoints" / f"{checkpoint}.pt", map_location="cpu"))
        self.net.eval()
        self.visits = visits
        self.name = f"oldnet:{checkpoint}:{shape}:{visits}"

    def evaluate(self, game: se.Game) -> tuple[dict[str, float], float]:
        fen = game.fen()
        x = self.fen_to_planes(fen, fen.split()[1] == "w").unsqueeze(0)
        with self.torch.no_grad():
            logits, value = self.net(x)
        legal = game.legal_moves()
        actions = game.legal_actions()
        lg = [float(logits[0, a]) for a in actions]
        m = max(lg)
        exps = [math.exp(v - m) for v in lg]
        total = sum(exps)
        return {mv: e / total for mv, e in zip(legal, exps)}, float(value[0])

    def choose(self, game: se.Game) -> str:
        priors, _ = self.evaluate(game)
        if self.visits <= 1:
            return max(priors, key=priors.get)
        return self.puct(game, priors)

    def puct(self, root_game: se.Game, root_priors: dict[str, float], c_puct: float = 1.5) -> str:
        # node: [priors, children{move: node}, visits, value_sum, terminal_value]
        def new_node(priors):
            return {"priors": priors, "children": {}, "n": 0, "w": 0.0}

        root = new_node(root_priors)
        for _ in range(self.visits):
            game = root_game.copy()
            node, path = root, [root]
            while True:
                if node["priors"] is None:
                    break
                sqrt_n = math.sqrt(max(1, node["n"]))
                best, best_score = None, -1e9
                for mv, p in node["priors"].items():
                    child = node["children"].get(mv)
                    q = -child["w"] / child["n"] if child and child["n"] else 0.0
                    score = q + c_puct * p * sqrt_n / (1 + (child["n"] if child else 0))
                    if score > best_score:
                        best, best_score = mv, score
                game.push(best)
                child = node["children"].get(best)
                if child is None:
                    child = new_node(None)
                    node["children"][best] = child
                    path.append(child)
                    break
                node = child
                path.append(node)
            leaf = path[-1]
            result = game.result()
            if result is not None:
                value = 1.0 if result[0] == game.side_to_move else -1.0
            else:
                priors, value = self.evaluate(game)
                leaf["priors"] = priors
            for n in reversed(path):
                n["n"] += 1
                n["w"] += value
                value = -value
        return max(root["children"].items(), key=lambda kv: kv[1]["n"])[0]


class NnuePlayer(Player):
    """Alpha-beta + small NNUE (the browser-student candidate), fixed node budget per move."""

    def __init__(self, path: str, nodes: int) -> None:
        self.name = f"nnue:{path}:{nodes}"
        self.weights = Path(path).read_bytes()
        self.nodes = nodes

    def choose(self, game: se.Game) -> str:
        best, *_ = se.ab_search(self.weights, game.root_fen(), game.moves(), 64, self.nodes, 18)
        return best


def make_player(spec: str, seed: int = 0) -> Player:
    parts = spec.split(":")
    if parts[0] == "fsf":
        return FsfPlayer(nnue=parts[1] == "nnue", nodes=int(parts[2]))
    if parts[0] == "nnue":
        return NnuePlayer(":".join(parts[1:-1]), int(parts[-1]))
    if parts[0] == "random":
        return RandomPlayer(seed)
    if parts[0] == "oldnet":
        return OldNetPlayer(parts[1], parts[2], int(parts[3]))
    raise ValueError(f"unknown player spec {spec!r}")
