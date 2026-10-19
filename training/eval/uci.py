"""Minimal synchronous UCI driver for Fairy-Stockfish (and, later, Sinsan's own UCI engine)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
FSF_PATH = TRAINING_DIR / "teacher" / "engine" / "Fairy-Stockfish" / "src" / "stockfish"
NNUE_PATH = TRAINING_DIR / "teacher" / "nnue" / "janggi-9991472750de.nnue"


@dataclass
class SearchInfo:
    bestmove: str
    score_cp: int | None
    score_mate: int | None
    depth: int
    nodes: int
    multipv: list[tuple[str, int | None, int | None]]  # (move, cp, mate), best first
    pv: list[str] = field(default_factory=list)  # full principal variation of the best line


class UciEngine:
    def __init__(self, cmd: list[str], options: dict[str, str | int | bool] | None = None) -> None:
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self.send("uci")
        self.read_until("uciok")
        for name, value in (options or {}).items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            self.send(f"setoption name {name} value {value}")
        self.ready()

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        assert self.proc.stdout is not None
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited while waiting for {prefix!r}")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def ready(self) -> None:
        self.send("isready")
        self.read_until("readyok")

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.ready()

    def search(self, fen: str, moves: list[str], go: str) -> SearchInfo:
        self.send(f"position fen {fen}" + (f" moves {' '.join(moves)}" if moves else ""))
        self.send(f"go {go}")
        lines = self.read_until("bestmove")
        best = lines[-1].split()[1]
        pvs: dict[int, tuple[str, int | None, int | None]] = {}
        depth = nodes = 0
        cp = mate = None
        pv: list[str] = []
        for line in lines:
            if not line.startswith("info") or " pv " not in line:
                continue
            tok = line.split()
            get = lambda key: tok[tok.index(key) + 1] if key in tok else None  # noqa: E731
            rank = int(get("multipv") or 1)
            s_cp = int(tok[tok.index("cp") + 1]) if "cp" in tok else None
            s_mate = int(tok[tok.index("mate") + 1]) if "mate" in tok else None
            pvs[rank] = (tok[tok.index("pv") + 1], s_cp, s_mate)
            if rank == 1:
                depth, nodes, cp, mate = int(get("depth") or 0), int(get("nodes") or 0), s_cp, s_mate
                pv = tok[tok.index("pv") + 1:]
        return SearchInfo(best, cp, mate, depth, nodes, [pvs[k] for k in sorted(pvs)], pv)

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def fairy_stockfish(nnue: bool = True, multipv: int = 1, hash_mb: int = 16, threads: int = 1) -> UciEngine:
    options: dict[str, str | int | bool] = {
        "UCI_Variant": "janggi",
        "Threads": threads,
        "Hash": hash_mb,
        "MultiPV": multipv,
        "Use NNUE": nnue,
    }
    if nnue:
        options["EvalFile"] = str(NNUE_PATH)
    return UciEngine([str(FSF_PATH)], options)
