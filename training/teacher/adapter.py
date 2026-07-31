"""Fairy-Stockfish teacher adapter (Phase 3 prototype).

Drives a pinned Fairy-Stockfish build over UCI to produce policy/value labels for
Janggi positions, per the Stage A labeling strategy in docs/DATASET_DESIGN.md:
MultiPV top-K candidate moves with scores, from a fixed node budget, rather than
a single best-move label.

Stdlib only, deliberately - this is a subprocess/text-protocol wrapper, not
something that needs a UCI library dependency.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENGINE_PATH = Path(__file__).parent / "engine" / "Fairy-Stockfish" / "src" / "stockfish"

# Sinsan's own default starting position (masang-sangma both sides, kja profile) - confirmed
# during Phase 3 verification to be byte-identical to Fairy-Stockfish's own janggi startpos.
STARTPOS_FEN = "rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1"

_MULTIPV_LINE = re.compile(
    r"info depth (?P<depth>\d+) .*?"
    r"multipv (?P<multipv>\d+) "
    r"score (?P<scoretype>cp|mate) (?P<scoreval>-?\d+) .*?"
    r"nodes (?P<nodes>\d+) .*?"
    r" pv (?P<pv>.+)"
)


@dataclass
class CandidateMove:
    move: str
    score_cp: int | None
    score_mate: int | None
    depth: int
    multipv_rank: int


@dataclass
class TeacherLabel:
    fen: str
    best_move: str
    candidates: list[CandidateMove]
    nodes: int
    depth: int


class TeacherEngine:
    """One persistent Fairy-Stockfish subprocess, driven over UCI.

    Deterministic config: fixed Threads=1 and a fixed node budget per `label()` call
    (not `movetime`), per docs/DATASET_DESIGN.md's requirement that labeling be
    reproducible rather than wall-clock-dependent.
    """

    def __init__(
        self,
        engine_path: Path = DEFAULT_ENGINE_PATH,
        variant: str = "janggi",
        multipv: int = 8,
        threads: int = 1,
    ) -> None:
        if not Path(engine_path).exists():
            raise FileNotFoundError(
                f"Fairy-Stockfish binary not found at {engine_path}. Run scripts/build-teacher.sh first."
            )
        self._proc = subprocess.Popen(
            [str(engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name UCI_Variant value {variant}")
        self._send(f"setoption name MultiPV value {multipv}")
        self._send(f"setoption name Threads value {threads}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, command: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def _read_until(self, marker: str) -> list[str]:
        assert self._proc.stdout is not None
        lines = []
        for line in self._proc.stdout:
            lines.append(line.rstrip("\n"))
            if marker in line:
                break
        return lines

    def label(self, fen: str, nodes: int = 20_000) -> TeacherLabel:
        """Fixed node-budget MultiPV search - Stage A of docs/DATASET_DESIGN.md."""
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        lines = self._read_until("bestmove")

        candidates: dict[int, CandidateMove] = {}
        max_depth = 0
        max_nodes = 0
        for line in lines:
            m = _MULTIPV_LINE.search(line)
            if not m:
                continue
            depth = int(m.group("depth"))
            rank = int(m.group("multipv"))
            score_cp = int(m.group("scoreval")) if m.group("scoretype") == "cp" else None
            score_mate = int(m.group("scoreval")) if m.group("scoretype") == "mate" else None
            move = m.group("pv").split(" ")[0]
            nodes_seen = int(m.group("nodes"))
            max_depth = max(max_depth, depth)
            max_nodes = max(max_nodes, nodes_seen)
            # Later (deeper) reports for the same rank supersede earlier ones.
            candidates[rank] = CandidateMove(
                move=move, score_cp=score_cp, score_mate=score_mate, depth=depth, multipv_rank=rank
            )

        best_move_line = next((l for l in lines if l.startswith("bestmove")), "bestmove 0000")
        best_move = best_move_line.split()[1]

        return TeacherLabel(
            fen=fen,
            best_move=best_move,
            candidates=[candidates[r] for r in sorted(candidates)],
            nodes=max_nodes,
            depth=max_depth,
        )

    def fen_after(self, fen: str, moves: list[str]) -> str:
        """Applies UCI moves to `fen` via the engine's own move application and returns the
        resulting FEN (parsed from the `d` command's `Fen:` line) - avoids hand-constructing
        successor positions, which is easy to get subtly wrong (e.g. an illegal cannon move)."""
        self._send(f"position fen {fen} moves {' '.join(moves)}")
        self._send("d")
        lines = self._read_until("Fen:")
        fen_line = next(l for l in lines if l.startswith("Fen:"))
        return fen_line.removeprefix("Fen:").strip()

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self) -> "TeacherEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def demo() -> None:
    """Self-check: label the startpos and a one-move-in position, print + assert sane output.

    ponytail: this doubles as the runnable check for the subprocess/regex parsing logic above -
    no test framework, just assertions against a real engine run.
    """
    with TeacherEngine(multipv=4) as engine:
        label = engine.label(STARTPOS_FEN, nodes=20_000)
        print(f"startpos: best={label.best_move} depth={label.depth} nodes={label.nodes}")
        for c in label.candidates:
            score = f"cp{c.score_cp}" if c.score_cp is not None else f"mate{c.score_mate}"
            print(f"  #{c.multipv_rank} {c.move} {score} (depth {c.depth})")

        assert label.best_move != "0000", "expected a real move, not a null move"
        assert len(label.candidates) > 1, "expected MultiPV>1 to produce multiple candidates"
        assert label.nodes > 0
        assert label.depth > 0
        assert all(c.move for c in label.candidates)

        # A position one ply after the opening move should also label cleanly. The successor FEN
        # comes from the engine's own move application (fen_after), not hand-constructed, so it's
        # guaranteed to be a legally reachable position rather than a possibly-illegal guess.
        follow_up_fen = engine.fen_after(STARTPOS_FEN, [label.best_move])
        label2 = engine.label(follow_up_fen, nodes=20_000)
        print(f"follow-up: best={label2.best_move} depth={label2.depth} nodes={label2.nodes}")
        assert label2.best_move != "0000"

    print("OK: teacher adapter self-check passed")


if __name__ == "__main__":
    demo()
