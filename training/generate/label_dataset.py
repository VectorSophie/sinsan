"""Teacher labeling pipeline (Phase 4 smoke tier). Reads positions produced by
training/generate/self-play.ts and labels each with the pinned Fairy-Stockfish teacher
(training/teacher/adapter.py), per the Stage A strategy in docs/DATASET_DESIGN.md.

Honesty notes carried over from docs/DATASET_DESIGN.md's own cautions:
  - teacher_value_raw is a simple bounded tanh(cp/SCORE_SCALE) transform, not a claimed-calibrated
    win probability - Section 12.2 explicitly warns against presenting an arbitrary cp->probability
    formula as calibrated truth. teacher_value_calibrated is left null; computing it requires
    held-out self-play outcomes, which this smoke run does not produce (Phase 6 concern).
  - teacher_wdl is left null - UCI_ShowWDL was not enabled, since its calibration for Janggi is an
    open question (docs/REFERENCES.md), not something to enable silently and then trust.

Run with: python3 training/generate/label_dataset.py [max_positions]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from action_space import load as load_action_space  # noqa: E402
from teacher.adapter import TeacherEngine  # noqa: E402

TEACHER_VERSION = "fairy-stockfish@c19b5f6c66894fdb0e88d0dd100e3885f744760a"
SCORE_SCALE = 800  # cp scale for the tanh bounding transform - arbitrary, not fit/calibrated
MATE_SCORE_CP_EQUIVALENT = 100_000  # sentinel magnitude so mate scores sort/bound sensibly

IN_PATH = Path(__file__).parent.parent / "datasets" / "smoke-positions.jsonl"
OUT_PATH = Path(__file__).parent.parent / "datasets" / "smoke-labeled.jsonl"


def parse_uci_square(square_text: str, board_cols: int, board_rows: int) -> int:
    """'b10' -> our square index. Rank 1 is Cho's back rank (our row board_rows-1); rank
    board_rows is Han's back rank (our row 0) - confirmed against the engine's own `d` board
    display during Phase 3 verification, not assumed."""
    file_char = square_text[0]
    rank_number = int(square_text[1:])
    col = ord(file_char) - ord("a")
    row = board_rows - rank_number
    return row * board_cols + col


def parse_uci_move(move_text: str, board_cols: int, board_rows: int) -> tuple[int, int]:
    half = len(move_text) // 2
    # ranks can be 1-2 chars ("9" vs "10"); files are always 1 char, so split by scanning for the
    # second file letter rather than assuming a fixed midpoint.
    for split in range(2, len(move_text) - 1):
        if move_text[split].isalpha():
            from_sq = parse_uci_square(move_text[:split], board_cols, board_rows)
            to_sq = parse_uci_square(move_text[split:], board_cols, board_rows)
            return from_sq, to_sq
    raise ValueError(f"could not parse UCI move: {move_text!r}")


def score_to_cp_equivalent(score_cp: int | None, score_mate: int | None) -> int:
    if score_cp is not None:
        return score_cp
    assert score_mate is not None
    sign = 1 if score_mate > 0 else -1
    return sign * (MATE_SCORE_CP_EQUIVALENT - abs(score_mate))


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def main() -> None:
    max_positions = int(sys.argv[1]) if len(sys.argv) > 1 else None
    action_space = load_action_space()

    positions = [json.loads(line) for line in IN_PATH.read_text().splitlines() if line.strip()]
    if max_positions is not None:
        positions = positions[:max_positions]

    labeled = []
    with TeacherEngine(multipv=8) as engine:
        for i, pos in enumerate(positions):
            label = engine.label(pos["fen"], nodes=8_000)

            candidate_cp_equivs = [score_to_cp_equivalent(c.score_cp, c.score_mate) for c in label.candidates]
            probs = softmax([v / SCORE_SCALE for v in candidate_cp_equivs])

            teacher_actions = []
            teacher_policy = []
            for candidate, prob in zip(label.candidates, probs):
                from_sq, to_sq = parse_uci_move(candidate.move, action_space.board_cols, action_space.board_rows)
                # The engine encodes pass as the same square twice (confirmed in docs/RULES.md);
                # a from==to move is never a real piece move, so must be routed to the dedicated
                # pass action rather than left for encode() to reject (it has no dr=0,dc=0
                # template - dropping it silently here left teacher_policy summing to <1).
                is_pass_candidate = from_sq == to_sq
                action_id = action_space.encode(from_sq, to_sq, is_pass=is_pass_candidate)
                if action_id is None:
                    continue  # shouldn't happen for a real engine move; skip defensively rather than crash a long run
                teacher_actions.append(action_id)
                teacher_policy.append(prob)

            top_cp_equiv = candidate_cp_equivs[0] if candidate_cp_equivs else 0

            labeled.append(
                {
                    "position": pos["fen"],
                    "legal_actions": pos["legal_actions"],
                    "teacher_actions": teacher_actions,
                    "teacher_policy": teacher_policy,
                    "teacher_value_raw": math.tanh(top_cp_equiv / SCORE_SCALE),
                    "teacher_value_calibrated": None,
                    "teacher_score": top_cp_equiv,
                    "teacher_wdl": None,
                    "search_nodes": label.nodes,
                    "search_depth": label.depth,
                    "multipv": len(label.candidates),
                    "source": pos["source"],
                    "game_id": pos["game_id"],
                    "ply": pos["ply"],
                    "setup_cho": pos["setup_cho"],
                    "setup_han": pos["setup_han"],
                    "result": None,
                    "teacher_version": TEACHER_VERSION,
                    "rule_profile": pos["rule_profile"],
                }
            )

            if (i + 1) % 50 == 0 or (i + 1) == len(positions):
                print(f"labeled {i + 1}/{len(positions)}")

    OUT_PATH.write_text("\n".join(json.dumps(r) for r in labeled) + "\n")
    print(f"Wrote {len(labeled)} labeled positions to {OUT_PATH}")


if __name__ == "__main__":
    main()
