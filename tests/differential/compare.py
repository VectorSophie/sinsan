"""Differential test (Section 8.4): compares packages/rules' legal-move sets (via
tests/differential/generate-cases.ts's cases.jsonl) against pyffish's for the same
position+variant. Mismatches are saved as minimal repros under artifacts/rule-mismatches/,
per the spec's requirement to document rule discrepancies rather than silently ignore them.

Run with: cd training && uv run python ../tests/differential/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pyffish

REPO_ROOT = Path(__file__).parent.parent.parent
CASES_PATH = REPO_ROOT / "tests" / "differential" / "cases.jsonl"
MISMATCH_DIR = REPO_ROOT / "artifacts" / "rule-mismatches"


def main() -> None:
    cases = [json.loads(line) for line in CASES_PATH.read_text().splitlines() if line.strip()]
    MISMATCH_DIR.mkdir(parents=True, exist_ok=True)

    mismatches = 0
    for case in cases:
        py_moves = set(pyffish.legal_moves(case["variant"], case["fen"], []))
        ts_moves = set(case["tsLegalMovesUci"])

        if py_moves == ts_moves:
            continue

        mismatches += 1
        ts_only = sorted(ts_moves - py_moves)
        py_only = sorted(py_moves - ts_moves)
        repro = {
            "fen": case["fen"],
            "variant": case["variant"],
            "rule_profile": case["ruleProfile"],
            "game_id": case["gameId"],
            "ply": case["ply"],
            "ts_only": ts_only,
            "pyffish_only": py_only,
            "ts_count": len(ts_moves),
            "pyffish_count": len(py_moves),
        }
        out_path = MISMATCH_DIR / f"mismatch-{case['gameId']}-{case['ply']}.json"
        out_path.write_text(json.dumps(repro, indent=2) + "\n")
        print(f"MISMATCH game={case['gameId']} ply={case['ply']} variant={case['variant']}")
        print(f"  ts_only ({len(ts_only)}): {ts_only}")
        print(f"  pyffish_only ({len(py_only)}): {py_only}")
        print(f"  fen: {case['fen']}")

    print(f"\n{len(cases)} positions compared, {mismatches} mismatches ({len(cases) - mismatches} matched)")


if __name__ == "__main__":
    main()
