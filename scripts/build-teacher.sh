#!/usr/bin/env bash
# Clones and builds Fairy-Stockfish as Sinsan's offline teacher engine, pinned to a specific
# commit for reproducibility (see docs/RESEARCH.md / THIRD_PARTY_NOTICES.md for why this commit
# and not a tagged release - the project has had no fresh tag since 2021 and develops on an
# untagged master). Never vendors the engine into the repo: source and binary live in
# training/teacher/engine/, which is gitignored, since Fairy-Stockfish (GPL-3.0) is used only as
# a separate offline process, never linked or distributed with Sinsan's own code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$REPO_ROOT/training/teacher/engine/Fairy-Stockfish"
PINNED_COMMIT="c19b5f6c66894fdb0e88d0dd100e3885f744760a"
ARCH="${SINSAN_TEACHER_ARCH:-x86-64-bmi2}"

if [ ! -d "$ENGINE_DIR" ]; then
  echo "Cloning fairy-stockfish/Fairy-Stockfish..."
  git clone https://github.com/fairy-stockfish/Fairy-Stockfish.git "$ENGINE_DIR"
fi

cd "$ENGINE_DIR"
git fetch origin "$PINNED_COMMIT" 2>/dev/null || true
git checkout "$PINNED_COMMIT"

echo "Building with ARCH=$ARCH largeboards=yes (override via SINSAN_TEACHER_ARCH)..."
cd src
make -j"$(nproc)" build ARCH="$ARCH" largeboards=yes

echo
echo "Built: $ENGINE_DIR/src/stockfish"
sha256sum stockfish
echo
echo "Verifying the janggi variant responds..."
printf 'uci\nsetoption name UCI_Variant value janggi\nposition startpos\nd\nquit\n' | ./stockfish | grep -A2 "info string variant janggi"
