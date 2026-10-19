#!/usr/bin/env bash
# Builds Fairy-Stockfish as Sinsan's offline teacher/referee engine, pinned to a specific commit for
# reproducibility (the project has had no fresh tag since 2021 and develops on an untagged master).
# Never vendors the engine into the repo: source and binary live in training/teacher/engine/, which
# is gitignored, since Fairy-Stockfish (GPL-3.0) is used only as a separate offline process.
#
# Also fetches the Janggi NNUE net into training/teacher/nnue/ (gitignored). Its license is
# unresolved (see docs/licenses.md): local label generation only, never redistributed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$REPO_ROOT/training/teacher/engine/Fairy-Stockfish"
NNUE_DIR="$REPO_ROOT/training/teacher/nnue"
PINNED_COMMIT="c19b5f6c66894fdb0e88d0dd100e3885f744760a"
NNUE_NAME="janggi-9991472750de.nnue"
NNUE_URL="https://raw.githubusercontent.com/fairy-stockfish/Fairy-Stockfish-NNUE/master/$NNUE_NAME"
# vnni512 suits AVX-512 VNNI CPUs (e.g. Zen 4 EPYC); fall back to x86-64-bmi2 on older hosts.
ARCH="${SINSAN_TEACHER_ARCH:-x86-64-vnni512}"

if [ ! -d "$ENGINE_DIR/.git" ]; then
  mkdir -p "$ENGINE_DIR"
  git -C "$ENGINE_DIR" init -q
  git -C "$ENGINE_DIR" remote add origin https://github.com/fairy-stockfish/Fairy-Stockfish.git
fi
# Shallow fetch of exactly the pinned commit - a full clone is slow on constrained networks.
git -C "$ENGINE_DIR" fetch -q --depth 1 origin "$PINNED_COMMIT"
git -C "$ENGINE_DIR" checkout -q "$PINNED_COMMIT"

echo "Building with ARCH=$ARCH largeboards=yes (override via SINSAN_TEACHER_ARCH)..."
make -C "$ENGINE_DIR/src" -j"$(nproc)" build ARCH="$ARCH" largeboards=yes > /dev/null

if [ ! -f "$NNUE_DIR/$NNUE_NAME" ]; then
  mkdir -p "$NNUE_DIR"
  curl -sSL -o "$NNUE_DIR/$NNUE_NAME" "$NNUE_URL"
fi
# The file name embeds the net's sha256 prefix - verify it rather than trusting the download.
EXPECTED="${NNUE_NAME#janggi-}"; EXPECTED="${EXPECTED%.nnue}"
ACTUAL="$(sha256sum "$NNUE_DIR/$NNUE_NAME" | cut -c1-12)"
[ "$ACTUAL" = "$EXPECTED" ] || { echo "NNUE checksum mismatch: $ACTUAL != $EXPECTED" >&2; exit 1; }

echo "Built: $ENGINE_DIR/src/stockfish ($(sha256sum "$ENGINE_DIR/src/stockfish" | cut -c1-16))"
echo "Verifying janggi + NNUE..."
printf 'uci\nsetoption name UCI_Variant value janggi\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo depth 6\n' \
  "$NNUE_DIR/$NNUE_NAME" | (cat; sleep 2) | "$ENGINE_DIR/src/stockfish" | grep -E 'NNUE evaluation|bestmove'
