#!/usr/bin/env bash
# Builds the engine for the browser into packages/engine-wasm/pkg (needs the wasm32-unknown-unknown
# target and wasm-bindgen-cli matching Cargo.lock's wasm-bindgen version).
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release --target wasm32-unknown-unknown --features wasm --lib
wasm-bindgen --target web --out-dir ../packages/engine-wasm/pkg target/wasm32-unknown-unknown/release/sinsan_engine.wasm
