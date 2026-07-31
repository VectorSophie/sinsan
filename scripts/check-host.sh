#!/usr/bin/env bash
# Reports the actual measured host profile for Sinsan development/training.
# Linux Mint is the official environment; this script also runs on other
# Linux distros for informational purposes but only Linux Mint is supported.
set -euo pipefail

section() { printf '\n== %s ==\n' "$1"; }

section "Operating System"
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "Distro: ${PRETTY_NAME:-unknown}"
  if [ "${ID:-}" != "linuxmint" ]; then
    echo "WARNING: official Sinsan dev/training environment is Linux Mint. Detected: ${ID:-unknown}"
  fi
else
  echo "WARNING: /etc/os-release not found; cannot confirm distro"
fi
echo "Kernel: $(uname -srm)"

section "CPU"
if command -v lscpu >/dev/null 2>&1; then
  MODEL=$(lscpu | awk -F': +' '/Model name/{print $2; exit}')
  SOCKETS=$(lscpu | awk -F': +' '/Socket\(s\)/{print $2; exit}')
  CORES_PER_SOCKET=$(lscpu | awk -F': +' '/Core\(s\) per socket/{print $2; exit}')
  THREADS_PER_CORE=$(lscpu | awk -F': +' '/Thread\(s\) per core/{print $2; exit}')
  echo "Model: ${MODEL:-unknown}"
  echo "Physical cores: $(( ${SOCKETS:-1} * ${CORES_PER_SOCKET:-0} ))"
  echo "Logical threads: $(nproc)"
  echo "Threads per core: ${THREADS_PER_CORE:-unknown}"
else
  echo "lscpu not found; logical threads: $(nproc)"
fi

section "CPU instruction flags (relevant to Fairy-Stockfish build selection)"
if [ -f /proc/cpuinfo ]; then
  FLAGS=$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2)
  for f in avx2 bmi2 avx512f sse4_2 popcnt; do
    if echo "$FLAGS" | grep -qw "$f"; then
      echo "$f: yes"
    else
      echo "$f: no"
    fi
  done
else
  echo "WARNING: /proc/cpuinfo not found"
fi

section "Memory"
free -h

section "Storage (project volume)"
df -h "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

section "Thermal state"
if command -v sensors >/dev/null 2>&1; then
  sensors 2>/dev/null | grep -E 'Package id 0|Core 0|temp1' || echo "sensors installed but no matching output; run 'sensors' manually"
else
  echo "lm-sensors not installed. Install with: sudo apt install lm-sensors && sudo sensors-detect"
  echo "During sustained teacher-engine or training benchmarks, run 'watch -n2 sensors' in another terminal"
  echo "and watch for thermal throttling (frequency drop under sustained load: 'watch -n1 \"lscpu | grep MHz\"')."
fi

section "Toolchain"
for bin in git python3 uv node npm pnpm cargo g++ cmake; do
  if command -v "$bin" >/dev/null 2>&1; then
    printf '%-8s: %s\n' "$bin" "$("$bin" --version 2>&1 | head -1)"
  else
    printf '%-8s: NOT FOUND\n' "$bin"
  fi
done

section "PyTorch"
python3 -c "
try:
    import torch
    print('torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('MKL enabled:', torch.backends.mkl.is_available())
    print('oneDNN (mkldnn) enabled:', torch.backends.mkldnn.is_available())
except ImportError:
    print('torch: NOT INSTALLED')
" 2>&1

section "Chromium"
for bin in chromium chromium-browser google-chrome; do
  if command -v "$bin" >/dev/null 2>&1; then
    "$bin" --version 2>&1
    break
  fi
done || echo "No Chromium/Chrome binary found on PATH"

section "Summary notes"
echo "This report reflects the ACTUAL machine at the time of running it, not the"
echo "project's planning assumptions. Re-run before any benchmark or training run,"
echo "since available RAM/thermal headroom can differ significantly from a cold-boot baseline."
