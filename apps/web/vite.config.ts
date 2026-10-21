import { defineConfig } from 'vite';

// Cross-origin isolation lets the in-browser engine (onnxruntime-web) use several threads.
const isolation = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };

// /api is forwarded to training/play_server.py (the full-size GPU checkpoint) when it is running.
const proxy = { '/api': 'http://127.0.0.1:8765' };

export default defineConfig({
  server: { proxy, headers: isolation },
  preview: { proxy, headers: isolation },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web', '@sinsan/engine-wasm'] },
});
