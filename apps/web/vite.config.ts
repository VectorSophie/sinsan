import { defineConfig } from 'vite';

// /api is forwarded to training/play_server.py (the full-size GPU checkpoint) when it is running.
export default defineConfig({
  server: {
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
});
