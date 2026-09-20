/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Type-only reference above (not an import from 'vitest/config') keeps
// `vitest` out of this file's runtime dependency graph: `vite preview` in the
// production container loads this config and must not require `vitest`,
// which is a devDependency only.

// FRONTEND_PORT is shared with docker-compose.yml; VITE_FRONTEND_API_BASE_URL
// and VITE_POLL_INTERVAL_MS are read by application code via import.meta.env
// and are baked in at build time (see docker-compose.yml build.args).
const port = Number(process.env.FRONTEND_PORT ?? 3000);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port,
  },
  preview: {
    port,
    host: true,
    strictPort: true,
  },
  test: {
    environment: 'jsdom',
    passWithNoTests: true,
  },
});
