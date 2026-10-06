import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import type { Plugin } from 'vite';

import { decodedPath, metadataPaths, rejectExcludedAsset } from './tools/replay_asset_boundary.ts';
import { releaseAssets } from './tools/release_assets.ts';
const appRoot = fileURLToPath(new URL('.', import.meta.url));

function metadataAssets(): Plugin {
  return {
    name: 'phase-six-argent-paper',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (rejectExcludedAsset(request, response, appRoot)) return;
        const pathname = decodedPath(request.url);
        if (!pathname.startsWith('/replay-v01/')) return next();
        const asset = metadataPaths.find(path => pathname === `/${path}`);
        if (!asset) {
          response.statusCode = 404;
          response.end('This asset is unavailable in this replay phase.');
          return;
        }
        response.setHeader('Content-Type', 'application/json; charset=utf-8');
        response.setHeader('Cache-Control', 'no-cache');
        response.end(readFileSync(new URL(`./public/${asset}`, import.meta.url)));
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((request, response, next) => {
        if (!rejectExcludedAsset(request, response, appRoot)) next();
      });
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'THIRD_PARTY_NOTICES.txt',
        source: readFileSync(new URL('./THIRD_PARTY_NOTICES.md', import.meta.url)) });
      for (const path of metadataPaths) {
        this.emitFile({
          type: 'asset', fileName: path,
          source: readFileSync(new URL(`./public/${path}`, import.meta.url)),
        });
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), releaseAssets(), metadataAssets()],
  publicDir: false,
  build: { sourcemap: false },
  server: {
    host: '127.0.0.1', port: 5173, strictPort: true,
    fs: { strict: true, allow: [fileURLToPath(new URL('.', import.meta.url))] },
  },
  preview: { host: '127.0.0.1', port: 4173, strictPort: true },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
    setupFiles: ['tests/setup.ts'],
    restoreMocks: true,
  },
});
