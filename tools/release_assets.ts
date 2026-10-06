import { existsSync, readFileSync } from 'node:fs';
import type { Plugin } from 'vite';
import { decodedPath, metadataPaths } from './replay_asset_boundary.ts';

export const releaseFiles = ['_headers', '_redirects', '404.html', 'not-found.css', 'favicon.svg',
  'favicon.ico', 'apple-touch-icon.png', 'social-preview-v01.png'] as const;
const types: Record<string, string> = { html: 'text/html; charset=utf-8', css: 'text/css; charset=utf-8',
  svg: 'image/svg+xml', png: 'image/png', ico: 'image/x-icon' };
export function publicOrigin(value: unknown): string {
  if (typeof value !== 'string') throw Error('Public origin is required.');
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.username || url.password || url.port || url.pathname !== '/' || url.search || url.hash
    || !/^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/.test(url.hostname) || /localhost|^127\.|^10\.|^192\.168\.|\.local$/.test(url.hostname)) throw Error('A public HTTPS origin is required.');
  return url.origin;
}
export function headersForPath(text: string, path: string): Record<string, string> {
  const result: Record<string, string> = {}; let matches = false;
  for (const line of text.split('\n')) {
    if (!line.trim() || line.trim().startsWith('#')) continue;
    if (!/^\s/.test(line)) {
      const expression = line.trim().split('*').map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*');
      matches = new RegExp(`^${expression}$`).test(path);
    } else if (matches) {
      const colon = line.indexOf(':');
      if (colon < 0) throw Error('Invalid release header.');
      result[line.slice(0, colon).trim()] = line.slice(colon + 1).trim();
    }
  }
  return result;
}
export function releaseAssets(): Plugin {
  const root = new URL('../release/', import.meta.url);
  const read = (name: string) => readFileSync(new URL(name, root));
  const origin = publicOrigin(JSON.parse(read('site.json').toString()).publicOrigin);
  const exposed = (path: string) => releaseFiles.find(name => !name.startsWith('_') && path === `/${name}`);
  return {
    name: 'static-public-release',
    transformIndexHtml(html) { return html.replaceAll('__PUBLIC_ORIGIN__', origin); },
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const name = exposed(decodedPath(request.url));
        if (!name || !existsSync(new URL(name, root))) return next();
        response.setHeader('Content-Type', types[name.split('.').at(-1)!]); response.end(read(name));
      });
    },
    configurePreviewServer(server) {
      const headerText = read('_headers').toString();
      server.middlewares.use((request, response, next) => {
        const path = decodedPath(request.url);
        for (const [key, value] of Object.entries(headersForPath(headerText, path))) response.setHeader(key, value);
        if (path === '/demo' || path === '/demo/') { request.url = '/index.html'; return next(); }
        const approved = path === '/' || path === '/index.html' || path === '/THIRD_PARTY_NOTICES.txt'
          || metadataPaths.some(name => path === `/${name}`) || Boolean(exposed(path)) || /^\/assets\/[a-zA-Z0-9_-]+\.(js|css)$/.test(path);
        if (approved) return next();
        response.statusCode = 404; response.setHeader('Content-Type', types.html); response.end(read('404.html'));
      });
    },
    generateBundle: { order: 'post', handler(_options, bundle) {
      // Cloudflare serves demo.html at /demo without canonicalizing to root.
      const entry = bundle['index.html'];
      if (!entry || entry.type !== 'asset') throw Error('Built entry HTML is required.');
      this.emitFile({ type: 'asset', fileName: 'demo.html', source: entry.source });
      for (const name of releaseFiles) if (existsSync(new URL(name, root))) this.emitFile({ type: 'asset', fileName: name, source: read(name) });
    } },
  };
}
