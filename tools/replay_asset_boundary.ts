import type { IncomingMessage, ServerResponse } from 'node:http';

// Phase 5 exposes frozen forecasts and events; anchor-status stays excluded.
// emitFile copies them into dist without changing any Phase 0 source file.
export const metadataPaths = [
  'replay-v01/manifest.json',
  'replay-v01/cases/index.json',
  ...Array.from({ length: 22 }, (_, i) => ['signals.json', 'prediction-windows.json', 'events.json'].map(name =>
    `replay-v01/cases/case-${String(i + 1).padStart(3, '0')}/${name}`)).flat(),
] as const;

export function decodedPath(url: string | undefined): string {
  try { return decodeURIComponent(url?.split('?')[0] ?? ''); }
  catch { return '/replay-v01/invalid'; }
}
export function rejectExcludedAsset(request: IncomingMessage, response: ServerResponse, appRoot: string): boolean {
  const pathname = decodedPath(request.url);
  if ((pathname.startsWith('/@fs/') && !pathname.startsWith(`/@fs${appRoot}`))
    || ['/work/', '/artifacts/', '/data/', '/intraop-prediction/'].some(prefix => pathname.startsWith(prefix))) {
    response.statusCode = 403;
    response.end('This path is unavailable in this replay phase.');
    return true;
  }
  if (pathname.includes('/replay-v01/') && !metadataPaths.some(path => pathname === `/${path}`)) {
    // Also reject /public/ and /@fs/ aliases rather than exposing excluded exports.
    response.statusCode = 404;
    response.end('This asset is unavailable in this replay phase.');
    return true;
  }
  return false;
}

