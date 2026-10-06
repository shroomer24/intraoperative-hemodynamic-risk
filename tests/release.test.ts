import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { headersForPath, publicOrigin, releaseFiles } from '../tools/release_assets';
const headers=readFileSync('release/_headers','utf8');
describe('static release policy',()=>{
  it('uses a public HTTPS origin without credentials or private addresses',()=>{
    expect(publicOrigin('https://intraop-hemodynamic-risk.pages.dev')).toBe('https://intraop-hemodynamic-risk.pages.dev');
    for(const value of ['http://example.com','https://localhost','https://127.0.0.1','https://10.0.0.1','https://192.168.0.1','https://user:pass@example.com','https://example.com/demo','https://example.com/?secret=a'])expect(()=>publicOrigin(value)).toThrow();
  });
  it.each(['/','/index.html','/demo','/demo/','/demo.html','/404.html'])('keeps %s revalidatable',path=>expect(headersForPath(headers,path)['Cache-Control']).toBe('no-cache'));
  it.each(['/assets/index-12345678.js','/replay-v01/cases/case-019/signals.json'])('immutably caches only versioned %s',path=>expect(headersForPath(headers,path)['Cache-Control']).toBe('public, max-age=31536000, immutable'));
  it('has no global cache collision and maintains strict scripts with functional style attributes',()=>{
    const policy=headersForPath(headers,'/unknown');expect(policy['Cache-Control']).toBeUndefined();
    expect(policy['Content-Security-Policy']).toContain("script-src 'self'");expect(policy['Content-Security-Policy']).not.toContain('unsafe-eval');
    expect(policy['Content-Security-Policy']).toContain("style-src-attr 'unsafe-inline'");expect(policy['Content-Security-Policy']).toContain("frame-ancestors 'none'");
    expect(policy['X-Content-Type-Options']).toBe('nosniff');expect(policy['Referrer-Policy']).toBe('no-referrer');expect(policy['Permissions-Policy']).toBe('camera=(), microphone=(), geolocation=()');
  });
  it('uses a dedicated extensionless demo entry and disables the generic SPA fallback with a 404 document',()=>{
    expect(readFileSync('release/_redirects','utf8').trim()).toBe('/demo/ /demo 301');
    expect(readFileSync('tools/release_assets.ts','utf8')).toContain("fileName: 'demo.html'");
    expect(readFileSync('release/404.html','utf8')).toContain('Page unavailable');
    expect(releaseFiles).not.toContain('site.json');expect(releaseFiles).not.toContain('source_crosswalk.json');
  });
  it('publishes exact neutral metadata, a real UI image and the existing crosshair',()=>{
    const html=readFileSync('index.html','utf8');expect(html).toContain('Intraoperative Hemodynamic Risk — Historical Replay');
    expect(html).toContain('Historical replay of frozen held-out predictions for new sustained intraoperative hypotension within five minutes.');
    expect(html).toContain('summary_large_image');expect(html).toContain('content="website"');expect(html).toContain('social-preview-v01.png');
    const svg=readFileSync('release/favicon.svg','utf8');const header=readFileSync('src/components/ProjectTitle.tsx','utf8'); for(const path of ['M11 4H4v7M21 4h7v7M28 21v7h-7M11 28H4v-7','M16 10v12M10 16h12']) { expect(svg).toContain(path); expect(header).toContain(path); }
    expect(readFileSync('vite.config.ts','utf8')).toContain('sourcemap: false');expect(readFileSync('wrangler.jsonc','utf8')).toContain('"pages_build_output_dir": "./dist"');
  });
});
