import { webcrypto, createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchReplayText } from '../src/data/assetIntegrity';
import { loadCaseSignals, signalAssetPath } from '../src/data/caseSignals';
import { validateSignals } from '../src/data/signalValidation';
import { initialViewport, availability, timeSplits } from '../src/charts/synchronizedAxes';
import { metadata, manifest, index } from './metadata.fixture';
import { validateShellMetadata } from '../src/data/loader';
import { signalFixture } from './signals.fixture';

beforeEach(() => vi.stubGlobal('crypto', webcrypto));
afterEach(() => vi.unstubAllGlobals());
const fixture = signalFixture();

describe('signal integrity and schema', () => {
  it('validates all 22 frozen signal files against their manifest hashes', () => {
    for (const selectedCase of metadata.cases) {
      const { text, signals } = signalFixture(selectedCase.ordinal);
      expect(createHash('sha256').update(text).digest('hex')).toBe(selectedCase.signalAsset.sha256);
      expect(signals.timeSeconds.length).toBe(selectedCase.durationSeconds + 1);
    }
  });
  it('preserves exported values, null gaps and observation ages by identity', () => {
    expect(fixture.signals.channels.map.values).toBe(fixture.raw.channels.map.values);
    expect(fixture.signals.channels.map.age_seconds).toBe(fixture.raw.channels.map.age_seconds);
    expect(fixture.signals.timeSeconds).toBe(fixture.raw.time_seconds);
    expect(fixture.signals.channels.map.values[81]).toBeNull();
  });
  it.each(['NaN', '+Infinity', '-Infinity', 'string'])('rejects %s values', name => {
    const raw = structuredClone(fixture.raw);
    raw.channels.map.values[0] = { NaN, '+Infinity': Infinity, '-Infinity': -Infinity, string: 'bad' }[name];
    expect(() => validateSignals(raw, fixture.selectedCase)).toThrow();
  });
  it('rejects unequal signal and age lengths', () => {
    const raw = structuredClone(fixture.raw); raw.channels.hr.values.pop();
    expect(() => validateSignals(raw, fixture.selectedCase)).toThrow();
    const age = structuredClone(fixture.raw); age.channels.hr.age_seconds.pop();
    expect(() => validateSignals(age, fixture.selectedCase)).toThrow();
  });
  it('rejects nonfinite/negative ages without applying freshness again', () => {
    for (const value of [NaN, Infinity, -1]) {
      const raw = structuredClone(fixture.raw); raw.channels.hr.age_seconds[0] = value;
      expect(() => validateSignals(raw, fixture.selectedCase)).toThrow();
    }
  });
  it('rejects duplicate, reversed, skipped or absolute time coordinates', () => {
    for (const value of [0, -1, 2, 1700000000, Infinity]) {
      const raw = structuredClone(fixture.raw); raw.time_seconds[1] = value;
      expect(() => validateSignals(raw, fixture.selectedCase)).toThrow();
    }
  });
  it('rejects private extra fields, missing channels and incorrect units', () => {
    const extra = structuredClone(fixture.raw); extra.subject_id = 'forbidden';
    expect(() => validateSignals(extra, fixture.selectedCase)).toThrow();
    const missing = structuredClone(fixture.raw); delete missing.channels.sbp;
    expect(() => validateSignals(missing, fixture.selectedCase)).toThrow();
    const units = structuredClone(fixture.raw); units.channels.map.unit = '%';
    expect(() => validateSignals(units, fixture.selectedCase)).toThrow();
  });
  it('rejects a signal pointer or hash not bound by the manifest', () => {
    const bad = structuredClone(index); bad[0].assets.signals = 'https://example.com/signals.json';
    expect(() => validateShellMetadata(manifest, bad)).toThrow();
    const changed = structuredClone(manifest); changed.presentation_files_sha256['cases/case-001/signals.json'] = 'bad';
    expect(() => validateShellMetadata(changed, index)).toThrow();
  });
});

describe('selected-operation loading boundary', () => {
  it('fetches only the selected signals file, with cancellation and redirects disabled', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, text: async () => fixture.text }));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    const result = await loadCaseSignals(fixture.selectedCase, controller.signal);
    expect(result.ordinal).toBe('case-004');
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith('/replay-v01/cases/case-004/signals.json', {
      signal: controller.signal, cache: 'no-cache', redirect: 'error',
    });
  });
  it.each(['../case-001', 'case-023', 'case-001/../../', 'https://example.com', 'case-001?x=1'])('rejects unsafe ordinal %s before a fetch', ordinal => {
    expect(() => signalAssetPath(ordinal)).toThrow();
  });
  it('rejects tampered signals before validation or publication', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, text: async () => fixture.text + ' ' })));
    await expect(loadCaseSignals(fixture.selectedCase, new AbortController().signal)).rejects.toThrow();
  });
  it('rejects cancellation even when a transport ignores AbortSignal', async () => {
    const controller = new AbortController();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, text: async () => { controller.abort(); return fixture.text; } })));
    await expect(loadCaseSignals(fixture.selectedCase, controller.signal)).rejects.toThrow();
  });
  it.each(['https://example.com/signals.json', '/replay-v01/cases/case-004/anchor-status.json', '/replay-v01/../private.json'])('rejects forbidden network path %s', async path => {
    const fetchMock = vi.fn(); vi.stubGlobal('fetch', fetchMock);
    await expect(fetchReplayText(path)).rejects.toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('fixed synchronized span', () => {
  it('retains every exported second and every null in the first 15 minutes', () => {
    const view = initialViewport(fixture.signals);
    expect(view.domain).toEqual([0, 900]);
    expect(view.timeSeconds).toEqual(fixture.raw.time_seconds.slice(0, 901));
    for (const id of ['map', 'hr', 'spo2', 'etco2'] as const) {
      expect(view.values[id]).toEqual(fixture.raw.channels[id].values.slice(0, 901));
    }
    expect(availability(view.values.map)).toEqual({ available: 842, missing: 59, total: 901, gaps: 2 });
  });
  it('resets to zero for long and short cases and clips only the domain to short-operation bounds', () => {
    for (const ordinal of ['case-010', 'case-019']) expect(initialViewport(signalFixture(ordinal).signals).domain).toEqual([0, 900]);
    // Boundary logic only: a truncated real prefix stands in for an operation shorter than 15 min.
    const prefix = { ...fixture.signals, timeSeconds: fixture.signals.timeSeconds.slice(0, 301) };
    expect(initialViewport(prefix).domain).toEqual([0, 300]);
  });
  it('uses shared numeric time splits, independent of wall-clock time', () => {
    expect(timeSplits([0, 900], 800)).toEqual([0, 180, 360, 540, 720, 900]);
    expect(timeSplits([0, 900], 400)).toEqual([0, 300, 600, 900]);
    expect(timeSplits([0, 900], 160)).toEqual([0, 900]);
  });
  it('contains no scientific model runtime dependencies or source imports', () => {
    const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
    expect(Object.keys(pkg.dependencies).sort()).toEqual(['@paper-design/shaders', '@paper-design/shaders-react', 'argentui', 'react', 'react-dom', 'uplot']);
    expect(readFileSync('src/data/caseSignals.ts', 'utf8')).not.toMatch(/tabpfn-client|pickle|xgboost|intraop-prediction|\.env/);
  });
});
