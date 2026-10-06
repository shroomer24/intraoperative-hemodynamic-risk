import { webcrypto } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CASE_INDEX_PATH, loadShellMetadata, MANIFEST_PATH, randomCase, validateShellMetadata } from '../src/data/loader';
import { manifest, index, metadata, manifestText, indexText } from './metadata.fixture';

beforeEach(() => vi.stubGlobal('crypto', webcrypto));
afterEach(() => vi.unstubAllGlobals());

describe('verified presentation metadata', () => {
  it('loads exactly the manifest and case index', async () => {
    const fetchMock = vi.fn(async (path: string) => ({ ok: true, text: async () => path === MANIFEST_PATH ? manifestText : indexText }));
    vi.stubGlobal('fetch', fetchMock);
    expect(await loadShellMetadata()).toEqual(metadata);
    expect(fetchMock.mock.calls.map(call => call[0])).toEqual([MANIFEST_PATH, CASE_INDEX_PATH]);
  });
  it('exposes exactly 22 ordered safe cases', () => {
    expect(metadata.cases).toHaveLength(22);
    expect(metadata.cases.map(item => item.ordinal)).toEqual(Array.from({ length: 22 }, (_, i) => `case-${String(i + 1).padStart(3, '0')}`));
  });
  it('retains real ordinals, durations and only verified signal, prediction and event pointers', () => {
    for (const [i, item] of metadata.cases.entries()) {
      expect(Object.keys(item)).toEqual(['ordinal', 'durationSeconds', 'signalAsset', 'predictionAsset', 'eventAsset']);
      expect(item.durationSeconds).toBe(index[i].duration_seconds);
    }
    expect(JSON.stringify(metadata)).not.toMatch(/subject_id|case_id|positive_window_count|episode_count|assets|channel_coverage/);
  });
  it('preserves the seven scientific model identities and feature counts', () => {
    expect(Object.fromEntries(metadata.models.map(model => [model.id, model.featureCount]))).toEqual({ prevalence: 0, current_map: 1, logistic_map: 18, logistic_full: 74, xgboost: 74, tabpfn_map: 18, tabpfn_full: 74 });
  });
  it('rejects a changed manifest before fetching the index', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, text: async () => manifestText + ' ' }));
    vi.stubGlobal('fetch', fetchMock);
    await expect(loadShellMetadata()).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
  it('rejects a changed case index', async () => {
    vi.stubGlobal('fetch', vi.fn(async (path: string) => ({ ok: true, text: async () => path === MANIFEST_PATH ? manifestText : indexText + ' ' })));
    await expect(loadShellMetadata()).rejects.toThrow('Presentation metadata could not be verified.');
  });
  it('rejects unsafe ordinals', () => {
    const bad = structuredClone(index); bad[0].case_ordinal = 'unsafe';
    expect(() => validateShellMetadata(manifest, bad)).toThrow();
  });
  it('rejects invalid duration and changed feature contracts', () => {
    const bad = structuredClone(index); bad[0].duration_seconds = NaN;
    expect(() => validateShellMetadata(manifest, bad)).toThrow();
    const changed = structuredClone(manifest); changed.models[2].feature_count = 74;
    expect(() => validateShellMetadata(changed, index)).toThrow();
  });
  it('rejects network failure without returning unchecked metadata', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false })));
    await expect(loadShellMetadata()).rejects.toThrow('Replay asset unavailable.');
  });
});

describe('whole-operation random selection', () => {
  it('always chooses one valid ordinal and changes the current selection', () => {
    let selected = metadata.cases[0].ordinal;
    for (let i = 0; i < 1000; i++) {
      const next = randomCase(metadata.cases, selected);
      expect(metadata.cases.some(item => item.ordinal === next)).toBe(true);
      expect(next).not.toBe(selected);
      selected = next;
    }
  });
  it('handles a singleton and rejects an empty case set', () => {
    expect(randomCase([metadata.cases[0]], metadata.cases[0].ordinal)).toBe('case-001');
    expect(() => randomCase([], 'case-001')).toThrow();
  });
});
