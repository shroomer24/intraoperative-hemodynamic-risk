import { webcrypto, createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type uPlot from 'uplot';
import { validatePredictions } from '../src/data/predictionValidation';
import { loadCasePredictions, predictionAssetPath } from '../src/data/casePredictions';
import { anchorAt, anchorNavigation, verifiedCasePredictions } from '../src/predictions/anchorLookup';
import { forecastValue, formatForecast } from '../src/predictions/representations';
import { historyBounds, historySamples } from '../src/predictions/history';
import { riskOptions, riskRange } from '../src/charts/riskOptions';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
import { predictionFixture } from './predictions.fixture';
import { metadata } from './metadata.fixture';

beforeEach(() => vi.stubGlobal('crypto', webcrypto));
afterEach(() => vi.unstubAllGlobals());
const fixture = predictionFixture();
const windows = fixture.predictions.windows;

describe('sealed forecast integrity and forecast-only schema', () => {
  it('validates all 22 manifest-bound files and 3146 retained anchors without retaining ground truth', () => {
    let count = 0;
    for (const selectedCase of metadata.cases) {
      const { text, predictions } = predictionFixture(selectedCase.ordinal);
      expect(createHash('sha256').update(text).digest('hex')).toBe(selectedCase.predictionAsset.sha256);
      expect(predictions.ordinal).toBe(selectedCase.ordinal);
      count += predictions.windows.length;
      expect(JSON.stringify(predictions)).not.toMatch(/ground_truth|label|episode|onset|subject_id|case_id/);
    }
    expect(count).toBe(3146);
  });
  it('never reads even a poisoned ground_truth property and strips all non-forecast metadata', () => {
    const raw = structuredClone(fixture.raw);
    for (const row of raw) Object.defineProperty(row, 'ground_truth', { enumerable: true, get: () => { throw new Error('Outcome consumed'); } });
    const result = validatePredictions(raw, fixture.selectedCase);
    expect(Object.keys(result.windows[0])).toEqual(['query_position', 'anchor_seconds', 'horizon_start_exclusive_seconds', 'horizon_end_inclusive_seconds', 'model_outputs']);
    expect(JSON.stringify(result)).not.toMatch(/ground_truth|history_start|future_observation|matched_episode/);
  });
  it.each(['subject_id', 'case_id', 'private', 'patient_identifier'])('rejects unexpected/private row field %s without accessing its value', field => {
    const raw = structuredClone(fixture.raw); raw[0][field] = 'not permitted';
    expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow('Frozen forecasts unavailable.');
  });
  it('rejects private/unknown output fields and unknown model identifiers', () => {
    const raw = structuredClone(fixture.raw); raw[0].model_outputs.xgboost.subject_id = 'forbidden';
    expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
    const models = structuredClone(fixture.raw); models[0].model_outputs.alternative_model = {};
    expect(() => validatePredictions(models, fixture.selectedCase)).toThrow();
  });
  it('rejects duplicate/noninteger/out-of-range query positions and non-increasing anchors', () => {
    for (const value of [fixture.raw[0].query_position, -1, 3146, 1.5, NaN]) {
      const raw = structuredClone(fixture.raw); raw[1].query_position = value;
      expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
    }
    for (const anchor of [461, 460, NaN, Infinity, 461.5]) {
      const raw = structuredClone(fixture.raw); raw[1].anchor_seconds = anchor;
      expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
    }
  });
  it('rejects changed eligibility, history/horizon contract or out-of-operation future bounds', () => {
    for (const field of ['eligible', 'history_start_seconds', 'history_end_seconds', 'horizon_start_exclusive_seconds', 'horizon_end_inclusive_seconds', 'future_observation_end_seconds']) {
      const raw = structuredClone(fixture.raw); raw[0][field] = field === 'eligible' ? false : raw[0][field] + 1;
      expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
    }
    expect(() => validatePredictions(fixture.raw, { ...fixture.selectedCase, durationSeconds: 820 })).toThrow();
  });
  it.each([NaN, Infinity, -Infinity, -.1, 1.1, '0.5'])('rejects invalid probability representation %s', value => {
    const raw = structuredClone(fixture.raw); raw[0].model_outputs.tabpfn_full.calibrated_probability = value;
    expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
  });
  it('requires finite score and exact -map_latest identity and unit', () => {
    const raw = structuredClone(fixture.raw); raw[0].model_outputs.current_map.raw_score = Infinity;
    expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
    raw[0].model_outputs.current_map.raw_score = -94; raw[0].model_outputs.current_map.score_definition = 'map_latest';
    expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
    raw[0].model_outputs.current_map.score_definition = '-map_latest'; raw[0].model_outputs.current_map.unit = '%';
    expect(() => validatePredictions(raw, fixture.selectedCase)).toThrow();
  });
  it('explicitly treats blank, null, absent representations/models as unavailable and never as zero', () => {
    const raw = structuredClone(fixture.raw);
    raw[0].model_outputs.xgboost.calibrated_probability = '';
    delete raw[0].model_outputs.logistic_map.calibrated_probability;
    raw[0].model_outputs.tabpfn_map = null;
    delete raw[0].model_outputs.logistic_full;
    const row = validatePredictions(raw, fixture.selectedCase).windows[0];
    for (const id of ['xgboost', 'logistic_map', 'tabpfn_map', 'logistic_full'] as const) expect(forecastValue(row, id)).toBeUndefined();
    expect(forecastValue(row, 'xgboost', 'raw')?.value).toBe(fixture.raw[0].model_outputs.xgboost.raw_probability);
    raw[0].model_outputs.xgboost.calibrated_probability = 0;
    expect(forecastValue(validatePredictions(raw, fixture.selectedCase).windows[0], 'xgboost')?.value).toBe(0);
  });
  it('loads only the correct selected-case asset with hash, cancellation and redirects disabled', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, text: async () => fixture.text })); vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    expect(await loadCasePredictions(fixture.selectedCase, controller.signal)).toEqual(fixture.predictions);
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith('/replay-v01/cases/case-004/prediction-windows.json', { signal: controller.signal, cache: 'no-cache', redirect: 'error' });
  });
  it('rejects tampering or cancellation before publication', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, text: async () => fixture.text + ' ' })));
    await expect(loadCasePredictions(fixture.selectedCase, new AbortController().signal)).rejects.toThrow();
    const controller = new AbortController();
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, text: async () => { controller.abort(); return fixture.text; } })));
    await expect(loadCasePredictions(fixture.selectedCase, controller.signal)).rejects.toThrow();
  });
  it('rejects mismatched pointers, unsafe paths and unverified case/signal sessions', async () => {
    const fetchMock = vi.fn(); vi.stubGlobal('fetch', fetchMock);
    await expect(loadCasePredictions({ ...fixture.selectedCase, predictionAsset: { ...fixture.selectedCase.predictionAsset, path: 'cases/case-019/prediction-windows.json' } }, new AbortController().signal)).rejects.toThrow();
    for (const id of ['case-023', '../case-004', 'case-004?x=1', 'https://example.com']) expect(() => predictionAssetPath(id)).toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
    const store = new ReplayStore('case-004', 7500, new ManualScheduler());
    const load = { status: 'ready' as const, ordinal: fixture.predictions.ordinal, data: fixture.predictions };
    expect(verifiedCasePredictions(load, store.getSnapshot())).toBeUndefined(); store.activate();
    expect(verifiedCasePredictions(load, store.getSnapshot())).toBe(fixture.predictions);
    expect(verifiedCasePredictions(load, { ...store.getSnapshot(), caseOrdinal: 'case-019' })).toBeUndefined();
  });
});

describe('causal anchors, fixed horizons, representations and discrete history', () => {
  it('selects only latest anchor <= floor(time), including before/exact/between/gapped/end states', () => {
    expect(anchorAt(windows, 460.99)).toBeUndefined();
    expect(anchorAt(windows, 461)).toBe(windows[0]);
    expect(anchorAt(windows, 498)).toBe(windows[0]);
    expect(anchorAt(windows, 520.99)).toBe(windows[0]);
    expect(anchorAt(windows, 521)).toBe(windows[1]);
    expect(anchorAt(windows, 1600)?.anchor_seconds).toBe(1481);
    expect(anchorAt(windows, 7500)).toBe(windows.at(-1));
  });
  it('preserves original exclusive/inclusive horizon between anchors; no sliding or value mutation', () => {
    const row = anchorAt(windows, 498)!;
    expect(row.anchor_seconds).toBe(461); expect(row.horizon_start_exclusive_seconds).toBe(461);
    expect(row.horizon_end_inclusive_seconds).toBe(761);
    expect(anchorAt(windows, 519)).toBe(row);
    expect(forecastValue(row, 'tabpfn_full')?.value).toBe(fixture.raw[0].model_outputs.tabpfn_full.calibrated_probability);
  });
  it('uses exact retained calibrated defaults and exact retained raw values for every learned model', () => {
    for (const id of ['logistic_map', 'logistic_full', 'xgboost', 'tabpfn_map', 'tabpfn_full'] as const) {
      expect(forecastValue(windows[0], id)).toEqual({ kind: 'probability', representation: 'calibrated_probability', value: fixture.raw[0].model_outputs[id].calibrated_probability });
      expect(forecastValue(windows[0], id, 'raw')).toEqual({ kind: 'probability', representation: 'raw_probability', value: fixture.raw[0].model_outputs[id].raw_probability });
    }
    expect(forecastValue(windows[0], 'prevalence')?.representation).toBe('raw_probability');
    expect(forecastValue(windows[0], 'current_map', 'calibrated')).toEqual({ kind: 'score', value: -94, representation: 'raw_score' });
    expect(formatForecast(forecastValue(windows[0], 'current_map'))).toBe('-94');
    expect(formatForecast(forecastValue(windows[0], 'current_map'))).not.toContain('%');
    expect(formatForecast(undefined)).toBe('—');
  });
  it('navigates actual earlier/later retained anchors, including a gap, without fabricated minutes', () => {
    expect(anchorNavigation(windows, 0)).toEqual({ previous: undefined, next: windows[0] });
    expect(anchorNavigation(windows, 461)).toEqual({ previous: undefined, next: windows[1] });
    expect(anchorNavigation(windows, 498)).toEqual({ previous: windows[0], next: windows[1] });
    expect(anchorNavigation(windows, 1481).next?.anchor_seconds).toBe(1721);
    expect(anchorNavigation(windows, 1721).previous?.anchor_seconds).toBe(1481);
    expect(anchorNavigation(windows, 7500).next).toBeUndefined();
  });
  it('Live history contains only issued anchors; Review permits future anchors; gaps stay discrete', () => {
    const liveBounds = historyBounds(windows, 0, 498);
    const live = historySamples(windows, ...liveBounds, 'tabpfn_map', 'calibrated');
    expect(live.anchors).toEqual([461]);
    const review = historySamples(windows, ...historyBounds(windows, 0, 900), 'tabpfn_map', 'calibrated');
    expect(review.anchors).toEqual([461, 521, 581, 641, 701, 761, 821, 881]);
    const gap = historySamples(windows, ...historyBounds(windows, 1400, 1800), 'xgboost', 'raw');
    expect(gap.anchors).toEqual([1421, 1481, 1721, 1781]);
    expect(gap.anchors).not.toContain(1541);
  });
  it('keeps probability axes fixed and score axes separate; points never connect or interpolate', () => {
    const view = { ...historySamples(windows, 0, 4, 'xgboost', 'calibrated'), domain: [0, 900] as const };
    expect(riskRange(view)).toEqual([0, 1]);
    const options = riskOptions(700, () => view);
    expect(options.series[1].width).toBe(0); expect(options.series[1].points?.show).toBe(true);
    expect((options.series[1].paths as uPlot.Series.PathBuilder)({} as uPlot, 1, 0, 4)).toBeNull();
    expect(options.series[1].spanGaps).toBe(false); expect(options.cursor?.show).toBe(false);
    const values = options.axes![1].values as uPlot.Axis.DynamicValues;
    expect(values({} as uPlot, [0, .25, .5, .75, 1], 1, 1, .25)).toEqual(['0%', '25%', '50%', '75%', '100%']);
    const score = { ...historySamples(windows, 0, 4, 'current_map', 'calibrated'), domain: [0, 900] as const };
    expect(score.kind).toBe('score'); expect(riskRange(score)).not.toEqual([0, 1]);
    const scoreValues = riskOptions(700, () => score).axes![1].values as uPlot.Axis.DynamicValues;
    expect(scoreValues({} as uPlot, [-100, -80], 1, 1, 1)).toEqual(['-100', '-80']);
  });
  it('keeps selectors pure and excludes model/calibration runtimes, outcome access and forbidden routes', () => {
    const before = JSON.stringify(fixture.predictions);
    for (let time = 0; time < 7500; time += 17) { anchorAt(windows, time); anchorNavigation(windows, time); }
    expect(JSON.stringify(fixture.predictions)).toBe(before);
    for (const folder of ['src/predictions', 'src/components', 'src/charts']) {
      for (const name of readdirSync(folder)) {
        const source = readFileSync(`${folder}/${name}`, 'utf8');
        expect(source).not.toMatch(/ground_truth|matched_episode|\.label\b|predict_proba|Platt|LogisticRegression/);
      }
    }
    expect(Object.keys(JSON.parse(readFileSync('package.json','utf8')).dependencies).sort()).toEqual(['@paper-design/shaders', '@paper-design/shaders-react', 'argentui', 'react', 'react-dom', 'uplot']);
  });
});
