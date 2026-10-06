import { useState } from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { CompareModelsDrawer } from '../src/components/CompareModelsDrawer';
import { ModelSelector } from '../src/components/ModelSelector';
import { RiskPanel } from '../src/components/RiskPanel';
import { alignedComparison, comparisonOrder } from '../src/predictions/alignedComparison';
import * as aligned from '../src/predictions/alignedComparison';
import { anchorAt } from '../src/predictions/anchorLookup';
import { forecastValue } from '../src/predictions/representations';
import type { ProbabilityRepresentation } from '../src/predictions/representations';
import type { ModelId } from '../src/data/types';
import type { PredictionLoadState } from '../src/data/casePredictions';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
import { predictionFixture } from './predictions.fixture';
import { metadata } from './metadata.fixture';

const fixture = predictionFixture();
function setup(seconds = 1577, representation: ProbabilityRepresentation = 'calibrated', load?: PredictionLoadState) {
  const scheduler = new ManualScheduler();
  const store = new ReplayStore('case-004', fixture.selectedCase.durationSeconds, scheduler); store.activate(); store.seek(seconds);
  const predictions: PredictionLoadState = load ?? { status: 'ready', ordinal: 'case-004', data: fixture.predictions };
  const onModelChange = vi.fn();
  const props = { store, predictions, modelId: 'tabpfn_full' as ModelId, representation, onModelChange };
  const result = render(<CompareModelsDrawer {...props} />);
  fireEvent.click(screen.getByRole('button', { name: /Compare models/ }));
  return { ...result, props, store, scheduler, onModelChange, region: screen.getByRole('region', { name: 'Model comparison' }) };
}
function modelRows(region: HTMLElement) { return [...region.querySelectorAll<HTMLButtonElement>('.comparison-model-row')]; }

describe('exact aligned comparison', () => {
  it('renders all seven models in frozen order with one exact issue/horizon', () => {
    const { region } = setup(); const rows = modelRows(region);
    expect(rows.map(row => row.dataset.model)).toEqual(comparisonOrder);
    expect(rows.every(row => row.dataset.anchorSeconds === '1481' && row.dataset.queryPosition === '501' && row.dataset.horizonEnd === '1781')).toBe(true);
    expect(region.textContent).toContain('00:24:41'); expect(region.textContent).toContain('00:29:41');
  });
  it.each(['calibrated', 'raw'] as const)('retains exact %s values and immediate geometry for every model', representation => {
    const { region } = setup(1577, representation); const anchor = anchorAt(fixture.predictions.windows, 1577)!;
    for (const row of modelRows(region)) {
      const value = forecastValue(anchor, row.dataset.model as ModelId, representation)!;
      expect(Number(row.dataset.value)).toBe(value.value); expect(row.dataset.representation).toBe(value.representation);
      const bar = row.querySelector<HTMLElement>('.comparison-probability-fill');
      if (value.kind === 'score') expect(bar).toBeNull(); else expect(bar!.style.width).toBe(`${value.value * 100}%`);
    }
  });
  it('never seeks a nearby output when an exact aligned model is missing', () => {
    const data = structuredClone(fixture.predictions); const anchor = anchorAt(data.windows, 1577)!;
    anchor.model_outputs.tabpfn_map = null;
    for (const window of data.windows) if (window !== anchor) Object.defineProperty(window, 'model_outputs', { get() { throw Error('Neighbor outputs must not be read'); } });
    const { region } = setup(1577, 'calibrated', { status: 'ready', ordinal: 'case-004', data });
    const row = region.querySelector<HTMLElement>('[data-model="tabpfn_map"]')!;
    expect(row.dataset.value).toBeUndefined(); expect(row.textContent).toContain('Not available at aligned anchor'); expect(row.querySelector('.comparison-probability-fill')).toBeNull();
    expect(modelRows(region)).toHaveLength(7);
  });
  it('does not derive a missing calibrated value from raw, or raw from calibrated', () => {
    const anchor = structuredClone(anchorAt(fixture.predictions.windows, 1577)!);
    anchor.model_outputs.logistic_map!.calibrated_probability = null;
    expect(alignedComparison(anchor, 'calibrated')[1].value).toBeUndefined(); expect(alignedComparison(anchor, 'raw')[1].value).toBeTruthy();
    anchor.model_outputs.logistic_map!.raw_probability = null;
    expect(alignedComparison(anchor, 'raw')[1].value).toBeUndefined();
  });
  it.each([0, 460.999])('before first anchor at %s has no future fallback, rows, bars or invented zeros', seconds => {
    const { region } = setup(seconds); expect(region.textContent).toContain('No aligned retained forecast at this replay time.');
    expect(modelRows(region)).toHaveLength(0); expect(region.querySelector('.comparison-probability-fill')).toBeNull(); expect(region.textContent).not.toContain('0.0%');
  });
  it('includes the exact first anchor at its boundary and does not interpolate between anchors', () => {
    const { region, store } = setup(461); const first = anchorAt(fixture.predictions.windows, 461)!;
    expect(modelRows(region).every(row => Number(row.dataset.anchorSeconds) === first.anchor_seconds)).toBe(true);
    const before = modelRows(region).map(row => row.dataset.value); act(() => store.seek(500.5));
    expect(modelRows(region).map(row => row.dataset.value)).toEqual(before);
  });
  it('Current MAP is an untransformed score, with no percent or probability geometry', () => {
    const { region } = setup(); const row = region.querySelector<HTMLElement>('[data-model="current_map"]')!;
    expect(row.dataset.value).toBe('-80'); expect(row.textContent).toContain('Ranking score · mmHg'); expect(row.textContent).toContain('Untransformed −latest MAP');
    expect(row.textContent).not.toMatch(/%|probability|calibrated/i); expect(row.querySelector('.comparison-probability-track')).toBeNull();
  });
  it('prevalence stays clearly separated and raw under either requested representation', () => {
    const { region } = setup(); const group = within(region).getByRole('region', { name: 'Baseline / technical' });
    const row = group.querySelector<HTMLElement>('[data-model="prevalence"]')!;
    expect(row.dataset.representation).toBe('raw_probability'); expect(row.textContent).toContain('Fixed TRAINING baseline');
  });
  it('marks the selected model without ranking, winner, correctness or outcome fields', () => {
    const { region } = setup(); const pressed = modelRows(region).filter(row => row.getAttribute('aria-pressed') === 'true');
    expect(pressed).toHaveLength(1); expect(pressed[0].dataset.model).toBe('tabpfn_full');
    expect(region.innerHTML).not.toMatch(/winner|best|correct|ground_truth|matched_episode|true positive|false positive/i);
  });
  it('real expired Case 019 keeps anchor 774 and exact values at 1200', () => {
    const f = predictionFixture('case-019'); const scheduler = new ManualScheduler(); const store = new ReplayStore('case-019', f.selectedCase.durationSeconds, scheduler); store.activate(); store.seek(1074);
    const props = { store, predictions: { status: 'ready' as const, ordinal: f.selectedCase.ordinal, data: f.predictions }, modelId: 'tabpfn_full' as const, representation: 'calibrated' as const, onModelChange: vi.fn() };
    const { container } = render(<CompareModelsDrawer {...props} />); fireEvent.click(screen.getByRole('button', { name: /Compare models/ }));
    const before = modelRows(container).map(row => row.dataset.value); act(() => store.seek(1200));
    expect(modelRows(container).map(row => row.dataset.value)).toEqual(before); expect(container.textContent).toContain('Forecast horizon ended'); expect(container.textContent).toContain('2m 06s ago');
  });
  it('Live and Review comparisons are identical and never consult truth', () => {
    const { region, store } = setup(); const before = modelRows(region).map(row => row.dataset.value); act(() => store.setMode('review'));
    expect(modelRows(region).map(row => row.dataset.value)).toEqual(before); expect(region.innerHTML).not.toMatch(/ground_truth|positive|negative|event-onset|historical-outcome/i);
  });
  it.each(['loading', 'unavailable'] as const)('fails closed while forecasts are %s', status => {
    const { region } = setup(1577, 'calibrated', { status, ordinal: 'case-004' }); expect(modelRows(region)).toHaveLength(0);
  });
  it('fails closed for stale old-case predictions', () => {
    const f = predictionFixture('case-019'); const { region } = setup(1577, 'calibrated', { status: 'ready', ordinal: 'case-019', data: f.predictions }); expect(modelRows(region)).toHaveLength(0);
  });
  it('selecting any row calls only model change, preserving the full replay snapshot', () => {
    const { region, store, scheduler, onModelChange } = setup(); act(() => { store.setMode('review'); store.setSpeed(1); store.play(); });
    const before = store.getSnapshot();
    for (const row of modelRows(region)) { fireEvent.click(row); expect(onModelChange).toHaveBeenLastCalledWith(row.dataset.model); expect(store.getSnapshot()).toBe(before); }
    expect(scheduler.pending.size).toBe(1); act(() => store.dispose());
  });
  it('row selection updates the primary selector/readout only and preserves representation and expansion', () => {
    const scheduler = new ManualScheduler(); const store = new ReplayStore('case-004', fixture.selectedCase.durationSeconds, scheduler); store.activate(); store.seek(1577); store.setMode('review');
    function Harness() { const [modelId, select] = useState<ModelId>('logistic_map'); const predictions = { status: 'ready' as const, ordinal: fixture.selectedCase.ordinal, data: fixture.predictions }; return <><ModelSelector models={metadata.models} value={modelId} onChange={select} /><RiskPanel model={metadata.models.find(m => m.id === modelId)!} store={store} predictions={predictions} representation="raw" onRepresentationChange={() => {}} /><CompareModelsDrawer modelId={modelId} onModelChange={select} store={store} predictions={predictions} representation="raw" /></>; }
    const { container } = render(<Harness />); fireEvent.click(screen.getByRole('button', { name: /Compare models/ })); const before = store.getSnapshot();
    fireEvent.click(container.querySelector('[data-model="tabpfn_full"].comparison-model-row')!);
    expect((screen.getByRole('combobox', { name: 'Model' }) as HTMLButtonElement).value).toBe('tabpfn_full'); expect(store.getSnapshot()).toBe(before);
    expect(container.querySelector<HTMLElement>('.risk-panel')!.dataset.representation).toBe('raw_probability'); expect(screen.getByRole('button', { name: /Compare models/ }).getAttribute('aria-expanded')).toBe('true');
  });
  it('rows are semantic keyboard buttons with accessible values; Escape returns focus to the trigger', () => {
    const { region } = setup(); for (const row of modelRows(region)) { expect(row.tagName).toBe('BUTTON'); expect(row.disabled).toBe(false); expect(row.getAttribute('aria-label')).toContain('Select '); }
    fireEvent.keyDown(region, { key: 'Escape' }); const toggle = screen.getByRole('button', { name: /Compare models/ });
    expect(toggle.getAttribute('aria-expanded')).toBe('false'); expect(document.activeElement).toBe(toggle); expect(screen.queryByRole('region', { name: 'Model comparison' })).toBeNull();
  });
  it('memoizes rows across clock publications and has no paused clock work', () => {
    const resolve = vi.spyOn(aligned, 'alignedComparison'); const { store, scheduler } = setup(470);
    const calls = resolve.mock.calls.length; act(() => { store.setSpeed(1); store.play(); });
    for (let i = 0; i < 625; i++) act(() => scheduler.frame(16)); expect(resolve).toHaveBeenCalledTimes(calls);
    act(() => store.pause()); const publications = store.diagnostics.publications; scheduler.frame(10000); expect(store.diagnostics.publications).toBe(publications); expect(scheduler.pending.size).toBe(0);
  });
  it('preserves prior scientific and interaction implementations and reduced-motion policy', () => {
    // Phase 8 adds presentation-only idle scheduling to PaperSurface; its lifecycle is tested in release-motion and metal suites.
    // All other prior source hashes, including every scientific selector and shader configuration, remain exact.
    const hashes = JSON.parse(readFileSync('tests/fixtures/phase7-source-hashes.json', 'utf8'));
    for (const [name, digest] of Object.entries(hashes)) if (name.startsWith('src/') && !['src/App.tsx', 'src/main.tsx', 'src/components/CompareModelsDrawer.tsx', 'src/metal/PaperSurface.tsx'].includes(name)) expect(createHash('sha256').update(readFileSync(name)).digest('hex')).toBe(digest);
    expect(readFileSync('src/styles/shell.css', 'utf8')).toContain('transition: none !important'); expect(readFileSync('src/styles/comparison.css', 'utf8')).not.toMatch(/animation:|transition:|min-width:.*fill/);
    expect(readFileSync('src/styles/comparison.css', 'utf8')).toContain('.compare-panel .metal-tab-decoration { filter: grayscale(1); }');
    for (const name of ['src/predictions/alignedComparison.ts', 'src/components/CompareModelsDrawer.tsx']) expect(readFileSync(name, 'utf8')).not.toMatch(/fetch\(|requestAnimationFrame|setInterval|predict_proba|\.fit\(|ground_truth|resolveOutcome|\.sort\(/);
  });
});
