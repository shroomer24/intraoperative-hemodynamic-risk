import { Profiler } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useCasePredictions } from '../src/data/casePredictions';
import type { PredictionLoader, PredictionLoadState } from '../src/data/casePredictions';
import type { CasePredictions } from '../src/data/predictionValidation';
import type { PresentationCase } from '../src/data/types';
import { RiskPanel } from '../src/components/RiskPanel';
import { RiskHistory } from '../src/components/RiskHistory';
import { PlaybackControls } from '../src/components/PlaybackControls';
import { ReplayStore } from '../src/replay/store';
import { predictionFixture } from './predictions.fixture';
import { metadata } from './metadata.fixture';
import { ManualScheduler } from './replay.fixture';

const mocks = vi.hoisted(() => ({ plots: [] as { setData: ReturnType<typeof vi.fn>; setScale: ReturnType<typeof vi.fn>; destroy: ReturnType<typeof vi.fn> }[] }));
vi.mock('uplot', () => ({ default: class {
  static pxRatio = 1;
  root = document.createElement('div');
  bbox = { left: 36, top: 8, width: 600, height: 58 };
  setData = vi.fn(); setScale = vi.fn(); setSize = vi.fn(); redraw = vi.fn();
  destroy = vi.fn(() => this.root.remove());
  constructor(_options: unknown, _data: unknown, target: HTMLElement) {
    this.root.append(document.createElement('canvas')); target.append(this.root); mocks.plots.push(this);
  }
} }));
class ResizeMock {
  constructor(private callback: ResizeObserverCallback) {}
  observe() { this.callback([{ contentRect: { width: 720 } } as ResizeObserverEntry], this as unknown as ResizeObserver); }
  disconnect() {}
}
beforeEach(() => { mocks.plots.length = 0; vi.stubGlobal('ResizeObserver', ResizeMock); });
afterEach(() => vi.unstubAllGlobals());
const fixture = predictionFixture();
const ready: PredictionLoadState = { ordinal: 'case-004', status: 'ready', data: fixture.predictions };
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}
function harnessStore() {
  const scheduler = new ManualScheduler();
  const store = new ReplayStore('case-004', fixture.selectedCase.durationSeconds, scheduler);
  store.activate(); return { store, scheduler };
}
function model(id: string) { return metadata.models.find(item => item.id === id)!; }
function LoaderHarness({ selectedCase, loader, store }: { selectedCase: PresentationCase; loader: PredictionLoader; store: ReplayStore }) {
  const predictions = useCasePredictions(selectedCase, loader);
  return <RiskPanel model={model('logistic_map')} store={store} predictions={predictions} representation="calibrated" onRepresentationChange={() => {}} />;
}

describe('forecast case-session isolation', () => {
  it('aborts an old request and discards its late success without exposing its forecast under a new case', async () => {
    const old = deferred<CasePredictions>(); const current = deferred<CasePredictions>();
    const next = predictionFixture('case-019');
    const loader = vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
    const { store } = harnessStore(); store.seek(498);
    const nextStore = new ReplayStore('case-019', next.selectedCase.durationSeconds); nextStore.activate(); nextStore.seek(498);
    const { container, rerender } = render(<LoaderHarness selectedCase={fixture.selectedCase} store={store} loader={loader} />);
    const signal = loader.mock.calls[0][1] as AbortSignal;
    rerender(<LoaderHarness selectedCase={next.selectedCase} store={nextStore} loader={loader} />);
    expect(signal.aborted).toBe(true);
    await act(async () => old.resolve(fixture.predictions));
    expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBeNull();
    await act(async () => current.resolve(next.predictions));
    expect(container.querySelector('.risk-panel')?.getAttribute('data-anchor-seconds')).toBe('474');
    expect(container.innerHTML).not.toMatch(/ground_truth|matched_episode|subject_id|case_id/);
    store.dispose(); nextStore.dispose();
  });
  it('clears an already published old forecast while the new case loads or fails; no new signals/old forecasts pairing', async () => {
    const next = predictionFixture('case-019'); const pending = deferred<CasePredictions>();
    const loader = vi.fn().mockResolvedValueOnce(fixture.predictions).mockReturnValueOnce(pending.promise);
    const { store } = harnessStore(); store.seek(498);
    const { container, rerender } = render(<LoaderHarness selectedCase={fixture.selectedCase} store={store} loader={loader} />);
    await screen.findByText('Verified frozen forecasts');
    expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).not.toBeNull();
    // Even before signals have a new session, render identity clears the old prediction.
    rerender(<LoaderHarness selectedCase={next.selectedCase} store={store} loader={loader} />);
    expect(container.querySelector('.risk-panel')?.getAttribute('data-value')).toBeNull();
    await act(async () => pending.reject(new Error('private transport detail')));
    expect(screen.getByText('Frozen forecasts unavailable')).toBeTruthy();
    expect(container.textContent).not.toContain('private transport detail');
  });
});

describe('retained forecast readout and anchor controls', () => {
  it('keeps original issue/horizon between anchors, preserves binary64, switches model/representation at same playing time', () => {
    const { store, scheduler } = harnessStore();
    const { container, rerender } = render(<RiskPanel model={model('logistic_map')} store={store} predictions={ready} representation="calibrated" onRepresentationChange={() => {}} />);
    expect(screen.getByText('No retained forecast at this time')).toBeTruthy();
    act(() => store.seek(498));
    const panel = () => container.querySelector<HTMLElement>('.risk-panel')!;
    expect(panel().dataset).toMatchObject({ anchorSeconds: '461', horizonStart: '461', horizonEnd: '761', queryPosition: '484', value: '0.0555634924654587' });
    expect(screen.getByText('37 s')).toBeTruthy();
    act(() => { store.setSpeed(1); store.play(); });
    const time = store.getSnapshot().currentSeconds;
    rerender(<RiskPanel model={model('tabpfn_full')} store={store} predictions={ready} representation="calibrated" onRepresentationChange={() => {}} />);
    expect(store.getSnapshot()).toMatchObject({ currentSeconds: time, isPlaying: true, caseOrdinal: 'case-004' });
    expect(panel().dataset).toMatchObject({ anchorSeconds: '461', horizonEnd: '761', value: '0.02370546600170726' });
    rerender(<RiskPanel model={model('tabpfn_full')} store={store} predictions={ready} representation="raw" onRepresentationChange={() => {}} />);
    expect(panel().dataset.value).toBe('0.029835958033800125');
    rerender(<RiskPanel model={model('current_map')} store={store} predictions={ready} representation="raw" onRepresentationChange={() => {}} />);
    expect(panel().dataset.value).toBe('-94');
    expect(panel().textContent).not.toMatch(/%|probability|calibrated/i);
    act(() => scheduler.frame(1000));
    expect(panel().dataset).toMatchObject({ anchorSeconds: '461', horizonEnd: '761' });
    act(() => store.dispose());
  });
  it('pauses and seeks actual retained anchors; skips a gap rather than fabricating cadence', () => {
    const { store, scheduler } = harnessStore();
    render(<PlaybackControls store={store} predictions={ready} />);
    const previous = () => screen.getByRole('button', { name: /Previous anchor/ }) as HTMLButtonElement;
    const next = () => screen.getByRole('button', { name: /Next anchor/ }) as HTMLButtonElement;
    expect(previous().disabled).toBe(true); expect(next().disabled).toBe(false);
    fireEvent.click(next()); expect(store.getSnapshot().currentSeconds).toBe(461);
    expect(previous().disabled).toBe(true);
    act(() => { store.seek(1600); store.play(); });
    fireEvent.click(next()); expect(store.getSnapshot()).toMatchObject({ currentSeconds: 1721, isPlaying: false });
    expect(scheduler.pending.size).toBe(0);
    fireEvent.click(previous()); expect(store.getSnapshot().currentSeconds).toBe(1481);
    act(() => store.seek(1600)); fireEvent.click(previous()); expect(store.getSnapshot().currentSeconds).toBe(1481);
    act(() => store.seek(fixture.predictions.windows.at(-1)!.anchor_seconds)); expect(next().disabled).toBe(true);
    act(() => store.seek(0)); expect(previous().disabled).toBe(true);
  });
});

describe('risk history shares the clock without rebuilds or outcome semantics', () => {
  it('Live hides all future points; Review reveals and distinguishes them on a fixed probability axis; MAP uses score axis', () => {
    const { store } = harnessStore();
    const { container, rerender } = render(<RiskHistory modelId="logistic_map" store={store} predictions={ready} representation="calibrated" />);
    const plot = () => container.querySelector<HTMLElement>('.risk-history-plot')!;
    expect(plot().dataset).toMatchObject({ anchorCount: '0', axisKind: 'probability', yMin: '0', yMax: '1' });
    act(() => store.seek(498)); expect(plot().dataset).toMatchObject({ anchorCount: '1', renderedLast: '461', cursorSeconds: '498' });
    act(() => store.setMode('review')); expect(plot().dataset.anchorCount).toBe('8');
    expect(screen.getByText('Historical future · not available at replay time')).toBeTruthy();
    expect(container.querySelector<HTMLElement>('.historical-future-region')!.hidden).toBe(false);
    const instance = plot().dataset.riskInstance;
    rerender(<RiskHistory modelId="current_map" store={store} predictions={ready} representation="calibrated" />);
    expect(plot().dataset.axisKind).toBe('score'); expect(plot().dataset.riskInstance).toBe(instance);
    expect(screen.getByRole('heading', { name: 'Score history' })).toBeTruthy();
    expect(container.textContent).not.toMatch(/%|probability|ground_truth|matched_episode/i);
    expect(mocks.plots).toHaveLength(1);
  });
  it('bounds commits, leaves data unchanged between anchors, moves imperative cursor, and has no paused animation', () => {
    const { store, scheduler } = harnessStore(); store.seek(470); store.setSpeed(1);
    let commits = 0;
    const { container, unmount } = render(<Profiler id="risk" onRender={() => commits++}><RiskHistory modelId="xgboost" store={store} predictions={ready} representation="calibrated" /></Profiler>);
    const plot = mocks.plots[0]; const dataCalls = plot.setData.mock.calls.length;
    act(() => store.play()); const startCommits = commits;
    for (let i = 0; i < 625; i++) act(() => scheduler.frame(16));
    expect(commits - startCommits).toBeLessThanOrEqual(100);
    expect(plot.setData).toHaveBeenCalledTimes(dataCalls); expect(mocks.plots).toHaveLength(1);
    expect(Number(container.querySelector<HTMLElement>('.risk-history-plot')!.dataset.cursorSeconds)).toBeCloseTo(480);
    act(() => store.pause()); const idleCommits = commits; scheduler.frame(10000);
    expect(commits).toBe(idleCommits); expect(scheduler.pending.size).toBe(0);
    act(() => store.seek(10000)); expect(mocks.plots).toHaveLength(1);
    expect(container.querySelector<HTMLElement>('.risk-history-plot')!.dataset).toMatchObject({ domainStart: '6600', domainEnd: '7500' });
    unmount(); expect(plot.destroy).toHaveBeenCalledTimes(1);
  });
});
