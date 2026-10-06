import { chooseOption } from './select.fixture';
import { Profiler } from 'react';
import { ReplayStore } from '../src/replay/store';
import { PlaybackControls } from '../src/components/PlaybackControls';
import { ManualScheduler } from './replay.fixture';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type uPlot from 'uplot';
import { SignalPanel } from '../src/components/SignalPanel';
import { UPlotAdapter } from '../src/charts/UPlotAdapter';
import { MAP_REFERENCE_LABEL, drawMapReference, signalOptions } from '../src/charts/signalOptions';
import { initialViewport } from '../src/charts/synchronizedAxes';
import type { CaseSignals } from '../src/data/signalValidation';
import { signalFixture } from './signals.fixture';
import App from '../src/App';
import { metadata } from './metadata.fixture';
import { predictionFixture } from './predictions.fixture';

const mocks = vi.hoisted(() => ({
  constructors: [] as { options: uPlot.Options; data: uPlot.AlignedData; root: HTMLElement; setSize: ReturnType<typeof vi.fn>; setData: ReturnType<typeof vi.fn>; setScale: ReturnType<typeof vi.fn>; redraw: ReturnType<typeof vi.fn>; destroy: ReturnType<typeof vi.fn> }[],
  stepped: vi.fn(() => vi.fn()),
}));
vi.mock('uplot', () => ({ default: class PlotMock {
  static pxRatio = 1;
  static paths = { stepped: mocks.stepped };
  root = document.createElement('div');
  bbox = { left: 36, top: 8, width: 600, height: 58 };
  setData = vi.fn();
  setScale = vi.fn();
  redraw = vi.fn();
  setSize = vi.fn();
  destroy = vi.fn(() => this.root.remove());
  constructor(options: uPlot.Options, data: uPlot.AlignedData, target: HTMLElement) {
    this.root.append(document.createElement('canvas')); target.append(this.root);
    mocks.constructors.push({ options, data, root: this.root, setSize: this.setSize, setData: this.setData, setScale: this.setScale, redraw: this.redraw, destroy: this.destroy });
  }
} }));

class ResizeMock {
  static instances: ResizeMock[] = [];
  target?: Element;
  disconnected = false;
  constructor(readonly callback: ResizeObserverCallback) { ResizeMock.instances.push(this); }
  observe(target: Element) { this.target = target; }
  disconnect() { this.disconnected = true; }
  deliver(width: number) { this.callback([{ contentRect: { width } } as ResizeObserverEntry], this as unknown as ResizeObserver); }
}
beforeEach(() => {
  mocks.constructors.length = 0; mocks.stepped.mockClear(); ResizeMock.instances.length = 0;
  vi.stubGlobal('ResizeObserver', ResizeMock);
});
afterEach(() => vi.unstubAllGlobals());
const gap = signalFixture();
const short = signalFixture('case-019');
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}

describe('operation transitions and accessibility', () => {
  it('publishes all four channels at surgical start with exact readouts', async () => {
    const { container } = render(<SignalPanel selectedCase={gap.selectedCase} loader={async () => gap.signals} />);
    await screen.findByText('Verified monitor traces');
    expect([...container.querySelectorAll('.signal-chart')].map(node => node.getAttribute('data-channel'))).toEqual(['map', 'hr', 'spo2', 'etco2']);
    expect(screen.getByLabelText(/^MAP at 00:00:00:/).textContent).toBe(String(gap.signals.channels.map.values[0]));
    expect(screen.getByText(MAP_REFERENCE_LABEL)).toBeTruthy();
    expect(container.textContent).toContain('1 of 1 displayed exported observations available');
    expect(container.textContent).toContain('Time since surgical start');
    expect(container.textContent).not.toMatch(/subject_id|case_id|calendar|alarm|treatment/);
  });
  it('clears traces immediately when changing cases and rejects an obsolete response', async () => {
    const old = deferred<CaseSignals>(); const current = deferred<CaseSignals>();
    const loader = vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
    const { container, rerender } = render(<SignalPanel selectedCase={gap.selectedCase} loader={loader} />);
    const firstSignal = loader.mock.calls[0][1] as AbortSignal;
    rerender(<SignalPanel selectedCase={short.selectedCase} loader={loader} />);
    expect(firstSignal.aborted).toBe(true);
    expect(container.querySelectorAll('.signal-chart')).toHaveLength(0);
    await act(async () => current.resolve(short.signals));
    expect(container.querySelectorAll('.signal-chart')).toHaveLength(4);
    await act(async () => old.resolve(gap.signals));
    expect(container.textContent).not.toContain('59 missing samples');
    expect(container.textContent).toContain('1 of 1 displayed exported observations available');
  });
  it('removes loaded old traces while the next operation is loading or unavailable', async () => {
    const next = deferred<CaseSignals>();
    const loader = vi.fn().mockResolvedValueOnce(gap.signals).mockReturnValueOnce(next.promise);
    const { container, rerender } = render(<SignalPanel selectedCase={gap.selectedCase} loader={loader} />);
    await screen.findByText('Verified monitor traces');
    rerender(<SignalPanel selectedCase={short.selectedCase} loader={loader} />);
    expect(container.querySelectorAll('.signal-chart')).toHaveLength(0);
    expect(screen.getByText('Loading monitor traces…')).toBeTruthy();
    await act(async () => next.reject(new Error('unsafe failure detail')));
    expect(screen.getByText('Monitor traces unavailable')).toBeTruthy();
    expect(container.textContent).not.toContain('unsafe failure detail');
    expect(container.querySelectorAll('.signal-chart')).toHaveLength(0);
  });
  it('shares one fixed domain and resets all plots on operation change', async () => {
    const loader = vi.fn().mockResolvedValueOnce(gap.signals).mockResolvedValueOnce(short.signals);
    const { container, rerender } = render(<SignalPanel selectedCase={gap.selectedCase} loader={loader} />);
    await screen.findByText('Verified monitor traces');
    expect([...container.querySelectorAll('.signal-chart')].map(node => [node.getAttribute('data-domain-start'), node.getAttribute('data-domain-end')])).toEqual(Array(4).fill(['0', '900']));
    rerender(<SignalPanel selectedCase={short.selectedCase} loader={loader} />);
    await screen.findByText('Verified monitor traces');
    expect([...container.querySelectorAll('.signal-chart')].map(node => node.getAttribute('data-domain-end'))).toEqual(Array(4).fill('900'));
  });
  it('keeps prediction panels empty while enabling playback for verified signals', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, text: async () => signalFixture('case-001').text })));
    render(<App loadData={async () => metadata} />);
    await screen.findByText('Verified monitor traces');
    expect(screen.getByLabelText('No probability selected').textContent).toBe('—');
    for (const name of [/Previous anchor/, /Next anchor/]) expect((screen.getByRole('button', { name }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: 'Play' }) as HTMLButtonElement).disabled).toBe(false);
    chooseOption(screen.getByRole('combobox', { name: 'Model' }), 'current_map');
    expect(screen.getByLabelText('No score selected').textContent).toBe('—');
    expect((screen.getByRole('slider') as HTMLInputElement).disabled).toBe(false);
  });
});

describe('owned uPlot lifecycle and options', () => {
  it('uses untouched numeric arrays and imperatively resizes without recreating plots', () => {
    const viewport = initialViewport(gap.signals);
    const { unmount } = render(<UPlotAdapter id="map" viewport={viewport} />);
    const observer = ResizeMock.instances[0];
    act(() => observer.deliver(720));
    expect(mocks.constructors).toHaveLength(1);
    const plot = mocks.constructors[0];
    expect(plot.data[0]).toBe(viewport.timeSeconds);
    expect(plot.data[1]).toBe(viewport.values.map);
    expect(plot.data[1][81]).toBeNull();
    act(() => { observer.deliver(600); observer.deliver(600); });
    expect(plot.setSize).toHaveBeenCalledExactlyOnceWith({ width: 600, height: 88 });
    expect(mocks.constructors).toHaveLength(1);
    unmount();
    expect(observer.disconnected).toBe(true);
    expect(plot.destroy).toHaveBeenCalledTimes(1);
  });
  it('updates data without destroying canvases when the viewport changes', () => {
    const { rerender, unmount } = render(<UPlotAdapter id="map" viewport={initialViewport(gap.signals)} />);
    act(() => ResizeMock.instances[0].deliver(720));
    rerender(<UPlotAdapter id="map" viewport={initialViewport(short.signals)} />);
    expect(mocks.constructors[0].destroy).not.toHaveBeenCalled();
    expect(mocks.constructors[0].setData).toHaveBeenCalledWith([initialViewport(short.signals).timeSeconds, initialViewport(short.signals).values.map], false);
    expect(mocks.constructors).toHaveLength(1);
    unmount();
    expect(mocks.constructors[0].destroy).toHaveBeenCalledTimes(1);
  });
  it('disables zoom, cursor, selection, fill and gap spanning; uses stepped paths only', () => {
    const options = signalOptions('map', 720, [0, 900], initialViewport(gap.signals).values.map);
    expect(options.cursor?.show).toBe(false);
    expect(options.cursor?.drag).toEqual({ x: false, y: false, setScale: false });
    expect(options.select?.show).toBe(false);
    expect(options.scales?.x?.time).toBe(false);
    expect(options.series[1].spanGaps).toBe(false);
    expect(options.series[1].fill).toBeUndefined();
    expect(mocks.stepped).toHaveBeenCalledWith({ align: 1, alignGaps: 1, ascDesc: false, extend: false });
    const scaleRange = options.scales?.x?.range as uPlot.Range.Function;
    expect(scaleRange({} as uPlot, 100, 200, 'x')).toEqual([0, 900]);
  });
  it('draws a MAP-only reference at 65 in data coordinates', () => {
    const ctx = { save: vi.fn(), restore: vi.fn(), beginPath: vi.fn(), setLineDash: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn() };
    const plot = { ctx, bbox: { left: 36, top: 8, width: 600, height: 58 }, valToPos: vi.fn(() => 40) } as unknown as uPlot;
    drawMapReference(plot);
    expect(plot.valToPos).toHaveBeenCalledExactlyOnceWith(65, 'y', true);
    expect(ctx.moveTo).toHaveBeenCalledWith(36, 40);
    expect(ctx.lineTo).toHaveBeenCalledWith(636, 40);
    expect(signalOptions('hr', 720, [0, 900], [null]).hooks).toEqual({});
  });
  it('formats only the bottom shared time axis as HH:MM:SS', async () => {
    const options = signalOptions('etco2', 720, [0, 900], [null]);
    const values = options.axes![0].values as uPlot.Axis.DynamicValues;
    expect(values({} as uPlot, [0, 300, 600, 900], 0, 100, 300)).toEqual(['00:00:00', '00:05:00', '00:10:00', '00:15:00']);
    await waitFor(() => expect(options.scales?.x?.auto).toBe(false));
  });
});


describe('Phase 3 synchronized replay integration', () => {
  it('shares cursor/domain, hides future accessibly, preserves null readouts and shades Review', async () => {
    const scheduler = new ManualScheduler();
    const store = new ReplayStore('case-004', 7500, scheduler);
    const { container } = render(<><SignalPanel selectedCase={gap.selectedCase} loader={async () => gap.signals} store={store} /><PlaybackControls store={store} /></>);
    await screen.findByText('Verified monitor traces');
    act(() => ResizeMock.instances.forEach(observer => observer.deliver(720)));
    act(() => store.seek(81.9));
    expect(screen.getByLabelText('MAP at 00:01:21: unavailable').textContent).toBe('—');
    expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('81.9');
    expect(container.textContent).toContain('81 of 82 displayed exported observations available');
    expect(container.textContent).not.toContain('59 missing samples');
    const plots = [...container.querySelectorAll<HTMLElement>('.signal-chart')];
    expect(plots.map(plot => plot.dataset.cursorSeconds)).toEqual(Array(4).fill('81.9'));
    expect(plots.map(plot => plot.dataset.renderedLast)).toEqual(Array(4).fill('81'));
    act(() => store.setMode('review'));
    expect(plots.map(plot => plot.dataset.renderedLast)).toEqual(Array(4).fill('900'));
    expect([...container.querySelectorAll<HTMLElement>('.historical-future-region')].every(region => !region.hidden)).toBe(true);
    expect(screen.getByText('Historical future · not model input')).toBeTruthy();
    expect(screen.getByLabelText('MAP at 00:01:21: unavailable').textContent).toBe('—');
    expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('81.9');
    act(() => { store.setMode('live'); store.seek(14400); });
    expect(plots.map(plot => [plot.dataset.domainStart, plot.dataset.domainEnd])).toEqual(Array(4).fill(['6600', '7500']));
    expect([...container.querySelectorAll<HTMLElement>('.historical-future-region')].every(region => region.hidden)).toBe(true);
    for (const name of [/Previous anchor/, /Next anchor/]) {
      const button = screen.getByRole('button', { name }) as HTMLButtonElement;
      expect(button.disabled).toBe(true); expect(button.title).toBe('Available when frozen forecasts are loaded');
    }
  });
  it('bounds React commits, keeps four instances during ticks/scrubs, and changes x scale only on domain movement', async () => {
    const scheduler = new ManualScheduler(); const store = new ReplayStore('case-004', 7500, scheduler);
    let commits = 0;
    const loader = async () => gap.signals;
    const { container, unmount } = render(<Profiler id="replay" onRender={() => commits++}><SignalPanel selectedCase={gap.selectedCase} loader={loader} store={store} /><PlaybackControls store={store} /></Profiler>);
    await screen.findByText('Verified monitor traces');
    act(() => ResizeMock.instances.forEach(observer => observer.deliver(720)));
    expect(mocks.constructors).toHaveLength(4);
    act(() => store.play()); const before = commits;
    for (let i = 0; i < 625; i++) act(() => scheduler.frame(16));
    expect(commits - before).toBeLessThanOrEqual(100);
    expect(mocks.constructors).toHaveLength(4);
    for (const plot of mocks.constructors) {
      expect(plot.destroy).not.toHaveBeenCalled();
      expect(plot.setScale.mock.calls.filter(call => call[0] === 'x')).toHaveLength(1);
    }
    const slider = screen.getByRole('slider') as HTMLInputElement;
    fireEvent.change(slider, { target: { value: '2345' } });
    expect(store.getSnapshot()).toMatchObject({ currentSeconds: 2345, isPlaying: false, visibleWindow: [1445, 2345] });
    expect(scheduler.pending.size).toBe(0); expect(mocks.constructors).toHaveLength(4);
    expect([...container.querySelectorAll<HTMLElement>('.signal-chart')].map(plot => plot.dataset.cursorSeconds)).toEqual(Array(4).fill('2345'));
    const idleCommits = commits; scheduler.frame(10000); expect(commits).toBe(idleCommits);
    fireEvent.click(screen.getByRole('button', { name: 'Start' })); expect(store.getSnapshot().currentSeconds).toBe(0);
    fireEvent.click(screen.getByRole('button', { name: 'End' })); expect(store.getSnapshot().currentSeconds).toBe(7500);
    unmount(); expect(mocks.constructors.every(plot => plot.destroy.mock.calls.length === 1)).toBe(true);
  });
  it('does not redraw charts between sample boundaries when the opening domain is fixed', async () => {
    const scheduler = new ManualScheduler(); const store = new ReplayStore('case-004', 7500, scheduler);
    render(<SignalPanel selectedCase={gap.selectedCase} loader={async () => gap.signals} store={store} />);
    await screen.findByText('Verified monitor traces');
    act(() => ResizeMock.instances.forEach(observer => observer.deliver(720)));
    const initialRedraws = mocks.constructors.map(plot => plot.redraw.mock.calls.length);
    act(() => { store.setSpeed(1); store.play(); });
    for (let i = 0; i < 9; i++) act(() => scheduler.frame(100));
    expect(mocks.constructors.map(plot => plot.redraw.mock.calls.length)).toEqual(initialRedraws);
    act(() => scheduler.frame(100));
    expect(mocks.constructors.every(plot => plot.redraw.mock.calls.length === 2)).toBe(true);
    act(() => store.dispose());
  });
  it('scrubs and switches speed/mode without fetching forbidden assets; selecting a case while playing resets time', async () => {
    const fetchMock = vi.fn(async (path: string) => ({ ok: true, text: async () => path.endsWith('prediction-windows.json') ? predictionFixture(path.includes('case-019') ? 'case-019' : 'case-001').text : signalFixture(path.includes('case-019') ? 'case-019' : 'case-001').text }));
    vi.stubGlobal('fetch', fetchMock);
    render(<App loadData={async () => metadata} />); await screen.findByText('Verified monitor traces');
    fireEvent.change(screen.getByRole('slider'), { target: { value: '2000' } });
    chooseOption(screen.getByRole('combobox', { name: 'Playback speed' }), '60');
    fireEvent.click(screen.getByRole('radio', { name: 'Review' }));
    expect(fetchMock).toHaveBeenCalledTimes(3);
    fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    expect(screen.getByRole('button', { name: 'Pause' })).toBeTruthy();
    chooseOption(screen.getByRole('combobox', { name: 'Held-out operation' }), 'case-019');
    await screen.findByText('Verified monitor traces');
    expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('0');
    expect((screen.getByRole('combobox', { name: 'Playback speed' }) as HTMLButtonElement).value).toBe('30');
    expect((screen.getByRole('radio', { name: 'Live Replay' }) as HTMLInputElement).checked).toBe(true);
    expect(screen.queryByRole('button', { name: 'Pause' })).toBeNull();
    expect(fetchMock.mock.calls.map(call => call[0])).toEqual(['/replay-v01/cases/case-001/signals.json', '/replay-v01/cases/case-001/prediction-windows.json', '/replay-v01/cases/case-001/events.json', '/replay-v01/cases/case-019/signals.json', '/replay-v01/cases/case-019/prediction-windows.json', '/replay-v01/cases/case-019/events.json']);
    fireEvent.keyDown(screen.getByRole('slider'), { key: 'End' });
    expect((screen.getByRole('slider') as HTMLInputElement).value).toBe('2100'); // Phase 6.1 explicitly implements accessible range End seeking.
  });
});
