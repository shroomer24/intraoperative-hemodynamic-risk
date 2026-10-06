import { fireEvent, render, screen, cleanup } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CompareModelsDrawer } from '../src/components/CompareModelsDrawer';
import { PaperSurface } from '../src/metal/PaperSurface';
import { surfaceBudget } from '../src/metal/lifecycle';
import { ReplayStore } from '../src/replay/store';
import { ManualScheduler } from './replay.fixture';
import { predictionFixture } from './predictions.fixture';

vi.mock('@paper-design/shaders', async importOriginal => ({ ...(await importOriginal<Record<string, unknown>>()),
  ShaderMount: class {
    canvasElement = document.createElement('canvas');
    setMaxPixelCount = vi.fn();
    dispose = vi.fn(() => this.canvasElement.remove());
    constructor(parent: HTMLElement) { parent.append(this.canvasElement); }
  },
}));
class IntersectionMock {
  constructor(public callback: IntersectionObserverCallback) {}
  observe(target: Element) { this.callback([{ target, isIntersecting: true } as IntersectionObserverEntry], this as unknown as IntersectionObserver); }
  disconnect = vi.fn();
}
class ResizeMock { observe = vi.fn(); disconnect = vi.fn(); }
let reduced = false;
beforeEach(() => {
  reduced = false;
  vi.stubGlobal('WebGL2RenderingContext', class {});
  vi.stubGlobal('IntersectionObserver', IntersectionMock);
  vi.stubGlobal('ResizeObserver', ResizeMock);
  vi.spyOn(window, 'matchMedia').mockImplementation(query => ({ matches: reduced, media: query, addEventListener: vi.fn(), removeEventListener: vi.fn() } as unknown as MediaQueryList));
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => ({ isContextLost: () => false, getExtension: () => ({ loseContext: vi.fn() }) }) as unknown as WebGLRenderingContext);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); expect(surfaceBudget.counts().total).toBe(0); });
function setup() {
  const f = predictionFixture();
  const store = new ReplayStore(f.selectedCase.ordinal, f.selectedCase.durationSeconds, new ManualScheduler());
  store.activate(); store.seek(1577);
  const onModelChange = vi.fn();
  const result = render(<><PaperSurface kind="mercury" /><PaperSurface kind="probability" /><PaperSurface kind="timeline" />
    <CompareModelsDrawer store={store} predictions={{ status: 'ready', ordinal: f.selectedCase.ordinal, data: f.predictions }}
      modelId="tabpfn_full" representation="calibrated" onModelChange={onModelChange} /></>);
  fireEvent.click(screen.getByRole('button', { name: /Compare models/ }));
  return { ...result, onModelChange };
}
describe('comparison shares the existing bounded metal lifecycle', () => {
  it('seven idle rows add no Paper context; every row replaces the single transient owner', () => {
    const { container } = setup(); const rows = [...container.querySelectorAll('.comparison-model-row')];
    expect(rows).toHaveLength(7); expect(container.querySelectorAll('canvas')).toHaveLength(3);
    for (const row of rows) {
      fireEvent.pointerEnter(row.parentElement!);
      expect(surfaceBudget.counts()).toEqual({ persistent: 3, transient: 1, total: 4 });
      expect(container.querySelectorAll('canvas')).toHaveLength(4);
    }
    fireEvent.pointerLeave(rows[6].parentElement!);
    expect(container.querySelectorAll('canvas')).toHaveLength(3);
  });
  it('collapse disposes the row transient without disturbing the three existing surfaces', () => {
    const { container } = setup(); const row = container.querySelector('.comparison-model-row')!;
    fireEvent.focus(row); expect(surfaceBudget.counts().total).toBe(4);
    fireEvent.click(screen.getByRole('button', { name: /Compare models/ }));
    expect(surfaceBudget.counts()).toEqual({ persistent: 3, transient: 0, total: 3 });
  });
  it('reduced motion keeps exact static values, CSS geometry and functional selection with zero shaders', () => {
    reduced = true;
    const { container, onModelChange } = setup(); const row = container.querySelector<HTMLElement>('[data-model="tabpfn_full"].comparison-model-row')!;
    fireEvent.pointerEnter(row.parentElement!); fireEvent.focus(row); fireEvent.click(row);
    expect(row.dataset.value).toBe('0.009513229723575736');
    expect(row.querySelector<HTMLElement>('.comparison-probability-fill')!.style.width).toBe(`${0.009513229723575736 * 100}%`);
    expect(onModelChange).toHaveBeenCalledWith('tabpfn_full'); expect(container.querySelectorAll('canvas')).toHaveLength(0);
    expect(surfaceBudget.counts().total).toBe(0);
  });
});
