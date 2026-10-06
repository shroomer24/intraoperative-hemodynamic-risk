import { usePointerScrub } from '../replay/usePointerScrub';
import { memo, useLayoutEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { DisplayChannelId } from '../data/signalValidation';
import type { SignalViewport } from './synchronizedAxes';
import type { ReplayStore } from '../replay/store';
import { CHART_HEIGHT, observationRange, signalOptions } from './signalOptions';

let instanceSequence = 0;
export const UPlotAdapter = memo(function UPlotAdapter({ id, viewport, store }: {
  id: DisplayChannelId; viewport: SignalViewport; store?: ReplayStore;
}) {
  const host = useRef<HTMLDivElement>(null);
  const hit = useRef<HTMLDivElement>(null);
  const scrub = usePointerScrub(store, element => {
    const rect = element.getBoundingClientRect(); return {left:rect.left,width:rect.width,domain:view.current.domain};
  }, true);
  const cursor = useRef<HTMLDivElement>(null);
  const future = useRef<HTMLDivElement>(null);
  const view = useRef(viewport);
  const update = useRef<() => void>(() => {});
  useLayoutEffect(() => {
    const element = host.current!;
    let plot: uPlot | undefined;
    let lastWidth = 0;
    let lastFirst: number | undefined;
    let lastLast: number | undefined;
    let lastValues: SignalViewport['values'] | undefined;
    let lastDomain: readonly [number, number] | undefined;
    let lastRange: readonly [number, number] | undefined;
    let updates = 0;
    let scales = 0;
    let current = store?.getSnapshot().currentSeconds ?? 0;
    const position = (seconds: number) => {
      current = seconds;
      if (!plot || !cursor.current || !future.current) return;
      const ratio = uPlot.pxRatio;
      const left = plot.bbox.left / ratio;
      const width = plot.bbox.width / ratio;
      const domain = view.current.domain;
      const x = left + width * Math.min(1, Math.max(0, (seconds - domain[0]) / (domain[1] - domain[0] || 1)));
      const top = plot.bbox.top / ratio;
      const height = plot.bbox.height / ratio;
      if (hit.current) Object.assign(hit.current.style, {left:`${left}px`,top:`${top}px`,width:`${width}px`,height:`${height}px`});
      Object.assign(cursor.current.style, { left: `${x}px`, top: `${top}px`, height: `${height}px` });
      cursor.current.hidden = !store;
      Object.assign(future.current.style, { left: `${x}px`, top: `${top}px`, height: `${height}px`, width: `${Math.max(0, left + width - x)}px` });
      future.current.hidden = store?.getSnapshot().replayMode !== 'review' || seconds >= domain[1];
      element.dataset.cursorSeconds = String(seconds);
    };
    const apply = () => {
      if (!plot) return;
      const next = view.current;
      const first = next.timeSeconds[0];
      const last = next.timeSeconds.at(-1);
      // Keep only the bounded window. No per-frame data allocation or retained history.
      const dataChanged = first !== lastFirst || last !== lastLast || !store && next.values !== lastValues;
      let changed = dataChanged;
      if (dataChanged) {
        plot.setData([next.timeSeconds, next.values[id]], false);
        updates++;
        lastFirst = first; lastLast = last; lastValues = next.values;
      }
      const range = dataChanged || !lastRange ? observationRange(next.values[id], id === 'map') : lastRange;
      if (!lastRange || range[0] !== lastRange[0] || range[1] !== lastRange[1]) {
        plot.setScale('y', { min: range[0], max: range[1] }); lastRange = range; changed = true;
      }
      if (!lastDomain || next.domain[0] !== lastDomain[0] || next.domain[1] !== lastDomain[1]) {
        plot.setScale('x', { min: next.domain[0], max: next.domain[1] }); lastDomain = next.domain; scales++; changed = true;
      }
      // uPlot must recompute visible indices when a slice grows, even with a fixed domain.
      // Redraw recalculates paths/axes; the dynamic range retains the exact shared domain.
      if (changed) plot.redraw(true, true);
      element.dataset.dataUpdates = String(updates);
      element.dataset.scaleUpdates = String(scales);
      position(current);
    };
    update.current = apply;
    const unsubscribe = store?.subscribeFrame(position);
    const resize = new ResizeObserver(entries => {
      const width = Math.floor(entries[0].contentRect.width);
      if (width <= 0 || width === lastWidth) return;
      lastWidth = width;
      if (plot) { plot.setSize({ width, height: CHART_HEIGHT }); position(current); }
      else {
        const initial = view.current;
        const options = signalOptions(id, width, initial.domain, initial.values[id],
          () => ({ domain: view.current.domain, values: view.current.values[id] }));
        options.hooks = { ...options.hooks, draw: [() => position(current)] };
        plot = new uPlot(options, [initial.timeSeconds, initial.values[id]], element);
        plot.root.setAttribute('aria-hidden', 'true');
        element.dataset.plotInstance = String(++instanceSequence);
        apply();
      }
    });
    resize.observe(element);
    return () => { update.current = () => {}; unsubscribe?.(); resize.disconnect(); plot?.destroy(); };
  }, [id, store]);
  useLayoutEffect(() => { view.current = viewport; update.current(); }, [viewport]);
  return <div ref={host} className="signal-chart" data-channel={id}
    data-domain-start={viewport.domain[0]} data-domain-end={viewport.domain[1]}
    data-rendered-first={viewport.timeSeconds[0]} data-rendered-last={viewport.timeSeconds.at(-1)}>
    <div ref={future} className="historical-future-region" hidden aria-hidden="true" />
    <div ref={cursor} className="replay-cursor" hidden aria-hidden="true" />
    {store && <div ref={hit} className="chart-scrub-hit" aria-hidden="true" title="Click or drag to seek surgical time" {...scrub} />}
  </div>;
});
