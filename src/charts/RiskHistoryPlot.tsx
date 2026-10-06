import { memo, useLayoutEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { ReplayStore } from '../replay/store';
import type { RiskView } from '../predictions/history';
import { RISK_CHART_HEIGHT, riskOptions, riskRange } from './riskOptions';

let instanceSequence = 0;
export const RiskHistoryPlot = memo(function RiskHistoryPlot({ view, store }: { view: RiskView; store: ReplayStore }) {
  const host = useRef<HTMLDivElement>(null);
  const cursor = useRef<HTMLDivElement>(null);
  const future = useRef<HTMLDivElement>(null);
  const latest = useRef(view);
  const update = useRef<() => void>(() => {});
  useLayoutEffect(() => {
    const element = host.current!;
    let plot: uPlot | undefined;
    let lastWidth = 0;
    let lastAnchors: number[] | undefined;
    let lastValues: RiskView['values'] | undefined;
    let lastDomain: RiskView['domain'] | undefined;
    let lastRange: readonly [number, number] | undefined;
    let lastKind: RiskView['kind'] | undefined;
    let current = store.getSnapshot().currentSeconds;
    let dataUpdates = 0;
    let scaleUpdates = 0;
    const position = (seconds: number) => {
      current = seconds;
      if (!plot || !cursor.current || !future.current) return;
      const { domain } = latest.current;
      const left = plot.bbox.left / uPlot.pxRatio;
      const width = plot.bbox.width / uPlot.pxRatio;
      const x = left + width * Math.min(1, Math.max(0, (seconds - domain[0]) / (domain[1] - domain[0] || 1)));
      const top = plot.bbox.top / uPlot.pxRatio;
      const height = plot.bbox.height / uPlot.pxRatio;
      Object.assign(cursor.current.style, { left: `${x}px`, top: `${top}px`, height: `${height}px` });
      Object.assign(future.current.style, { left: `${x}px`, top: `${top}px`, height: `${height}px`, width: `${Math.max(0, left + width - x)}px` });
      future.current.hidden = store.getSnapshot().replayMode !== 'review' || seconds >= domain[1];
      element.dataset.cursorSeconds = String(seconds);
    };
    const apply = () => {
      if (!plot) return;
      const next = latest.current;
      let changed = false;
      if (next.anchors !== lastAnchors || next.values !== lastValues) {
        plot.setData([next.anchors, next.values], false);
        lastAnchors = next.anchors; lastValues = next.values; dataUpdates++; changed = true;
      }
      const range = riskRange(next);
      if (!lastRange || lastRange[0] !== range[0] || lastRange[1] !== range[1] || lastKind !== next.kind) {
        plot.setScale('y', { min: range[0], max: range[1] }); lastRange = range; changed = true;
      }
      if (!lastDomain || lastDomain[0] !== next.domain[0] || lastDomain[1] !== next.domain[1]) {
        plot.setScale('x', { min: next.domain[0], max: next.domain[1] }); lastDomain = next.domain; scaleUpdates++; changed = true;
      }
      lastKind = next.kind;
      if (changed) plot.redraw(true, true);
      element.dataset.dataUpdates = String(dataUpdates);
      element.dataset.scaleUpdates = String(scaleUpdates);
      element.dataset.yMin = String(range[0]); element.dataset.yMax = String(range[1]);
      position(current);
    };
    update.current = apply;
    const unsubscribe = store.subscribeFrame(position);
    const resize = new ResizeObserver(entries => {
      const width = Math.floor(entries[0].contentRect.width);
      if (width <= 0 || width === lastWidth) return;
      lastWidth = width;
      if (plot) { plot.setSize({ width, height: RISK_CHART_HEIGHT }); position(current); }
      else {
        const options = riskOptions(width, () => latest.current);
        options.hooks = { draw: [() => position(current)] };
        plot = new uPlot(options, [latest.current.anchors, latest.current.values], element);
        plot.root.setAttribute('aria-hidden', 'true');
        element.dataset.riskInstance = String(++instanceSequence);
        apply();
      }
    });
    resize.observe(element);
    return () => { update.current = () => {}; unsubscribe(); resize.disconnect(); plot?.destroy(); };
  }, [store]);
  useLayoutEffect(() => { latest.current = view; update.current(); }, [view]);
  return <div ref={host} className="risk-history-plot" aria-hidden="true" data-axis-kind={view.kind}
    data-domain-start={view.domain[0]} data-domain-end={view.domain[1]}
    data-anchor-count={view.anchors.length} data-rendered-first={view.anchors[0]} data-rendered-last={view.anchors.at(-1)}>
    <div ref={future} className="historical-future-region" hidden />
    <div ref={cursor} className="replay-cursor" />
  </div>;
});
