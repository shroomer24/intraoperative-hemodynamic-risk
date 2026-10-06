import uPlot from 'uplot';
import { formatDuration } from '../data/types';
import { timeSplits } from './synchronizedAxes';
import type { RiskView } from '../predictions/history';

export const RISK_CHART_HEIGHT = 128;
export function riskRange(view: RiskView): [number, number] {
  if (view.kind === 'probability') return [0, 1];
  let min = Infinity; let max = -Infinity;
  for (const value of view.values) if (value !== null) { min = Math.min(min, value); max = Math.max(max, value); }
  if (!Number.isFinite(min)) return [0, 1];
  const pad = Math.max(1, (max - min) * 0.1);
  return [min - pad, max + pad];
}
export function riskOptions(width: number, getView: () => RiskView): uPlot.Options {
  return {
    width, height: RISK_CHART_HEIGHT, padding: [8, 32, 0, 0],
    legend: { show: false },
    cursor: { show: false, drag: { x: false, y: false, setScale: false }, bind: {
      mousedown: () => null, mouseup: () => null, mousemove: () => null,
      mouseenter: () => null, mouseleave: () => null, click: () => null, dblclick: () => null,
    } },
    select: { show: false, left: 0, top: 0, width: 0, height: 0 },
    scales: { x: { time: false, auto: false, range: () => [...getView().domain] },
      y: { auto: false, range: () => riskRange(getView()) } },
    axes: [
      { size: 24, gap: 5, font: '10px ui-monospace, SFMono-Regular, Menlo, monospace', stroke: '#aab0b9',
        splits: plot => timeSplits(getView().domain, plot.bbox.width / uPlot.pxRatio),
        values: (_plot, splits) => splits.map(formatDuration),
        grid: { show: true, stroke: '#1e2229', width: 1 }, ticks: { show: false }, border: { show: false } },
      { size: 44, gap: 6, font: '10px ui-monospace, SFMono-Regular, Menlo, monospace', stroke: '#aab0b9',
        splits: () => {
          const view = getView();
          if (view.kind === 'probability') return [0, .25, .5, .75, 1];
          const range = riskRange(view);
          return view.values.some(value => value !== null)
            ? [Math.ceil(range[0]), Math.round((range[0] + range[1]) / 2), Math.floor(range[1])] : [];
        },
        values: (_plot, splits) => splits.map(value => getView().kind === 'probability' ? `${Math.round(value * 100)}%` : `${value}`),
        grid: { show: true, stroke: '#1e2229', width: 1 }, ticks: { show: false }, border: { show: false } },
    ],
    series: [ {}, { label: 'Retained forecast', stroke: '#aab0b9', width: 0, fill: undefined,
      spanGaps: false, paths: () => null,
      points: { show: true, size: 4, width: 1, stroke: '#aab0b9', fill: '#aab0b9' } } ],
  };
}
