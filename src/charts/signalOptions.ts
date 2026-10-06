import uPlot from 'uplot';
import { formatDuration } from '../data/types';
import type { DisplayChannelId, Observations } from '../data/signalValidation';
import { timeSplits } from './synchronizedAxes';

export const MAP_EVENT_REFERENCE = 65;
export const MAP_REFERENCE_LABEL = 'MAP 65 · event-definition threshold';
export const CHART_HEIGHT = 88;

export function observationRange(values: Observations, map: boolean): [number, number] {
  let min = map ? MAP_EVENT_REFERENCE : Infinity;
  let max = map ? MAP_EVENT_REFERENCE : -Infinity;
  for (const value of values) if (value !== null) { min = Math.min(min, value); max = Math.max(max, value); }
  if (!Number.isFinite(min)) return [0, 1]; // empty-axis extent, never an observation
  const pad = Math.max(1, (max - min) * 0.12);
  return [min - pad, max + pad];
}

export function drawMapReference(plot: uPlot) {
  const { ctx, bbox } = plot;
  const y = plot.valToPos(MAP_EVENT_REFERENCE, 'y', true);
  if (!Number.isFinite(y) || y < bbox.top || y > bbox.top + bbox.height) return;
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = '#80616a';
  ctx.lineWidth = uPlot.pxRatio;
  ctx.setLineDash([4 * uPlot.pxRatio, 6 * uPlot.pxRatio]);
  ctx.moveTo(bbox.left, y);
  ctx.lineTo(bbox.left + bbox.width, y);
  ctx.stroke();
  ctx.restore();
}

export function signalOptions(id: DisplayChannelId, width: number, domain: readonly [number, number], values: Observations, getView?: () => { domain: readonly [number, number]; values: Observations }): uPlot.Options {
  const map = id === 'map';
  const view = () => getView?.() ?? { domain, values };
  const range = () => observationRange(view().values, map);
  return {
    width, height: CHART_HEIGHT,
    padding: [8, 32, 0, 0],
    legend: { show: false },
    cursor: { show: false, drag: { x: false, y: false, setScale: false }, bind: {
      mousedown: () => null, mouseup: () => null, mousemove: () => null,
      mouseenter: () => null, mouseleave: () => null, click: () => null, dblclick: () => null,
    } },
    select: { show: false, left: 0, top: 0, width: 0, height: 0 },
    scales: { x: { time: false, auto: false, range: () => [view().domain[0], view().domain[1]] }, y: { range } },
    axes: [
      { size: 22, gap: 5, font: '11px ui-monospace, SFMono-Regular, Menlo, monospace', stroke: '#aab0b9',
        splits: plot => timeSplits(view().domain, plot.bbox.width / uPlot.pxRatio),
        values: (_plot, splits) => splits.map(value => id === 'etco2' ? formatDuration(value) : ''),
        grid: { show: true, stroke: '#1e2229', width: 1 }, ticks: { show: false }, border: { show: false } },
      { size: 36, gap: 6, font: '11px ui-monospace, SFMono-Regular, Menlo, monospace', stroke: '#aab0b9',
        splits: () => {
          const yRange = range();
          return view().values.some(value => value !== null)
            ? [Math.ceil(yRange[0]), Math.round((yRange[0] + yRange[1]) / 2), Math.floor(yRange[1])] : [];
        },
        space: 28, values: (_plot, splits) => splits.map(value => `${Math.round(value * 10) / 10}`),
        grid: { show: false }, ticks: { show: false }, border: { show: false } },
    ],
    series: [ {}, { label: id, stroke: map ? '#d8dce3' : '#a4abb6', width: map ? 1.5 : 1.25,
      fill: undefined, points: { show: plot => plot.data[0].length === 1, size: 3 }, spanGaps: false,
      // Zero-order display of the exported grid, not linear/spline interpolation.
      // Clip every null interval; never extend the final sample or bridge a gap.
      paths: uPlot.paths.stepped!({ align: 1, alignGaps: 1, ascDesc: false, extend: false }) } ],
    hooks: map ? { drawAxes: [drawMapReference] } : {},
  };
}
