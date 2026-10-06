import uPlot from 'uplot';
import { afterEach, expect, it, vi } from 'vitest';
import { signalFixture } from './signals.fixture';
import { initialViewport } from '../src/charts/synchronizedAxes';
import { signalOptions } from '../src/charts/signalOptions';

class RecordingPath {
  lines: [number, number][] = [];
  lineTo(x: number, y: number) { this.lines.push([x, y]); }
  moveTo(x: number, y: number) { this.lines.push([x, y]); }
  rect() {}
}
afterEach(() => vi.unstubAllGlobals());
it('the real uPlot stepped renderer masks exported null intervals and generates no diagonal interpolation', () => {
  vi.stubGlobal('Path2D', RecordingPath);
  const view = initialViewport(signalFixture().signals);
  const options = signalOptions('map', 900, view.domain, view.values.map);
  const plot = {
    mode: 1, _data: [view.timeSeconds, view.values.map], bands: [],
    series: [{ scale: 'x' }, { ...options.series[1], scale: 'y', pxRound: (value: number) => value,
      gaps: (_plot: uPlot, _series: number, _start: number, _end: number, gaps: number[][]) => gaps }],
    scales: { x: { ori: 0, dir: 1 }, y: { ori: 1, dir: 1 } },
    bbox: { left: 0, top: 0, width: 900, height: 100 },
    valToPosH: (value: number) => value, valToPosV: (value: number) => value,
  } as unknown as uPlot;
  const paths = options.series[1].paths!(plot, 1, 0, view.timeSeconds.length - 1)!;
  expect(paths.clip).not.toBeNull();
  expect(paths.gaps).toHaveLength(2);
  for (let i = 0; i < view.values.map.length; i++) {
    if (view.values.map[i] === null && view.values.map[i - 1] === null) {
      expect(paths.gaps!.some(([start, end]) => i >= start && i <= end)).toBe(true);
    }
  }
  const lines = (paths.stroke as unknown as RecordingPath).lines;
  expect(lines.length).toBeGreaterThan(1);
  for (let i = 1; i < lines.length; i++) expect(lines[i][0] === lines[i - 1][0] || lines[i][1] === lines[i - 1][1]).toBe(true);
});
