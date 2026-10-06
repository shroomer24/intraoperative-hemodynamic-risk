import type { CaseSignals, DisplayChannelId, Observations } from '../data/signalValidation';

export interface SignalViewport {
  domain: readonly [number, number];
  timeSeconds: number[];
  values: Record<DisplayChannelId, Observations>;
}
export function initialViewport(signals: CaseSignals): SignalViewport {
  const end = Math.min(900, signals.timeSeconds[signals.timeSeconds.length - 1]);
  // An exact contiguous slice of exported seconds; no decimation or resampling.
  const count = end + 1;
  return {
    domain: [0, end],
    timeSeconds: signals.timeSeconds.slice(0, count),
    values: {
      map: signals.channels.map.values.slice(0, count),
      hr: signals.channels.hr.values.slice(0, count),
      spo2: signals.channels.spo2.values.slice(0, count),
      etco2: signals.channels.etco2.values.slice(0, count),
    },
  };
}
export function timeSplits(domain: readonly [number, number], plotWidth: number): number[] {
  const segments = plotWidth >= 560 ? 5 : plotWidth >= 280 ? 3 : 1;
  return Array.from({ length: segments + 1 }, (_, i) => Math.round(domain[0] + (domain[1] - domain[0]) * i / segments));
}
export function availability(values: Observations) {
  let available = 0;
  let gaps = 0;
  let wasMissing = false;
  for (const value of values) {
    if (value !== null) available++;
    else if (!wasMissing) gaps++;
    wasMissing = value === null;
  }
  return { available, missing: values.length - available, total: values.length, gaps };
}
