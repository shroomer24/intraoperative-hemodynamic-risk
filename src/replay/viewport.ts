import type { CaseSignals, DisplayChannelId } from '../data/signalValidation';
import type { ReplayMode } from '../data/types';
import type { SignalViewport } from '../charts/synchronizedAxes';

export function visibleWindow(current: number, duration: number): readonly [number, number] {
  const end = Math.min(duration, Math.max(0, current));
  return end <= 900 ? [0, Math.min(900, duration)] : [end - 900, end];
}
export function exactSample(signals: CaseSignals, id: DisplayChannelId, current: number): number | null {
  // No forward search, hold, interpolation, or freshness reprocessing.
  return signals.channels[id].values[Math.floor(current)] ?? null;
}
export function replayViewport(signals: CaseSignals, current: number, mode: ReplayMode,
  domain: readonly [number, number]): SignalViewport {
  // Include the left boundary's exported predecessor for the stepped path's clipping.
  // Live's right boundary contains no samples beyond the current integer second.
  const first = Math.floor(domain[0]);
  const last = Math.floor(mode === 'live' ? Math.min(current, domain[1]) : domain[1]);
  return { domain, ...sliceSignals(signals, first, last) };
}
export function sliceSignals(signals: CaseSignals, first: number, last: number): Omit<SignalViewport, 'domain'> {
  const count = last + 1;
  return {
    timeSeconds: signals.timeSeconds.slice(first, count),
    values: {
      map: signals.channels.map.values.slice(first, count),
      hr: signals.channels.hr.values.slice(first, count),
      spo2: signals.channels.spo2.values.slice(first, count),
      etco2: signals.channels.etco2.values.slice(first, count),
    },
  };
}
