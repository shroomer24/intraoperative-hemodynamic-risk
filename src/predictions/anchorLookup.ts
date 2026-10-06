import type { CasePredictions, PredictionWindow } from '../data/predictionValidation';
import type { PredictionLoadState } from '../data/casePredictions';
import type { ReplayState } from '../replay/store';

export function verifiedCasePredictions(load: PredictionLoadState, replay: ReplayState): CasePredictions | undefined {
  return replay.ready && load.status === 'ready' && load.ordinal === replay.caseOrdinal
    && load.data.ordinal === replay.caseOrdinal ? load.data : undefined;
}
// Upper bound: first retained anchor strictly later than the supplied second.
export function upperAnchor(windows: readonly PredictionWindow[], seconds: number): number {
  let low = 0; let high = windows.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (windows[mid].anchor_seconds <= seconds) low = mid + 1;
    else high = mid;
  }
  return low;
}
export function anchorAt(windows: readonly PredictionWindow[], currentSeconds: number): PredictionWindow | undefined {
  return windows[upperAnchor(windows, Math.floor(currentSeconds)) - 1];
}
export function anchorNavigation(windows: readonly PredictionWindow[], currentSeconds: number) {
  const index = upperAnchor(windows, currentSeconds);
  const atExactAnchor = windows[index - 1]?.anchor_seconds === currentSeconds;
  return { previous: windows[index - (atExactAnchor ? 2 : 1)], next: windows[index] };
}
