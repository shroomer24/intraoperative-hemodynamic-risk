import type { PredictionWindow } from '../data/predictionValidation';
import type { ModelId } from '../data/types';
import { upperAnchor } from './anchorLookup';
import { forecastValue } from './representations';
import type { ProbabilityRepresentation } from './representations';

export interface RiskSamples { anchors: number[]; values: (number | null)[]; kind: 'probability' | 'score' }
export interface RiskView extends RiskSamples { domain: readonly [number, number] }
export function historyBounds(windows: readonly PredictionWindow[], start: number, end: number): readonly [number, number] {
  const upper = upperAnchor(windows, start);
  const first = upper > 0 && windows[upper - 1].anchor_seconds === start ? upper - 1 : upper;
  return [first, upperAnchor(windows, end)];
}
export function historySamples(windows: readonly PredictionWindow[], first: number, last: number,
  model: ModelId, representation: ProbabilityRepresentation): RiskSamples {
  const selected = windows.slice(first, last);
  return { anchors: selected.map(window => window.anchor_seconds),
    values: selected.map(window => forecastValue(window, model, representation)?.value ?? null),
    kind: model === 'current_map' ? 'score' : 'probability' };
}
