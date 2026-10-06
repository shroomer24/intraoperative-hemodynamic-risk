import type { ModelId } from '../data/types';
import type { PredictionWindow } from '../data/predictionValidation';

export type ProbabilityRepresentation = 'calibrated' | 'raw';
export interface ForecastValue {
  kind: 'probability' | 'score';
  value: number;
  representation: 'calibrated_probability' | 'raw_probability' | 'raw_score';
}
export function forecastValue(window: PredictionWindow | undefined, model: ModelId,
  representation: ProbabilityRepresentation = 'calibrated'): ForecastValue | undefined {
  const output = window?.model_outputs[model];
  if (!output) return;
  if (output.kind === 'score') return output.raw_score === null ? undefined
    : { kind: 'score', value: output.raw_score, representation: 'raw_score' };
  const key = model === 'prevalence' || representation === 'raw' ? 'raw_probability' : 'calibrated_probability';
  return output[key] === null ? undefined : { kind: 'probability', value: output[key], representation: key };
}
export function representationLabel(model: ModelId, representation: ProbabilityRepresentation) {
  return model === 'current_map' ? 'Ranking score · mmHg'
    : model === 'prevalence' ? 'Fixed TRAINING probability baseline'
    : representation === 'calibrated' ? 'Calibrated model estimate' : 'Retained raw probability';
}
export function formatForecast(value: ForecastValue | undefined): string {
  if (!value) return '—';
  return value.kind === 'probability' ? `${(value.value * 100).toFixed(1)}%` : `${value.value}`;
}
