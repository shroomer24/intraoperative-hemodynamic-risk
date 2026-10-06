import type { PredictionWindow } from '../data/predictionValidation';
import type { ModelId } from '../data/types';
import { forecastValue } from './representations';
import type { ProbabilityRepresentation } from './representations';

export const comparisonOrder: readonly ModelId[] = [
  'current_map', 'logistic_map', 'logistic_full', 'xgboost', 'tabpfn_map', 'tabpfn_full', 'prevalence',
];

// One validated window is the only lookup source. Missing outputs never search another window.
export function alignedComparison(anchor: PredictionWindow, representation: ProbabilityRepresentation) {
  return comparisonOrder.map(id => ({ id, value: forecastValue(anchor, id, representation) }));
}
