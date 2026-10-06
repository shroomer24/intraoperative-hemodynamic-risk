import { modelIds } from './types';
import type { CaseOrdinal, ModelId, PresentationCase } from './types';

export interface ProbabilityOutput {
  kind: 'probability';
  raw_probability: number | null;
  calibrated_probability: number | null;
}
export interface CurrentMapOutput {
  kind: 'score';
  raw_score: number | null;
  score_definition: '-map_latest';
  unit: 'mmHg';
}
export type ModelOutput = ProbabilityOutput | CurrentMapOutput;
export type ModelOutputs = { current_map: CurrentMapOutput | null }
  & Record<Exclude<ModelId, 'current_map'>, ProbabilityOutput | null>;
// Forecast-only structure: no outcomes, labels, episode fields or source identities.
export interface PredictionWindow {
  query_position: number;
  anchor_seconds: number;
  horizon_start_exclusive_seconds: number;
  horizon_end_inclusive_seconds: number;
  model_outputs: ModelOutputs;
}
export interface CasePredictions { ordinal: CaseOrdinal; windows: readonly PredictionWindow[] }

function check(condition: unknown): asserts condition {
  if (!condition) throw new Error('Frozen forecasts unavailable.');
}
function record(value: unknown, allowedKeys: readonly string[]): Record<string, unknown> {
  check(value !== null && typeof value === 'object' && !Array.isArray(value));
  // Reject unknown/private fields without reading their values.
  check(Object.keys(value).every(key => allowedKeys.includes(key)));
  return value as Record<string, unknown>;
}
function storedNumber(value: unknown, probability: boolean): number | null {
  if (value === undefined || value === null || value === '') return null;
  check(typeof value === 'number' && Number.isFinite(value));
  check(!probability || value >= 0 && value <= 1);
  return value;
}
const windowKeys = ['query_position', 'anchor_seconds', 'eligible', 'history_start_seconds',
  'history_end_seconds', 'horizon_start_exclusive_seconds', 'horizon_end_inclusive_seconds',
  'future_observation_end_seconds', 'model_outputs', 'ground_truth'] as const;

export function validatePredictions(input: unknown, selectedCase: PresentationCase): CasePredictions {
  check(Array.isArray(input) && input.length <= 3146);
  const positions = new Set<number>();
  let previous = -Infinity;
  const windows = input.map((item): PredictionWindow => {
    const row = record(item, windowKeys);
    const query = row.query_position;
    const anchor = row.anchor_seconds;
    check(typeof query === 'number' && Number.isInteger(query) && query >= 0 && query < 3146 && !positions.has(query));
    positions.add(query);
    check(typeof anchor === 'number' && Number.isInteger(anchor) && anchor >= 300 && anchor > previous);
    previous = anchor;
    check(row.eligible === true && row.history_start_seconds === anchor - 300 && row.history_end_seconds === anchor);
    check(row.horizon_start_exclusive_seconds === anchor && row.horizon_end_inclusive_seconds === anchor + 300);
    check(row.future_observation_end_seconds === anchor + 360 && anchor + 360 <= selectedCase.durationSeconds);
    const rawModels = record(row.model_outputs, modelIds);
    const outputs = {} as ModelOutputs;
    for (const id of modelIds) {
      const value = rawModels[id];
      if (value === undefined || value === null || value === '') { outputs[id] = null; continue; }
      if (id === 'current_map') {
        const model = record(value, ['raw_score', 'score_definition', 'unit']);
        check(model.score_definition === '-map_latest' && model.unit === 'mmHg');
        outputs.current_map = { kind: 'score', raw_score: storedNumber(model.raw_score, false),
          score_definition: '-map_latest', unit: 'mmHg' };
      } else {
        const model = record(value, id === 'prevalence' ? ['raw_probability'] : ['raw_probability', 'calibrated_probability']);
        outputs[id] = { kind: 'probability', raw_probability: storedNumber(model.raw_probability, true),
          calibrated_probability: id === 'prevalence' ? null : storedNumber(model.calibrated_probability, true) };
      }
    }
    // Deliberately never access row.ground_truth (including its type, label or onset).
    return { query_position: query, anchor_seconds: anchor,
      horizon_start_exclusive_seconds: anchor, horizon_end_inclusive_seconds: row.horizon_end_inclusive_seconds as number,
      model_outputs: outputs };
  });
  return { ordinal: selectedCase.ordinal, windows };
}
