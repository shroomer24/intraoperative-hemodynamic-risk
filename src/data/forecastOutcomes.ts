import { useEffect, useState } from 'react';
import { fetchReplayText, sha256 } from './assetIntegrity';
import { predictionAssetPath } from './casePredictions';
import type { PredictionLoadState } from './casePredictions';
import { validatePredictions } from './predictionValidation';
import type { CasePredictions } from './predictionValidation';
import type { CaseOrdinal, PresentationCase } from './types';

export interface FrozenForecastOutcome {
  query_position: number;
  anchor_seconds: number;
  horizon_end_inclusive_seconds: number;
  future_observation_end_seconds: number;
  label: 0 | 1;
  matched_episode_onset_seconds: number | null;
}
export interface CaseForecastOutcomes { ordinal: CaseOrdinal; windows: readonly FrozenForecastOutcome[] }
function check(condition: unknown): asserts condition {
  if (!condition) throw new Error('Historical outcomes unavailable.');
}
export function validateForecastOutcomes(raw: unknown, forecasts: CasePredictions,
  selectedCase: PresentationCase): CaseForecastOutcomes {
  check(Array.isArray(raw) && raw.length === forecasts.windows.length && forecasts.ordinal === selectedCase.ordinal);
  const windows = raw.map((value, index): FrozenForecastOutcome => {
    check(value !== null && typeof value === 'object' && !Array.isArray(value));
    const row = value as Record<string, unknown>; const forecast = forecasts.windows[index];
    check(row.query_position === forecast.query_position && row.anchor_seconds === forecast.anchor_seconds
      && row.horizon_end_inclusive_seconds === forecast.horizon_end_inclusive_seconds);
    const observationEnd = row.future_observation_end_seconds;
    check(observationEnd === forecast.anchor_seconds + 360 && observationEnd <= selectedCase.durationSeconds);
    const truth = row.ground_truth;
    check(truth !== null && typeof truth === 'object' && !Array.isArray(truth));
    check(Object.keys(truth).length === 2 && Object.keys(truth).every(key => ['label', 'matched_episode_onset_seconds'].includes(key)));
    const outcome = truth as Record<string, unknown>; const label = outcome.label;
    const onset = outcome.matched_episode_onset_seconds;
    check(label === 0 || label === 1);
    if (label === 0) check(onset === null);
    else check(typeof onset === 'number' && Number.isInteger(onset)
      && onset > forecast.anchor_seconds && onset <= forecast.horizon_end_inclusive_seconds);
    return { query_position: forecast.query_position, anchor_seconds: forecast.anchor_seconds,
      horizon_end_inclusive_seconds: forecast.horizon_end_inclusive_seconds,
      future_observation_end_seconds: observationEnd as number, label,
      matched_episode_onset_seconds: onset as number | null };
  });
  return { ordinal: selectedCase.ordinal, windows };
}
export interface ForecastBundle { ordinal: CaseOrdinal; forecasts: CasePredictions; outcomes?: CaseForecastOutcomes }
export async function loadCaseForecasts(selectedCase: PresentationCase, signal: AbortSignal): Promise<ForecastBundle> {
  const path = predictionAssetPath(selectedCase.ordinal);
  if (selectedCase.predictionAsset.path !== path || !/^[a-f0-9]{64}$/.test(selectedCase.predictionAsset.sha256)) throw new Error('Frozen forecasts unavailable.');
  const text = await fetchReplayText(`/replay-v01/${path}`, signal);
  if (await sha256(text) !== selectedCase.predictionAsset.sha256) throw new Error('Frozen forecasts unavailable.');
  signal.throwIfAborted(); const raw = JSON.parse(text);
  const forecasts = validatePredictions(raw, selectedCase);
  let outcomes: CaseForecastOutcomes | undefined;
  try { outcomes = validateForecastOutcomes(raw, forecasts, selectedCase); }
  catch { /* Invalid historical truth cannot contaminate or erase verified model forecasts. */ }
  return { ordinal: selectedCase.ordinal, forecasts, outcomes };
}
export type ForecastLoadState = { ordinal?: CaseOrdinal; status: 'loading' | 'unavailable' }
  | { ordinal: CaseOrdinal; status: 'ready'; data: ForecastBundle };
export type OutcomeLoadState = { ordinal?: CaseOrdinal; status: 'loading' | 'unavailable' }
  | { ordinal: CaseOrdinal; status: 'ready'; data: CaseForecastOutcomes };
export type ForecastLoader = typeof loadCaseForecasts;
export function useCaseForecasts(selectedCase?: PresentationCase, loader: ForecastLoader = loadCaseForecasts): ForecastLoadState {
  const [state, setState] = useState<ForecastLoadState>({ status: 'loading' });
  useEffect(() => {
    if (!selectedCase) return;
    const controller = new AbortController(); let current = true;
    setState({ ordinal: selectedCase.ordinal, status: 'loading' });
    loader(selectedCase, controller.signal).then(data => {
      if (current && !controller.signal.aborted && data.ordinal === selectedCase.ordinal
        && data.forecasts.ordinal === selectedCase.ordinal && (!data.outcomes || data.outcomes.ordinal === selectedCase.ordinal)) {
        setState({ ordinal: selectedCase.ordinal, status: 'ready', data });
      }
    }).catch(() => {
      if (current && !controller.signal.aborted) setState({ ordinal: selectedCase.ordinal, status: 'unavailable' });
    });
    return () => { current = false; controller.abort(); };
  }, [selectedCase, loader]);
  return state.ordinal === selectedCase?.ordinal ? state : { ordinal: selectedCase?.ordinal, status: 'loading' };
}
export function forecastState(state: ForecastLoadState): PredictionLoadState {
  return state.status === 'ready' ? { status: 'ready', ordinal: state.ordinal, data: state.data.forecasts } : state;
}
export function outcomeState(state: ForecastLoadState): OutcomeLoadState {
  if (state.status !== 'ready') return state;
  return state.data.outcomes ? { status: 'ready', ordinal: state.ordinal, data: state.data.outcomes }
    : { status: 'unavailable', ordinal: state.ordinal };
}
