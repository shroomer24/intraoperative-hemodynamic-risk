import type { CaseEvents, HistoricalEvent } from '../data/eventValidation';
import type { CaseForecastOutcomes } from '../data/forecastOutcomes';
import type { PredictionWindow } from '../data/predictionValidation';
import type { ReplayMode } from '../data/types';

export type ResolvedOutcome = { status: 'not-selected' | 'pending' | 'unavailable' }
  | { status: 'positive'; event: HistoricalEvent; timeAfterForecast: number; resolutionSeconds: number }
  | { status: 'negative'; resolutionSeconds: number };
export function resolveOutcome(forecast: PredictionWindow | undefined, outcomes: CaseForecastOutcomes,
  events: CaseEvents, current: number, mode: ReplayMode): ResolvedOutcome {
  if (!forecast) return { status: 'not-selected' };
  if (outcomes.ordinal !== events.ordinal) return { status: 'unavailable' };
  const matches = outcomes.windows.filter(row => row.query_position === forecast.query_position);
  if (matches.length !== 1) return { status: 'unavailable' };
  const truth = matches[0];
  if (truth.anchor_seconds !== forecast.anchor_seconds || truth.horizon_end_inclusive_seconds !== forecast.horizon_end_inclusive_seconds) return { status: 'unavailable' };
  if (truth.label === 0) {
    return mode === 'review' || current >= truth.future_observation_end_seconds
      ? { status: 'negative', resolutionSeconds: truth.future_observation_end_seconds } : { status: 'pending' };
  }
  const eventMatches = events.events.filter(event => event.onset_seconds === truth.matched_episode_onset_seconds);
  if (eventMatches.length !== 1 || !eventMatches[0].evaluable) return { status: 'unavailable' };
  const event = eventMatches[0];
  if (mode === 'live' && current < event.confirmation_seconds) return { status: 'pending' };
  return { status: 'positive', event, timeAfterForecast: event.onset_seconds - forecast.anchor_seconds,
    resolutionSeconds: event.confirmation_seconds };
}
