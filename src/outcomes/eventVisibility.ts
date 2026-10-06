import type { CaseEvents, HistoricalEvent } from '../data/eventValidation';
import type { EventLoadState } from '../data/caseEvents';
import type { OutcomeLoadState } from '../data/forecastOutcomes';
import type { PredictionLoadState } from '../data/casePredictions';
import type { ReplayMode } from '../data/types';
import type { ReplayState } from '../replay/store';
import { verifiedCasePredictions } from '../predictions/anchorLookup';

export function historicalContext(events: EventLoadState, outcomes: OutcomeLoadState,
  predictions: PredictionLoadState, replay: ReplayState) {
  const forecasts = verifiedCasePredictions(predictions, replay);
  if (!forecasts || events.status !== 'ready' || outcomes.status !== 'ready'
    || events.ordinal !== replay.caseOrdinal || events.data.ordinal !== replay.caseOrdinal
    || outcomes.ordinal !== replay.caseOrdinal || outcomes.data.ordinal !== replay.caseOrdinal) return undefined;
  return { events: events.data, outcomes: outcomes.data, forecasts };
}
export function visibleEvents(data: CaseEvents, current: number, mode: ReplayMode): readonly HistoricalEvent[] {
  return mode === 'review' ? data.events : data.events.filter(event => current >= event.confirmation_seconds);
}
export interface EventBand { event: HistoricalEvent; left: number; width: number; onsetVisible: boolean; confirmationVisible: boolean }
export function eventBands(events: readonly HistoricalEvent[], domain: readonly [number, number]): EventBand[] {
  const span = domain[1] - domain[0];
  return events.filter(event => event.confirmation_seconds >= domain[0] && event.onset_seconds <= domain[1]).map(event => ({
    event, left: Math.max(0, (event.onset_seconds - domain[0]) / span * 100),
    width: (Math.min(domain[1], event.confirmation_seconds) - Math.max(domain[0], event.onset_seconds)) / span * 100,
    onsetVisible: event.onset_seconds >= domain[0], confirmationVisible: event.confirmation_seconds <= domain[1],
  }));
}
