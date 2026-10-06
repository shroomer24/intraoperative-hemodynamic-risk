import type { CaseOrdinal, PresentationCase } from './types';

export interface HistoricalEvent {
  presentation_event_ordinal: number;
  onset_seconds: number;
  confirmation_seconds: number;
  evaluable: boolean;
  is_recurrent: boolean;
}
export interface CaseEvents { ordinal: CaseOrdinal; events: readonly HistoricalEvent[] }
function check(condition: unknown): asserts condition {
  if (!condition) throw new Error('Historical events unavailable.');
}
export function validateEvents(input: unknown, selectedCase: PresentationCase): CaseEvents {
  check(Array.isArray(input) && input.length <= 3146);
  let previousOnset = -1;
  const allowed = ['presentation_event_ordinal', 'onset_seconds', 'confirmation_seconds',
    'evaluable', 'is_recurrent', 'minimum_confirmed_low_seconds', 'sustained_duration_seconds'];
  const events = input.map((value, index): HistoricalEvent => {
    check(value !== null && typeof value === 'object' && !Array.isArray(value));
    check(Object.keys(value).length === allowed.length && Object.keys(value).every(key => allowed.includes(key)));
    const row = value as Record<string, unknown>;
    check(row.presentation_event_ordinal === index + 1);
    const onset = row.onset_seconds; const confirmation = row.confirmation_seconds;
    check(typeof onset === 'number' && Number.isInteger(onset) && onset >= 0 && onset > previousOnset);
    check(typeof confirmation === 'number' && Number.isInteger(confirmation) && confirmation <= selectedCase.durationSeconds);
    // Frozen export uses 60 inclusive one-second observations, ending at onset + 59.
    check(row.minimum_confirmed_low_seconds === 60 && confirmation === onset + 59);
    check(row.sustained_duration_seconds === null); // Export supplies no episode duration.
    check(typeof row.evaluable === 'boolean' && typeof row.is_recurrent === 'boolean');
    previousOnset = onset;
    return { presentation_event_ordinal: index + 1, onset_seconds: onset, confirmation_seconds: confirmation,
      evaluable: row.evaluable, is_recurrent: row.is_recurrent };
  });
  return { ordinal: selectedCase.ordinal, events };
}
