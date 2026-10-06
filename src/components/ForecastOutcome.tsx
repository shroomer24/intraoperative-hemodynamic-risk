import type { ResolvedOutcome } from '../outcomes/outcomeResolution';
import type { ReplayMode } from '../data/types';
import { formatDuration } from '../data/types';

export function ForecastOutcome({ outcome, mode }: { outcome: ResolvedOutcome; mode: ReplayMode }) {
  return <section className={`forecast-outcome outcome-${outcome.status}`} aria-label="Selected forecast historical outcome"
    data-outcome-state={outcome.status}
    data-event-ordinal={outcome.status === 'positive' ? outcome.event.presentation_event_ordinal : undefined}
    data-onset-seconds={outcome.status === 'positive' ? outcome.event.onset_seconds : undefined}
    data-confirmation-seconds={outcome.status === 'positive' ? outcome.event.confirmation_seconds : undefined}
    data-resolution-seconds={outcome.status === 'positive' || outcome.status === 'negative' ? outcome.resolutionSeconds : undefined}>
    <h3>Historical outcome</h3>
    {outcome.status === 'not-selected' && <p>No retained forecast selected.</p>}
    {outcome.status === 'unavailable' && <p>Historical outcome unavailable · verified event linkage required.</p>}
    {outcome.status === 'pending' && <p>Outcome pending</p>}
    {outcome.status === 'negative' && <p>No sustained hypotension onset occurred within this forecast horizon</p>}
    {outcome.status === 'positive' && <>
      <p className="historical-positive">Sustained hypotension occurred within forecast horizon</p>
      <dl><div><dt>Event onset</dt><dd>{formatDuration(outcome.event.onset_seconds)}</dd></div>
        <div><dt>Confirmed sustained</dt><dd>{formatDuration(outcome.event.confirmation_seconds)}</dd></div>
        <div><dt>Time after forecast</dt><dd>+{formatDuration(outcome.timeAfterForecast).slice(3)}</dd></div></dl>
    </>}
    <p className="outcome-note">{mode === 'review' ? 'Historical outcome · hindsight, not model input.'
      : outcome.status === 'positive' ? 'Onset annotated retrospectively after sustained confirmation.'
      : 'Historical results appear only after required observation.'}</p>
  </section>;
}
