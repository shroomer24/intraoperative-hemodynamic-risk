import { useLayoutEffect, useMemo, useRef } from 'react';
import type { EventLoadState } from '../data/caseEvents';
import type { OutcomeLoadState } from '../data/forecastOutcomes';
import type { PredictionLoadState } from '../data/casePredictions';
import { formatDuration } from '../data/types';
import { eventBands, historicalContext, visibleEvents } from '../outcomes/eventVisibility';
import { useReplay } from '../replay/store';
import type { ReplayStore } from '../replay/store';

export function EventLane({ events, outcomes, predictions, store }: {
  events: EventLoadState; outcomes: OutcomeLoadState; predictions: PredictionLoadState; store: ReplayStore;
}) {
  const replay = useReplay(store);
  const context = historicalContext(events, outcomes, predictions, replay);
  const count = context ? visibleEvents(context.events, replay.currentSeconds, replay.replayMode).length : 0;
  const permitted = useMemo(() => context?.events.events.slice(0, count) ?? [], [context?.events, count]);
  const bands = useMemo(() => eventBands(permitted, replay.visibleWindow), [permitted, replay.visibleWindow]);
  const cursor = useRef<HTMLDivElement>(null); const domain = useRef(replay.visibleWindow);
  useLayoutEffect(() => {
    const move = (seconds: number) => {
      const span = domain.current[1] - domain.current[0];
      if (cursor.current) cursor.current.style.left = `${Math.max(0, Math.min(100, (seconds - domain.current[0]) / span * 100))}%`;
    };
    move(store.getSnapshot().currentSeconds);
    return store.subscribeFrame(move);
  }, [store]);
  useLayoutEffect(() => {
    domain.current = replay.visibleWindow;
    const span = domain.current[1] - domain.current[0];
    if (cursor.current) cursor.current.style.left = `${Math.max(0, Math.min(100, (replay.currentSeconds - domain.current[0]) / span * 100))}%`;
  }, [replay.visibleWindow, replay.currentSeconds]);
  return <section className="event-lane" aria-labelledby="event-lane-heading" data-case-ordinal={context ? replay.caseOrdinal : undefined}
    data-permitted-events={permitted.length} data-domain-start={replay.visibleWindow[0]} data-domain-end={replay.visibleWindow[1]}>
    <div className="event-lane-heading"><h2 id="event-lane-heading">Historical events</h2>
      <span>Onset → sustained confirmation</span></div>
    <p className="event-lane-copy">{replay.replayMode === 'review' ? 'Historical outcome · hindsight, not model input.'
      : 'Episodes appear only after confirmation; onset is then annotated retrospectively.'}</p>
    {context ? <>
      <div className="event-lane-track" aria-hidden="true">
        {bands.map(({ event, left, width, onsetVisible, confirmationVisible }) => <div key={event.presentation_event_ordinal}
          className={`event-band${event.evaluable ? '' : ' event-not-evaluable'}`} style={{ left: `${left}%`, width: `${width}%` }}
          data-event-ordinal={event.presentation_event_ordinal} data-onset-seconds={event.onset_seconds} data-confirmation-seconds={event.confirmation_seconds}>
          {onsetVisible && <span className="event-onset-marker" />}{confirmationVisible && <span className="event-confirmation-marker" />}
        </div>)}
        <div ref={cursor} className="event-lane-cursor" />
      </div>
      {permitted.length > 0 ? <details className="event-details"><summary>Event details · {permitted.length} {replay.replayMode === 'review' ? 'historical' : 'confirmed'}</summary>
        <ol>{permitted.map(event => <li key={event.presentation_event_ordinal} data-event-ordinal={event.presentation_event_ordinal}>
          <b>Sustained hypotension · episode {event.presentation_event_ordinal}</b>
          <span>Onset {formatDuration(event.onset_seconds)} · Confirmed sustained {formatDuration(event.confirmation_seconds)}</span>
          <span>{event.is_recurrent ? 'Recurrent episode' : 'First frozen episode'}{!event.evaluable && ' · Historical annotation only; not evaluable for forecast outcomes'}</span>
        </li>)}</ol></details> : <p className="event-lane-empty">{replay.replayMode === 'review'
          ? 'No frozen sustained hypotension episodes in this operation.' : 'Only confirmed historical episodes appear during Live Replay.'}</p>}
    </> : <p className="event-lane-empty">{events.status === 'unavailable' || outcomes.status === 'unavailable'
      ? 'Historical events unavailable · no unverified outcome displayed.' : 'Verifying historical events…'}</p>}
    <p className="event-definition">Definition: invasive MAP &lt;65 mmHg for ≥60 continuously observable seconds. Band marks onset to confirmation, not episode duration.</p>
  </section>;
}
