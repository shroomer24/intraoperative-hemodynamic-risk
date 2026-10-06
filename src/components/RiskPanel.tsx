import { MoltenProbabilityProgress } from '../metal/MoltenProbabilityProgress';
import { GunmetalSelect } from '../metal/GunmetalSelect';
import type { EventLoadState } from '../data/caseEvents';
import type { OutcomeLoadState } from '../data/forecastOutcomes';
import { historicalContext } from '../outcomes/eventVisibility';
import { resolveOutcome } from '../outcomes/outcomeResolution';
import { forecastAge } from '../outcomes/forecastAge';
import { ForecastOutcome } from './ForecastOutcome';
import type { ModelDefinition } from '../data/types';
import { formatDuration } from '../data/types';
import type { PredictionLoadState } from '../data/casePredictions';
import { useReplay } from '../replay/store';
import type { ReplayStore } from '../replay/store';
import { anchorAt, verifiedCasePredictions } from '../predictions/anchorLookup';
import { forecastValue, formatForecast, representationLabel } from '../predictions/representations';
import type { ProbabilityRepresentation } from '../predictions/representations';

export function RiskPanel({ model, store, predictions, representation, onRepresentationChange, events, outcomes }: {
  model: ModelDefinition; store: ReplayStore; predictions: PredictionLoadState;
  events?: EventLoadState; outcomes?: OutcomeLoadState;
  representation: ProbabilityRepresentation; onRepresentationChange: (value: ProbabilityRepresentation) => void;
}) {
  const replay = useReplay(store);
  const data = verifiedCasePredictions(predictions, replay);
  const anchor = data ? anchorAt(data.windows, replay.currentSeconds) : undefined;
  const timing = forecastAge(anchor, replay.currentSeconds);
  const context = events && outcomes ? historicalContext(events, outcomes, predictions, replay) : undefined;
  const resolved = context ? resolveOutcome(anchor, context.outcomes, context.events, replay.currentSeconds, replay.replayMode)
    : { status: 'unavailable' as const };
  const value = forecastValue(anchor, model.id, representation);
  const score = model.id === 'current_map';
  const baseline = model.id === 'prevalence';
  const status = data ? 'Verified frozen forecasts' : predictions.status === 'unavailable'
    ? 'Frozen forecasts unavailable' : 'Verifying frozen forecasts…';
  return <section className="risk-panel" aria-labelledby="risk-heading" aria-busy={!data && predictions.status !== 'unavailable'}
    data-model={model.id} data-query-position={anchor?.query_position} data-anchor-seconds={anchor?.anchor_seconds}
    data-horizon-start={anchor?.horizon_start_exclusive_seconds} data-horizon-end={anchor?.horizon_end_inclusive_seconds}
    data-representation={value?.representation} data-value={value?.value}>
    <div className="panel-heading"><h2 id="risk-heading">{score ? 'MAP-based ranking score' : 'Prediction'}</h2><span className="panel-status" role="status">{status}</span></div>
    <div className="risk-body">
      <span className="forecast-source-label">{score ? 'Model score' : 'Model forecast'}</span>
      <div className="prediction-target">
        <h3>{score ? 'Current MAP' : 'New sustained hypotension'}</h3>
        <p>{score ? 'Untransformed −latest MAP' : 'within original 5-minute forecast horizon'}</p>
      </div>
      <div className="risk-readout" aria-label={value ? score ? 'Retained MAP ranking score' : 'Retained forecast probability'
        : score ? 'No score selected' : 'No probability selected'}>{formatForecast(value)}</div>
      {!score && !baseline && <MoltenProbabilityProgress key={`${replay.caseOrdinal}:${model.id}`} probability={value?.value}
        representation={representation === 'raw' ? 'raw_probability' : 'calibrated_probability'} />}
      {!value && <p className="forecast-unavailable">{anchor ? 'Selected representation unavailable at this anchor'
        : score ? 'No retained score at this time' : 'No retained forecast at this time'}</p>}
      <span className="representation-label">{representationLabel(model.id, representation)}</span>
      <div className="risk-divider" />
      <dl className="risk-context">
        <div><dt>Representation</dt><dd>{score ? 'Score' : baseline || representation === 'raw' ? 'Raw probability' : 'Calibrated probability'}</dd></div>
        <div><dt>Forecast issued</dt><dd>{anchor ? formatDuration(anchor.anchor_seconds) : 'Not selected'}</dd></div>
        <div><dt>Original horizon</dt><dd>{anchor ? `${formatDuration(anchor.horizon_start_exclusive_seconds)} → ${formatDuration(anchor.horizon_end_inclusive_seconds)}` : '—'}</dd></div>
        {timing && <div><dt>{timing.title}</dt><dd>{timing.value}</dd></div>}
      </dl>
      {timing?.expired && <p className="forecast-expired">No newer retained forecast · original estimate retained.</p>}
      {!score && !baseline && <label className="representation-control">Probability representation
        <GunmetalSelect label="Probability representation" value={representation}
          onChange={next => onRepresentationChange(next as ProbabilityRepresentation)}
          options={[{value:'calibrated',label:'Calibrated'},{value:'raw',label:'Retained raw'}]} />
      </label>}
      <p className="risk-note">{score ? 'Lower MAP produces a higher ranking score. Retained score at its original anchor.'
        : 'Original issued forecast: onset after issue time and through horizon end. No continuous refresh between anchors.'}</p>
      {events && outcomes && <ForecastOutcome outcome={resolved} mode={replay.replayMode} />}
    </div>
  </section>;
}
