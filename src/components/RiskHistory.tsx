import { useMemo } from 'react';
import { formatDuration, modelLabels } from '../data/types';
import type { ModelId } from '../data/types';
import type { PredictionLoadState } from '../data/casePredictions';
import { useReplay } from '../replay/store';
import type { ReplayStore } from '../replay/store';
import { verifiedCasePredictions } from '../predictions/anchorLookup';
import { historyBounds, historySamples } from '../predictions/history';
import type { ProbabilityRepresentation } from '../predictions/representations';
import { RiskHistoryPlot } from '../charts/RiskHistoryPlot';

export function RiskHistory({ modelId, store, predictions, representation }: {
  modelId: ModelId; store: ReplayStore; predictions: PredictionLoadState; representation: ProbabilityRepresentation;
}) {
  const replay = useReplay(store);
  const data = verifiedCasePredictions(predictions, replay);
  const score = modelId === 'current_map';
  const [first, last] = data ? historyBounds(data.windows, replay.visibleWindow[0],
    replay.replayMode === 'live' ? Math.floor(replay.currentSeconds) : replay.visibleWindow[1]) : [0, 0];
  const samples = useMemo(() => data ? historySamples(data.windows, first, last, modelId, representation) : undefined,
    [data, first, last, modelId, representation]);
  const view = useMemo(() => samples ? { ...samples, domain: replay.visibleWindow } : undefined, [samples, replay.visibleWindow]);
  const available = samples?.values.filter(value => value !== null).length ?? 0;
  return <section className="history-panel" aria-labelledby="history-heading" aria-describedby="risk-history-summary">
    <div className="panel-heading"><h2 id="history-heading">{score ? 'Score history' : 'Risk history'}</h2>
      <span className="panel-status">{score ? 'Retained −MAP score · mmHg' : 'Retained issued probabilities · 0–100%'}</span></div>
    {view ? <div className="risk-history-body"><RiskHistoryPlot view={view} store={store} />
      <p id="risk-history-summary" className="history-caption">{modelLabels[modelId]} · {available} available of {samples!.anchors.length} retained anchors in view · discrete points, no connecting curve.
        {' '}Time since surgical start {formatDuration(replay.visibleWindow[0])}–{formatDuration(replay.visibleWindow[1])}.</p>
    </div> : <div className="history-empty"><span id="risk-history-summary">No {score ? 'score' : 'forecast'} history loaded</span><span className="history-rail" aria-hidden="true" /></div>}
    {replay.replayMode === 'review' && view && <p className="historical-future-note">Historical future · not available at replay time</p>}
  </section>;
}
