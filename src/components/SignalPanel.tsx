import { useLayoutEffect, useMemo } from 'react';
import { useCaseSignals } from '../data/caseSignals';
import type { SignalLoader } from '../data/caseSignals';
import type { PresentationCase } from '../data/types';
import { formatDuration } from '../data/types';
import { UPlotAdapter } from '../charts/UPlotAdapter';
import { availability } from '../charts/synchronizedAxes';
import { ReplayStore, useReplay } from '../replay/store';
import { exactSample, sliceSignals } from '../replay/viewport';
import { MAP_REFERENCE_LABEL } from '../charts/signalOptions';
import '../styles/charts.css';

const signals = [['map', 'MAP', 'mmHg'], ['hr', 'HR', 'bpm'], ['spo2', 'SpO₂', '%'], ['etco2', 'ETCO₂', 'mmHg']] as const;
export function SignalPanel({ selectedCase, loader, store: suppliedStore }: {
  selectedCase: PresentationCase; loader?: SignalLoader; store?: ReplayStore;
}) {
  const localStore = useMemo(() => new ReplayStore(selectedCase.ordinal, selectedCase.durationSeconds), [selectedCase]);
  const store = suppliedStore ?? localStore;
  const replay = useReplay(store);
  const state = useCaseSignals(selectedCase, loader);
  const data = state.status === 'ready' ? state.data : undefined;
  useLayoutEffect(() => { if (data) store.activate(); }, [data, store]);
  useLayoutEffect(() => () => localStore.dispose(), [localStore]);
  const first = Math.floor(replay.visibleWindow[0]);
  const last = Math.floor(replay.replayMode === 'live' ? Math.min(replay.currentSeconds, replay.visibleWindow[1]) : replay.visibleWindow[1]);
  // Reuse the slices until an exported-second boundary changes. The shared domain may be fractional.
  const slices = useMemo(() => data ? sliceSignals(data, first, last) : undefined, [data, first, last]);
  const viewport = useMemo(() => slices ? { ...slices, domain: replay.visibleWindow } : undefined, [slices, replay.visibleWindow]);
  const status = state.status === 'loading' ? 'Loading monitor traces…'
    : state.status === 'unavailable' ? 'Monitor traces unavailable' : 'Verified monitor traces';
  return <section className="signal-panel" aria-labelledby="signal-heading" aria-busy={state.status === 'loading'}>
    <div className="panel-heading"><h2 id="signal-heading">Physiology</h2><span className="panel-status" role="status">{status}</span></div>
    <div className="signal-stack real-signals">
      {signals.map(([id, name, unit]) => {
        const summary = viewport ? availability(viewport.values[id]) : undefined;
        const description = summary
          ? `${summary.available} of ${summary.total} displayed exported observations available; ${summary.gaps} missing-data gaps. Null observations remain breaks. ${replay.replayMode === 'live' ? 'Only observations at or before replay time are displayed.' : 'Historical future · not model input.'}`
          : status;
        return <div className="signal-row real-signal-row" key={id} role="group"
          aria-label={`${name}, ${unit}, ${summary ? summary.available > 0 ? 'observations available' : 'no observations in displayed span' : 'no observations loaded'}`}
          aria-describedby={`summary-${id}`}>
          <div className="signal-label"><span>{name}</span><small>{unit}</small>
            <b aria-label={`${name} at ${formatDuration(replay.currentSeconds)}: ${data && exactSample(data, id, replay.currentSeconds) !== null ? `${exactSample(data, id, replay.currentSeconds)} ${unit}` : 'unavailable'}`}>
              {data ? exactSample(data, id, replay.currentSeconds) ?? '—' : '—'}</b></div>
          <div className="chart-and-summary">
            {viewport ? <UPlotAdapter key={selectedCase.ordinal} id={id} viewport={viewport} store={store} />
              : <div className="trace-unavailable" aria-hidden="true"><span className="axis-rail" /></div>}
            <span id={`summary-${id}`} className="sr-only">{name}, {unit}. {description}</span>
            {id === 'map' && <span className="map-reference-label">{MAP_REFERENCE_LABEL}</span>}
            {id !== 'etco2' && summary && summary.missing > 0 && <span className="gap-summary">{summary.missing} missing samples · {summary.gaps} {summary.gaps === 1 ? 'gap' : 'gaps'}</span>}
          </div>
        </div>;
      })}
    </div>
    {replay.replayMode === 'review' && <p className="historical-future-note">Historical future · not model input</p>}
    <div className="signal-footnote time-footnote">
      <span>Time since surgical start <b className="visible-domain">{viewport ? `${formatDuration(viewport.domain[0])} – ${formatDuration(viewport.domain[1])}` : '—'}</b></span>
      <span>Reserved channels <b>SBP</b> <b>DBP</b></span>
    </div>
  </section>;
}
