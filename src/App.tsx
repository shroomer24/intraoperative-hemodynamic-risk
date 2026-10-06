import { CaseContext } from './case-context/CaseContext';
import { InteractiveMetalTab } from './metal/InteractiveMetalTab';
import { useEffect, useMemo, useRef, useState } from 'react';
import { loadShellMetadata, randomCase } from './data/loader';
import { caseLabel, formatDuration } from './data/types';
import type { CaseOrdinal, ModelId, ReplayMode, ShellMetadata } from './data/types';
import { Header } from './components/Header';
import { SignalPanel } from './components/SignalPanel';
import { RiskPanel } from './components/RiskPanel';
import { RiskHistory } from './components/RiskHistory';
import { PlaybackControls } from './components/PlaybackControls';
import { CompareModelsDrawer } from './components/CompareModelsDrawer';
import { ReplayStore } from './replay/store';
import { forecastState, outcomeState, useCaseForecasts } from './data/forecastOutcomes';
import { useCaseEvents } from './data/caseEvents';
import { EventLane } from './components/EventLane';
import type { ProbabilityRepresentation } from './predictions/representations';
import { DemoStart } from './demo/DemoStart';
import { ReplayMetalMotion } from './metal/ReplayMetalMotion';
import { demoPreset, isDemoPath } from './demo/preset';

let metadataPromise: Promise<ShellMetadata> | undefined;
function metadataOnce(): Promise<ShellMetadata> {
  metadataPromise ??= loadShellMetadata().catch(error => {
    metadataPromise = undefined;
    throw error;
  });
  return metadataPromise;
}

export default function App({ loadData = metadataOnce }: {
  loadData?: () => Promise<ShellMetadata>;
}) {
  const [data, setData] = useState<ShellMetadata>();
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);
  const [demo] = useState(() => isDemoPath(window.location.pathname));
  const demoPending = useRef(demo);
  const [modelId, setModelId] = useState<ModelId>(demo ? demoPreset.modelId : 'logistic_map');
  const [caseOrdinal, setCaseOrdinal] = useState<CaseOrdinal>(demo ? demoPreset.caseOrdinal : 'case-001');
  const [representation, setRepresentation] = useState<ProbabilityRepresentation>('calibrated');
  const [mode, setMode] = useState<ReplayMode>('live');
  const selected = data?.cases.find(item => item.ordinal === caseOrdinal);
  const forecasts = useCaseForecasts(selected);
  const predictions = useMemo(() => forecastState(forecasts), [forecasts]);
  const outcomes = useMemo(() => outcomeState(forecasts), [forecasts]);
  const events = useCaseEvents(selected);
  const replay = useMemo(() => selected ? new ReplayStore(selected.ordinal, selected.durationSeconds) : undefined, [selected]);
  useEffect(() => () => replay?.dispose(), [replay]);
  const changeCase = (ordinal: CaseOrdinal) => { demoPending.current = false; replay?.pause(); setMode('live'); setCaseOrdinal(ordinal); };
  const changeMode = (value: ReplayMode) => { replay?.setMode(value); setMode(value); };
  useEffect(() => {
    let active = true;
    setError(false);
    loadData().then(result => { if (active) setData(result); })
      .catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, [loadData, retry]);

  if (!data) return (
    <main className="startup-shell">
      <span className="eyebrow">Surgical replay</span>
      <h1>Intraoperative Hemodynamic Risk</h1>
      <p role={error ? 'alert' : 'status'}>{error
        ? 'Presentation metadata could not be verified. No case data have been displayed.'
        : 'Verifying frozen presentation metadata…'}</p>
      {error && <InteractiveMetalTab><button className="button" onClick={() => setRetry(value => value + 1)}>Retry metadata</button></InteractiveMetalTab>}
    </main>
  );
  const selectedCase = data.cases.find(item => item.ordinal === caseOrdinal)!;
  const model = data.models.find(item => item.id === modelId)!;
  return (
    <ReplayMetalMotion store={replay!}><div className="app-shell">
      <a className="skip-link" href="#workspace">Skip to workspace</a>
      <Header metadata={data} modelId={modelId} onModelChange={setModelId}
        selectedCase={selectedCase} onCaseChange={changeCase}
        onRandomCase={() => changeCase(randomCase(data.cases, caseOrdinal))}
        mode={mode} onModeChange={changeMode} />
      <main id="workspace" tabIndex={-1}>
        {demo && <DemoStart store={replay!} predictions={predictions} pending={demoPending} />}
        <div className="workspace-heading">
          <div className="workspace-case"><div className="workspace-case-heading"><span className="eyebrow">Operation workspace</span>
            <span className="operation-name">{caseLabel(caseOrdinal)}</span></div>
            <CaseContext ordinal={caseOrdinal} />
          </div>
          <span className="operation-duration">Operation length <b>{formatDuration(selectedCase.durationSeconds)}</b></span>
        </div>
        <div className="workspace-frame">
          <div className="workspace-grid">
            <SignalPanel key={selectedCase.ordinal} selectedCase={selectedCase} store={replay!} />
            <RiskPanel model={model} store={replay!} events={events} outcomes={outcomes} predictions={predictions} representation={representation} onRepresentationChange={setRepresentation} />
          </div>
          <RiskHistory modelId={modelId} store={replay!} predictions={predictions} representation={representation} />
          <EventLane store={replay!} predictions={predictions} events={events} outcomes={outcomes} />
          <PlaybackControls store={replay!} predictions={predictions} />
        </div>
        <CompareModelsDrawer store={replay!} predictions={predictions} modelId={modelId}
          representation={representation} onModelChange={setModelId} />
      </main>
      <footer className="provenance-footer">
        <span><span className="status-dot" aria-hidden="true" /> Frozen presentation · metadata verified</span>
        <span>{data.caseCount} held-out operations <span className="footer-divider">/</span> {data.predictionWindowCount.toLocaleString('en-US')} eligible windows</span>
        <span className="provenance-code" title={`Model lock SHA-256: ${data.modelLockHash}`}>LOCK {data.modelLockHash.slice(0, 8)}</span>
      </footer>
    </div></ReplayMetalMotion>
  );
}
