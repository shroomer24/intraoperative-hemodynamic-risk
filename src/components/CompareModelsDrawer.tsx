import { memo, useMemo, useRef, useState } from 'react';
import { InteractiveMetalTab } from '../metal/InteractiveMetalTab';
import type { PredictionLoadState } from '../data/casePredictions';
import type { PredictionWindow } from '../data/predictionValidation';
import { formatDuration, modelLabels } from '../data/types';
import type { ModelId } from '../data/types';
import { anchorAt, verifiedCasePredictions } from '../predictions/anchorLookup';
import { alignedComparison } from '../predictions/alignedComparison';
import { formatForecast, representationLabel } from '../predictions/representations';
import type { ProbabilityRepresentation } from '../predictions/representations';
import { forecastAge } from '../outcomes/forecastAge';
import { useReplay } from '../replay/store';
import type { ReplayStore } from '../replay/store';

interface ComparisonProps {
  store: ReplayStore;
  predictions: PredictionLoadState;
  modelId: ModelId;
  representation: ProbabilityRepresentation;
  onModelChange: (model: ModelId) => void;
}

// Timing can publish at 10 Hz. Rows only update when their anchor, representation or selection changes.
const AlignedRows = memo(function AlignedRows({ anchor, modelId, representation, onModelChange }: {
  anchor: PredictionWindow;
} & Pick<ComparisonProps, 'modelId' | 'representation' | 'onModelChange'>) {
  const rows = useMemo(() => alignedComparison(anchor, representation), [anchor, representation]);
  const row = ({ id, value }: typeof rows[number]) => {
    const selected = id === modelId;
    const score = id === 'current_map';
    const baseline = id === 'prevalence';
    const description = representationLabel(id, representation);
    return <InteractiveMetalTab key={id} active={selected}>
      <button type="button" className={`comparison-model-row${score ? ' comparison-score-row' : ''}`}
        aria-pressed={selected} aria-label={`Select ${modelLabels[id]}: ${value ? formatForecast(value) : 'Not available at aligned anchor'}. ${description}${selected ? '. Selected model' : ''}`}
        data-model={id} data-selected={selected} data-query-position={anchor.query_position}
        data-anchor-seconds={anchor.anchor_seconds} data-horizon-start={anchor.horizon_start_exclusive_seconds}
        data-horizon-end={anchor.horizon_end_inclusive_seconds} data-value={value?.value}
        data-representation={value?.representation} data-available={Boolean(value)} onClick={() => onModelChange(id)}>
        <span className="comparison-model-label">{modelLabels[id]}
          {score && <small>Untransformed −latest MAP · Ranking score · mmHg</small>}
          {baseline && <small>Fixed TRAINING baseline · retained raw probability</small>}
        </span>
        {!score && <span className="comparison-probability-track" aria-hidden="true" data-available={Boolean(value)}>
          {value && <span className="comparison-probability-fill" style={{ width: `${value.value * 100}%` }} />}
        </span>}
        <strong className={value ? 'comparison-value' : 'comparison-unavailable'}>{value ? formatForecast(value) : 'Not available at aligned anchor'}</strong>
        <span className="comparison-selected" aria-hidden="true">{selected ? 'Selected' : ''}</span>
      </button>
    </InteractiveMetalTab>;
  };
  return <div className="comparison-values" data-aligned-query-position={anchor.query_position}>
    <section className="comparison-group" aria-labelledby="comparison-score-heading">
      <h3 id="comparison-score-heading">Current MAP score</h3>{row(rows[0])}
      <p className="comparison-score-note">Lower MAP produces a higher ranking score.</p>
    </section>
    <section className="comparison-group" aria-labelledby="comparison-learned-heading">
      <h3 id="comparison-learned-heading">Learned model probabilities</h3>{rows.slice(1, 6).map(row)}
    </section>
    <section className="comparison-group comparison-baseline" aria-labelledby="comparison-baseline-heading">
      <h3 id="comparison-baseline-heading">Baseline / technical</h3>{row(rows[6])}
    </section>
  </div>;
});

function ComparisonContent({ store, predictions, ...props }: ComparisonProps) {
  const replay = useReplay(store);
  const data = verifiedCasePredictions(predictions, replay);
  const anchor = data ? anchorAt(data.windows, replay.currentSeconds) : undefined;
  const timing = forecastAge(anchor, replay.currentSeconds);
  return <div className="aligned-comparison" data-case-ordinal={replay.caseOrdinal}
    data-anchor-seconds={anchor?.anchor_seconds} data-query-position={anchor?.query_position}
    data-horizon-end={anchor?.horizon_end_inclusive_seconds} data-representation={props.representation}>
    {!anchor ? <p className="comparison-empty">{data ? 'No aligned retained forecast at this replay time.'
      : predictions.status === 'unavailable' ? 'Aligned frozen forecasts unavailable.' : 'Verifying aligned frozen forecasts…'}</p>
      : <>
        <div className="comparison-context">
          <div><span className="eyebrow">Aligned forecast</span><p>Issued <b>{formatDuration(anchor.anchor_seconds)}</b></p>
            <p>Horizon <b>{formatDuration(anchor.horizon_start_exclusive_seconds)} → {formatDuration(anchor.horizon_end_inclusive_seconds)}</b></p></div>
          <div className="comparison-timing"><span>{timing!.title}</span><b>{timing!.value}</b>
            {timing?.expired && <p>No newer retained forecast · original estimates retained.</p>}</div>
        </div>
        <AlignedRows anchor={anchor} {...props} />
        <p className="compare-note">Representation: {props.representation === 'calibrated' ? 'Calibrated' : 'Retained raw'} · learned probabilities.
          Current MAP remains a ranking score; TRAINING prevalence retains its raw baseline.</p>
      </>}
  </div>;
}

export function CompareModelsDrawer(props: ComparisonProps) {
  const [open, setOpen] = useState(false);
  const toggle = useRef<HTMLButtonElement>(null);
  return <section className={`compare-panel ${open ? 'is-open' : ''}`} onKeyDown={event => {
    if (event.key === 'Escape' && open) { setOpen(false); toggle.current?.focus({ preventScroll: true }); }
  }}>
    <InteractiveMetalTab active={open}><button type="button" className="compare-toggle" ref={toggle}
      aria-expanded={open} aria-controls="compare-content" onClick={() => setOpen(value => !value)}>
      <span>Compare models</span><span className="compare-hint">Aligned estimates at one anchor</span>
      <svg className="chevron" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="none"><path d="m4 6 4 4 4-4" stroke="currentColor" strokeWidth="1.3" /></svg>
    </button></InteractiveMetalTab>
    <div id="compare-content" className="compare-content" hidden={!open} role="region" aria-label="Model comparison">
      {open && <ComparisonContent {...props} />}
    </div>
  </section>;
}
