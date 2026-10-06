import { GunmetalSelect } from '../metal/GunmetalSelect';
import { InteractiveMetalTab } from '../metal/InteractiveMetalTab';
import { caseLabel } from '../data/types';
import type { CaseOrdinal, PresentationCase } from '../data/types';

export function CaseSelector({ cases, value, onChange, onRandom }: {
  cases: PresentationCase[]; value: CaseOrdinal;
  onChange: (ordinal: CaseOrdinal) => void; onRandom: () => void;
}) {
  return <div className="selector-field case-field">
    <label htmlFor="case-select" className="field-label">Held-out operation</label>
    <div className="case-controls">
      <GunmetalSelect id="case-select" label="Held-out operation" value={value} onChange={next => onChange(next as CaseOrdinal)}
        options={cases.map(item => ({value:item.ordinal,label:caseLabel(item.ordinal)}))} />
      <InteractiveMetalTab><button className="button random-button" onClick={onRandom}>
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M2 4h2.5l7 8H14m-3-3 3 3-3 3M2 12h2.5l7-8H14m-3-3 3 3-3 3" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
        </svg>Random Test Case
      </button></InteractiveMetalTab>
    </div>
    <span className="field-note">{cases.length} whole operations · safe case ordinals</span>
  </div>;
}
