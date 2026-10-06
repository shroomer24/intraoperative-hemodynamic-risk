import { MercuryReplaySwitch } from '../metal/MercuryReplaySwitch';
import type { ReplayMode } from '../data/types';

export function ReplayModeSelector({ value, onChange }: {
  value: ReplayMode; onChange: (value: ReplayMode) => void;
}) {
  return <fieldset className="selector-field mode-field">
    <legend className="field-label">View</legend>
    <MercuryReplaySwitch value={value} onChange={onChange} />
    <p className="field-note mode-helper">{value === 'live'
      ? 'Future physiology and outcomes hidden as the surgery progresses.'
      : 'Historical future and outcomes shown · not model input.'}</p>
  </fieldset>;
}
