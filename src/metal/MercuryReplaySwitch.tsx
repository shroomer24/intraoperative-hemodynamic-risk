import type { ReplayMode } from '../data/types';
import { PaperSurface } from './PaperSurface';

export function MercuryReplaySwitch({ value, onChange }: { value: ReplayMode; onChange: (value: ReplayMode) => void }) {
  return <div className="mode-options mercury-switch" data-tone="gunmetal" data-engine="paper" data-mode={value}>
    <span className="mercury-thumb" aria-hidden="true"><PaperSurface kind="mercury" /><span className="mercury-core" /><span className="mercury-sheen" /></span>
    {([['live', 'Live Replay'], ['review', 'Review']] as const).map(([mode, label]) =>
      <label key={mode} className="mode-option">
        <input type="radio" name="replay-mode" value={mode} checked={value === mode} onChange={() => onChange(mode)} />
        <span>{label}</span>
      </label>)}
  </div>;
}
