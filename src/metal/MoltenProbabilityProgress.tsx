import { PaperSurface } from './PaperSurface';

export function MoltenProbabilityProgress({ probability, representation }: {
  probability?: number; representation: 'calibrated_probability' | 'raw_probability';
}) {
  if (probability === undefined) return <div className="molten-progress molten-unavailable" aria-hidden="true" data-available="false" data-tone="gunmetal" />;
  // Input is the already-validated retained fraction. No normalization, threshold or easing of data.
  const percent = probability * 100;
  return <div className="molten-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
    aria-label={`${representation === 'calibrated_probability' ? 'Calibrated' : 'Raw'} frozen forecast probability ${percent} percent`}
    data-probability={probability} data-representation={representation} data-available="true" data-tone="gunmetal" data-engine="paper">
    <span className="molten-fill" style={{ clipPath: `inset(0 ${100 - percent}% 0 0)` }} aria-hidden="true"><PaperSurface kind="probability" /></span>
  </div>;
}
