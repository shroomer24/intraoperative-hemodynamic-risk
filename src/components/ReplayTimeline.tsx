import { MoltenReplayTimeline } from '../metal/MoltenReplayTimeline';
import { formatDuration } from '../data/types';
import type { ReplayState, ReplayStore } from '../replay/store';

export function ReplayTimeline({ store, state }: { store: ReplayStore; state: ReplayState }) {
  return <div className="replay-timeline">
    <label htmlFor="surgical-timeline">Time since surgical start</label>
    <MoltenReplayTimeline store={store} state={state} />
    <output htmlFor="surgical-timeline">{formatDuration(state.currentSeconds)} <span>/</span> {formatDuration(state.durationSeconds)}</output>
  </div>;
}
