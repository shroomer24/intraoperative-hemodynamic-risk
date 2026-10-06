import { GunmetalSelect } from '../metal/GunmetalSelect';
import { InteractiveMetalTab } from '../metal/InteractiveMetalTab';
import { useEffect } from 'react';
import { useReplay } from '../replay/store';
import type { ReplayStore } from '../replay/store';
import { playbackSpeeds } from '../replay/clock';
import type { PlaybackSpeed } from '../replay/clock';
import { pauseWhenHidden, replayKeyboard } from '../replay/visibility';
import type { PredictionLoadState } from '../data/casePredictions';
import { anchorNavigation, verifiedCasePredictions } from '../predictions/anchorLookup';
import { ReplayTimeline } from './ReplayTimeline';

export function PlaybackControls({ store, predictions = { status: 'loading' } }: { store: ReplayStore; predictions?: PredictionLoadState }) {
  const state = useReplay(store);
  const data = verifiedCasePredictions(predictions, state);
  const navigation = anchorNavigation(data?.windows ?? [], state.currentSeconds);
  useEffect(() => {
    const visibility = pauseWhenHidden(store);
    const keyboard = replayKeyboard(store);
    return () => { visibility(); keyboard(); };
  }, [store]);
  return <section className="playback-panel" aria-label="Historical playback controls"
    data-playing={state.isPlaying} data-current-seconds={state.currentSeconds} data-replay-publications={store.diagnostics.publications}
    data-replay-frames={store.diagnostics.frames}>
    <div className="transport-controls">
      <InteractiveMetalTab><button className="button transport-button" disabled={!navigation.previous}
        title={data ? 'Previous retained forecast anchor' : 'Available when frozen forecasts are loaded'}
        onClick={() => { if (navigation.previous) store.seek(navigation.previous.anchor_seconds); }}>‹ Previous anchor</button></InteractiveMetalTab>
      <InteractiveMetalTab active={state.isPlaying}><button className="button play-button" disabled={!state.ready || state.currentSeconds >= state.durationSeconds && !state.isPlaying}
        onClick={() => store.toggle()} aria-label={state.isPlaying ? 'Pause' : 'Play'}>
        <span aria-hidden="true">{state.isPlaying ? 'Ⅱ' : '▶'}</span>{state.isPlaying ? 'Pause' : 'Play'}</button></InteractiveMetalTab>
      <InteractiveMetalTab><button className="button transport-button" disabled={!navigation.next}
        title={data ? 'Next retained forecast anchor' : 'Available when frozen forecasts are loaded'}
        onClick={() => { if (navigation.next) store.seek(navigation.next.anchor_seconds); }}>Next anchor ›</button></InteractiveMetalTab>
      <InteractiveMetalTab><button className="button" disabled={!state.ready} onClick={() => store.seek(0)}>Start</button></InteractiveMetalTab>
      <InteractiveMetalTab><button className="button" disabled={!state.ready} onClick={() => store.seek(state.durationSeconds)}>End</button></InteractiveMetalTab>
    </div>
    <label className="speed-control">Speed
      <GunmetalSelect label="Playback speed" value={String(state.playbackSpeed)} onChange={next => store.setSpeed(Number(next) as PlaybackSpeed)}
        options={playbackSpeeds.map(speed => ({value:String(speed),label:`${speed}×`}))} />
    </label>
    <ReplayTimeline store={store} state={state} />
    <p id="replay-shortcuts" className="replay-shortcuts">Outside controls: Space play/pause · Home/End start/end · ←/→ 1 second · Shift + ←/→ 10 seconds. Replay pauses when the tab is hidden.</p>
  </section>;
}
