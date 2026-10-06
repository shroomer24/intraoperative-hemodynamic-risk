import { useLayoutEffect, useState } from 'react';
import type { RefObject } from 'react';
import type { PredictionLoadState } from '../data/casePredictions';
import { useReplay } from '../replay/store';
import type { ReplayStore } from '../replay/store';
import { resolveDemoAnchor } from './preset';

export function DemoStart({ store, predictions, pending }: {
  store: ReplayStore; predictions: PredictionLoadState; pending: RefObject<boolean>;
}) {
  const replay = useReplay(store);
  const [unavailable, setUnavailable] = useState(false);
  useLayoutEffect(() => {
    if (!pending.current || !replay.ready) return;
    if (replay.currentSeconds !== 0 || replay.isPlaying || replay.replayMode !== 'live') { pending.current = false; return; }
    if (predictions.status !== 'ready') return;
    pending.current = false;
    const seconds = resolveDemoAnchor(predictions.data);
    if (seconds === undefined || predictions.ordinal !== replay.caseOrdinal) { setUnavailable(true); return; }
    store.seek(seconds);
  }, [store, predictions, replay.ready, replay.caseOrdinal, replay.currentSeconds, replay.isPlaying, replay.replayMode, pending]);
  return unavailable ? <p role="status" className="demo-unavailable">The curated frozen anchor is unavailable. Replay remains paused.</p> : null;
}
