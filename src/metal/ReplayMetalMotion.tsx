import { createContext, useContext, useEffect, useState, useSyncExternalStore } from 'react';
import type { ReactNode } from 'react';
import type { ReplayStore } from '../replay/store';

const Motion = createContext(false);
export const METAL_SETTLE_MS = 500;
export function useReplayMetalMotion() { return useContext(Motion); }

// Presentation-only scheduling: no prediction, outcome or physiology values are read.
export function ReplayMetalMotion({ store, children }: { store: ReplayStore; children: ReactNode }) {
  const playing = useSyncExternalStore(store.subscribe, () => store.getSnapshot().isPlaying, () => false);
  const [recentInput, setRecentInput] = useState(false);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const input = () => {
      setRecentInput(true);
      clearTimeout(timer);
      timer = setTimeout(() => { setRecentInput(false); }, METAL_SETTLE_MS);
    };
    const events = ['pointermove', 'pointerdown', 'keydown'] as const;
    for (const name of events) window.addEventListener(name, input, { passive: true });
    return () => { clearTimeout(timer); for (const name of events) window.removeEventListener(name, input); };
  }, []);
  return <Motion.Provider value={playing || recentInput}>{children}</Motion.Provider>;
}
